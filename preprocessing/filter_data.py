"""
Extract, align, filter, and aggregate goalkeeper tracking data from match archives.

For every game package found in ``data_dir``, this script performs the following pipeline:
    1. Unpacking: Extracts the ``<game_id> Tracking Data.7z`` archive into a temporary directory.
    2. Loading & Sorting: Identifies and sorts ``scrubbed.samples.joints`` and ``scrubbed.samples.ball`` 
       files chronologically by match half and minute.
    3. Synchronization:
       - Converts joint timestamps to absolute time (seconds from game start).
       - Synchronizes specific goalkeeper skeletons with the ball's position.
       - Handles missing ball data by holding the last known position for up to 5 seconds.
    4. Filtering:
       - Retains frames only when the ball is within the goalkeeper's defending final third.
       - Discards sequences shorter than ``MIN_SEQUENCE_FRAMES`` (50 frames).
    5. Aggregation: Groups valid sequences by Goalkeeper ID (UEFA ID).
    6. Export: Saves the processed data to ``<game_id>_gk_joints.json`` in ``save_dir``.
    7. Cleanup: Removes the extracted temporary files to save space.

Prerequisite for manual inspection (optional):
The raw files inside the archives are JSON-formatted but may have extensions like .joints or .ball.
To pretty-print them for readability:
```
for f in *.joints; do jq '.' "$f" > "${f%.joints}.json"; done
```

Run: python data_filtering.py
"""

import os
import re
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone
import py7zr
import shutil


LEFT_THIRD = 0
RIGHT_THIRD = 1
MIN_SEQUENCE_FRAMES = 50


def get_half_minute_extra(file_path: Path, game_id: str):
    """
    Returns (half, minute, extra) where extra is an int or None.
    Matches names like:
      ..._1_41.football.samples.ball
      ..._2_90_1.football.samples.joints
    """
    name = Path(file_path).name
    # ensure we capture the final `_half_minute(_extra)` directly before the .football... suffix
    pattern = re.compile(rf'2024_3_{game_id}_(\d+)_(\d+)(?:_(\d+))?(?=\.football\.samples\.(?:joints|ball)$)')
    m = pattern.search(name)
    if not m:
        return None
    half = int(m.group(1))
    minute = int(m.group(2))
    extra = int(m.group(3)) if m.group(3) is not None else None
    return half, minute, extra


def _to_epoch(ts):
    s = ts
    # handle trailing Z (UTC) for fromisoformat
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        # fallback: numeric string (secs or ms)
        nv = float(ts)
        if nv > 1e12:
            nv /= 1000.0
        return nv
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def ball_third(x):
    if x is None:
        return None
    if x <= left_final_third:
        return LEFT_THIRD
    if x >= right_final_third:
        return RIGHT_THIRD
    return None


def keeper_defending_third(joint_frame):
    mid_hip = joint_frame.get("midHip")
    x = mid_hip[0]
    if x <= left_final_third:
        return LEFT_THIRD
    if x >= right_final_third:
        return RIGHT_THIRD
    return None


if __name__ == "__main__":
    # Expected directory structure: data/euro2024/<game_id>/
    # These paths are resolved relative to this script to ensure they work 
    # whether run from preprocessing/ or the project root.
    script_dir = Path(__file__).parent
    data_dir = script_dir / "../data/euro2024"
    save_dir = script_dir / "../data/filtered"
    tmp_dir = data_dir / "tmp"
    games = []

    pitch_length = 105.0
    half_length = pitch_length / 2.0
    third_span = pitch_length / 3.0
    left_final_third = -half_length + third_span
    right_final_third = half_length - third_span

    pattern = re.compile(r"^(2036\d+)$")

    for game_dir in os.listdir(data_dir):
        match = pattern.match(game_dir)
        if match:
            games.append((data_dir / game_dir, match.group(1)))

    # per game loop
    for (game_path, game_id) in games:
        output_file = save_dir / f"{game_id}_gk_joints.json"
        if output_file.exists():
            print(f"Skipping game {game_id}: Output file already exists.")
            continue

        print(f"Game {game_id}: Extracting tracking data zip...")

        with py7zr.SevenZipFile(game_path / f"{game_id}_tracking_data.7z", 'r') as archive:
            archive.extractall(path=tmp_dir)

        extracted_game_dir = next(
            (p for p in tmp_dir.iterdir() if p.is_dir() and game_id in p.name),
            None,
        )
        if extracted_game_dir is None:
            raise FileNotFoundError(f"No extracted directory like '{game_id}_*' inside {tmp_dir}")

        print(f"Game {game_id}: Tracking data zip file extracted...")
        
        dir_path_joints = extracted_game_dir / "scrubbed.samples.joints"
        dir_path_ball = extracted_game_dir / "scrubbed.samples.ball"
        json_files_joints = []
        json_files_ball = []

        for file in os.listdir(dir_path_joints):
            json_files_joints.append(dir_path_joints / file)

        for file in os.listdir(dir_path_ball):
            json_files_ball.append(dir_path_ball / file)

        # Make sure we access files and add entries by order of time
        def file_sort_key(f):
            res = get_half_minute_extra(f, game_id)
            if not res: return (999, 999, 999)
            h, m, e = res
            return (h, m, e if e is not None else -1)

        json_files_joints.sort(key=file_sort_key)
        json_files_ball.sort(key=file_sort_key)

        ball = []
        gk_samples = {}
        gk_pending = {}
        game_start_time = None

        print(f"Game {game_id}: Reading all json files and compiling data points...")

        # per minute of game loop
        for index, (joint_file, ball_file) in enumerate(zip(json_files_joints, json_files_ball)):
            h1, m1, e1 = get_half_minute_extra(joint_file, game_id)
            h2, m2, e2 = get_half_minute_extra(ball_file, game_id)

            # Ensure we are processing the same minute
            if (h1, m1, e1) != (h2, m2, e2):
                print(f"Warning: Mismatch in files processing. Joints: {joint_file.name}, Ball: {ball_file.name}")

            half, minute, extra = h1, m1, e1

            # print(f"Half: {half}, Minute: {minute}, Extras: {extra}")
            try:
                with open(joint_file, "r") as f:
                    joints_data = json.load(f)

                with open(ball_file, "r") as f:
                    ball_data = json.load(f)
            
            except json.JSONDecodeError as e:
                print(f"ERROR: Corrupted JSON file. Skipping minute {minute}.")
                print(f"Files: {joint_file.name}, {ball_file.name}")
                print(f"Details: {e}")
                
                # Break any ongoing sequences because we are missing a chunk of time
                for gk_id, buffer in gk_pending.items():
                    if len(buffer) >= MIN_SEQUENCE_FRAMES:
                        gk_samples.setdefault(gk_id, []).append(list(buffer))
                    buffer.clear()
                continue

            start_time = joints_data["time"]["startUTC"]
            if index == 0:  # first minute of the game
                game_start_time = joints_data["time"]["startUTC"]
                game_start_epoch = _to_epoch(game_start_time)

            ball_info = ball_data["samples"]["ball"]

            file_epoch = _to_epoch(start_time)
            file_offset = file_epoch - game_start_epoch

            # collects the ball x coordinates for each frame
            ball_samples = []
            for sample in ball_info:
                sample_time = float(sample.get("time", 0))
                x_coord = float(sample.get("pos")[0])
                ball_samples.append({"time": sample_time, "x": x_coord})

            ball_idx = 0

            for joint_sample in joints_data["samples"]["people"]:
                if joint_sample["role"]["name"] != "Goalkeeper":
                    continue

                gk_id = joint_sample["personId"]["uefaId"]
                joints = joint_sample.setdefault("joints", {})[0]

                jt_offset = float(joints.get("time", 0))
                joints["time"] = file_offset + jt_offset

                # Advance ball_idx to the correct window
                # We use a small epsilon to handle floating point jitter
                epsilon = 1e-4
                while (
                    ball_idx + 1 < len(ball_samples)
                    and ball_samples[ball_idx + 1]["time"] <= jt_offset + epsilon
                ):
                    ball_idx += 1
                
                # Check if we found a match
                if ball_idx < len(ball_samples):
                    b_sample = ball_samples[ball_idx]
                    
                    if abs(b_sample["time"] - jt_offset) <= epsilon:
                        ball_x = b_sample["x"]
                    else:
                        # No matching ball frame found (Gap in data, likely Out of Bounds)
                        # Use the last known ball position if it's recent enough (< 5.0s)
                        if jt_offset - b_sample["time"] < 5.0:
                            ball_x = b_sample["x"]
                        else:
                            ball_x = None
                else:
                    # No ball samples available at all or index out of bounds
                    ball_x = None

                ball_zone = ball_third(ball_x)
                keeper_zone = keeper_defending_third(joints)
                # gk_pending allows to save sequences that span across multiple files (i.e. different minutes)
                buffer = gk_pending.setdefault(gk_id, [])      # reference to value list inside dict

                # accumulate sequences where the ball stays in the keeper's final third
                # Note: when switching from 1st half to 2nd half, the ball goes to the 
                #       middle of the pitch and so the sequence will be broken according to this code
                if ball_zone is not None and keeper_zone is not None and ball_zone == keeper_zone:
                    buffer.append(joints)
                else: # interrupted sequence
                    # add finished sequences that are large enough to gk_samples
                    if len(buffer) >= MIN_SEQUENCE_FRAMES:
                        gk_samples.setdefault(gk_id, []).append(list(buffer))
                    buffer.clear()

        # check for in-progress sequence of frames, add only if large enough
        for gk_id, buffer in gk_pending.items():
            if len(buffer) >= MIN_SEQUENCE_FRAMES:
                gk_samples.setdefault(gk_id, []).append(list(buffer))
        
        print(f"Dumping contents into json file for game {game_id}...")

        with open(save_dir / f"{game_id}_gk_joints.json", "w") as f:
            json.dump(gk_samples, f, ensure_ascii=False)

        # remove extracted data from tmp as not needed anymore
        for path in tmp_dir.iterdir():
            if path.is_dir() and game_id in path.name:
                shutil.rmtree(path)
        
        print(f"Done with game {game_id}!")

    print("Data successfully saved!")