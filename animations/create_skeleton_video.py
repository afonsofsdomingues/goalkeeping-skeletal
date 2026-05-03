"""
Generate a 3D animated video of a player's skeleton from tracking data.

Usage:
    python create_skeleton_video.py <player_id> [--start START] [--length LENGTH] [--output FILENAME] [--input_dir INPUT_DIR]

Example:
    python create_skeleton_video.py 1cc2530b-3412-4ec8-742a-08d958b7d325 --start 0 --length 300
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import concurrent.futures
from tqdm.auto import tqdm
import functools
import orjson
import shutil

from utils import retrieve_tracking_data_for_player 

# PLAYER IDS
# 1cc2530b-3412-4ec8-742a-08d958b7d325 -- SOMMER
# 9c263634-b6fb-46a0-7386-08d958b7d325 -- DONNARUMMA

# Map integer IDs to readable names
BODY_PART_MAPPING = {
    0: "lAnkle",
    1: "lEar",
    2: "lElbow",
    3: "lEye",
    4: "lHip",
    5: "lKnee",
    6: "lShoulder",
    7: "lWrist",
    8: "neck",
    9: "nose",
    10: "rAnkle",
    11: "rEar",
    12: "rElbow",
    13: "rEye",
    14: "rHip",
    15: "rKnee",
    16: "rShoulder",
    17: "rWrist",
    18: "midHip",
    19: "lBigToe",
    20: "rBigToe",
    21: "lSmallToe",
    22: "rSmallToe",
    23: "lHeel",
    24: "rHeel",
    25: "lThumb",
    26: "lPinky",
    27: "rThumb",
    28: "rPinky"
}

# reverse mapping (Name -> ID)
BODY_PART_IDS = {v: k for k, v in BODY_PART_MAPPING.items()}

# skeletal connections
BONES = [
        (11, 13), (13, 9), (9, 3), (3, 1), (9, 8), (8, 6), (8, 16), (16, 12), 
        (12, 17), (17, 28), (17, 27), (6, 2), (2, 7), (7, 25), (7, 26), 
        (8, 18), (18, 14), (14, 15), (15, 10), (10, 24), (10, 20), (20, 22), 
        (18, 4), (4, 5), (5, 0), (0, 23), (0, 19), (19, 21)
]


def process_single_file(file_path: Path, player_id: str):
    try:
        with file_path.open("rb") as f:
            data = orjson.loads(f.read())
        samples = data.get('samples', {}).get('people', [])
        player_samples = retrieve_tracking_data_for_player(samples, player_id)
        
        if not player_samples:
            return None
        
        # Return the whole dictionary of joints for this frame
        # Structure: [{'lAnkle': [x,y,z], ...}, ...]
        return [ps['joints'][0] for ps in player_samples]
    except Exception:
        return None

def get_player_sequence(directory: Path, player_id: str):
    files = sorted([f for f in directory.iterdir() if f.is_file() and not f.name.startswith('.')])
    worker = functools.partial(process_single_file, player_id=player_id)

    print("Loading data...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(worker, files), total=len(files), leave=False))
    
    flattened = []
    for r in results:
        if r is not None:
            flattened.extend(r)
    return flattened

# --- DATA CONVERSION ---
def convert_dict_to_indexed_list(frames_of_dicts):
    """
    Converts [{'lAnkle': [x,y,z]}, ...] 
    Into [ [[x,y,z], [x,y,z], ...], ... ]
    Where the inner list is ordered by BODY_PART_IDS (0 to 28)
    """
    num_joints = max(BODY_PART_IDS.values()) + 1
    formatted_frames = []

    for frame_dict in frames_of_dicts:
        # Create a list of Nones
        frame_list = [None] * num_joints
        
        for name, coords in frame_dict.items():
            if name in BODY_PART_IDS:
                idx = BODY_PART_IDS[name]
                # Ensure 3D
                if len(coords) >= 3:
                    frame_list[idx] = coords[:3]
                elif len(coords) == 2:
                    # Pad Z with 0 if missing
                    frame_list[idx] = [coords[0], coords[1], 0.0]
        
        formatted_frames.append(frame_list)
    
    return formatted_frames

# --- ANIMATION FUNCTION ---
def animate_skeleton(frames_list, edges):
    print("Initializing Animation...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Calculate Dynamic Limits
    all_points = []
    for frame in frames_list:
        for point in frame:
            if point is not None:
                all_points.append(point)
    
    all_points = np.array(all_points)
    
    # Handle empty data case
    if len(all_points) == 0:
        print("Error: No valid points found to plot.")
        return None

    min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
    min_y, max_y = all_points[:, 1].min(), all_points[:, 1].max()
    min_z, max_z = all_points[:, 2].min(), all_points[:, 2].max()

    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    mid_z = (min_z + max_z) / 2

    max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2
    padding = max_range * 1.2

    ax.set_xlim(mid_x - padding, mid_x + padding)
    ax.set_ylim(mid_y - padding, mid_y + padding)
    ax.set_zlim(mid_z - padding, mid_z + padding)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1]) 

    # 2. Setup Plot Elements
    scat = ax.scatter([], [], [], c='red', s=15)
    lines = [ax.plot([], [], [], 'k-', linewidth=2)[0] for _ in edges]

    def update(frame_idx):
        joints = frames_list[frame_idx]
        clean_joints = [j for j in joints if j is not None]
        
        if not clean_joints: 
            return lines + [scat]

        xs = [j[0] for j in clean_joints]
        ys = [j[1] for j in clean_joints]
        zs = [j[2] for j in clean_joints]
        scat._offsets3d = (xs, ys, zs)

        for line, (start_idx, end_idx) in zip(lines, edges):
            if (start_idx < len(joints) and end_idx < len(joints) and 
                joints[start_idx] is not None and joints[end_idx] is not None):
                
                p1 = joints[start_idx]
                p2 = joints[end_idx]

                line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                line.set_3d_properties([p1[2], p2[2]])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        return lines + [scat]

    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=range(len(frames_list)), 
        interval=50, 
        blit=False
    )
    
    return ani

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D Skeleton Animation")
    
    # Positional Argument
    parser.add_argument("player_id", type=str, help="Player UUID")
    
    # Optional Arguments
    parser.add_argument("--start", type=int, default=0, help="Frame index to start the video at (default: 0)")
    parser.add_argument("--length", type=int, default=300, help="Number of frames to render (default: 300). Set to -1 for all remaining.")
    parser.add_argument("--output", type=str, default="skeleton_video.mp4", help="Output filename")
    parser.add_argument("--input_dir", type=str, default="../data/euro2024/2036198/20240629_SUI-2-0-ITA_2036198/2036198_Switzerland_Italy/scrubbed.samples.joints", help="Directory containing tracking data")
    
    args = parser.parse_args()

    # Paths
    joints_dir = Path(args.input_dir)


    # 1. Load ALL Data first (to ensure we have the correct frame indices)
    # Note: We load everything because we need to know where "frame 5000" actually is.
    print(f"Loading data for {args.player_id}...")
    raw_data = get_player_sequence(joints_dir, args.player_id)
    
    total_frames = len(raw_data)
    print(f"Total frames available: {total_frames}")

    if not raw_data:
        print("No data found for this player.")
        exit()

    # 2. Validation & Slicing
    start_frame = args.start
    
    # Handle length logic
    if args.length == -1:
        end_frame = total_frames
    else:
        end_frame = start_frame + args.length

    # Safety checks
    if start_frame >= total_frames:
        print(f"Error: Start frame ({start_frame}) is larger than total frames ({total_frames}).")
        exit()
    
    # Perform the slice
    print(f"Slicing data from frame {start_frame} to {end_frame}...")
    sliced_data = raw_data[start_frame:end_frame]

    # 3. Convert Data Format
    print("Formatting data for animation...")
    formatted_frames = convert_dict_to_indexed_list(sliced_data)

    # 4. Animate
    # Note: The animation function recalculates camera bounds based on THIS slice only,
    # so the player will be perfectly centered.
    ani = animate_skeleton(formatted_frames, BONES)

    if ani:
        save_location = Path("../data/videos") / args.output
        save_location.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving animation to {save_location}...")
        
        if shutil.which("ffmpeg") is None:
            print("\nERROR: ffmpeg not found in PATH.")
            print("To generate MP4 files, you need ffmpeg installed and available in your system PATH.")
            print("Install via Conda: conda install -c conda-forge ffmpeg")
            print("Or download from https://ffmpeg.org/ and add /bin folder to PATH.")
            exit(1)

        writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(save_location, writer=writer)
        print("Done!")