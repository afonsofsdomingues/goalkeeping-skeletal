"""
Data cleaning utils for goalkeeper skeletal data.

Usage
    python preprocessing/cleaning_utils.py <player_uuid>
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from sibling directories
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
from tqdm.auto import tqdm
import functools
import orjson
from animations.utils import retrieve_tracking_data_for_player 

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


OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- helper function for single file processing ---
def process_single_file(file_path: Path, player_id: str, key_type: str):
    """
    Reads a single file and extracts data ONLY for the specific player.
    Returns None if player is not found in that frame.
    """
    try:
        with file_path.open("rb") as f: # orjson reads bytes
            data = orjson.loads(f.read())
        
        # Look through the people in this specific frame
        samples = data.get('samples', {}).get('people', [])
        
        # Note: Depending on your utils implementation, ensure this returns
        # a list of dicts for that specific player
        player_samples = retrieve_tracking_data_for_player(samples, player_id) 
        
        if not player_samples:
            return None

        if key_type == 'centroid':
            # Return list of positions [x, y]
            return [ps['centroid'][0]['pos'] for ps in player_samples]
        elif key_type == 'joints':
            # Return list of midHip [x, y], dropping Z
            return [ps['joints'][0] for ps in player_samples]
        
        return None 
    except Exception:
        return None

def get_player_sequence(directory: Path, player_id: str, key_type: str):
    """
    Parallelized loader that preserves order.
    """
    files = sorted([
        f for f in directory.iterdir() 
        if f.is_file() and not f.name.startswith('.')
    ])

    results = []
    
    # Partial function passes the specific player_id to the worker
    worker = functools.partial(process_single_file, player_id=player_id, key_type=key_type)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(worker, files), 
            total=len(files), 
            desc=f"Processing {directory.name}",
            leave=False,
            mininterval=1.0 # Prevents screen flashing
        ))
    
    flatten_results = []
    for minute in results:
        if minute is not None:
            flatten_results.extend(minute)
    
    return flatten_results

def visualise_com_deviation(player_id: str):
    print(f"Extracting data for player: {player_id}")
    
    base_path = Path("data/euro2024/2036198/20240629_SUI-2-0-ITA_2036198/2036198_Switzerland_Italy")
    centroids_dir = base_path / "scrubbed.samples.centroids"
    joints_dir = base_path / "scrubbed.samples.joints"

    player_com = get_player_sequence(centroids_dir, player_id, 'centroid')
    player_joints = get_player_sequence(joints_dir, player_id, 'joints')
    player_hips_2d = [player_joint["midHip"][:2] for player_joint in player_joints]

    print("Data extraction complete. Aligning data...")

    # Ensure lengths match before plotting
    min_len = min(len(player_com), len(player_hips_2d))
    player_com = player_com[:min_len]
    player_hips_2d = player_hips_2d[:min_len]

    print(f"Total frames processed: {min_len}")

    if min_len == 0:
        print("No matching data found for this player ID.")
        return

    valid_com = np.array(player_com)
    valid_hips = np.array(player_hips_2d)
    
    deviations = np.linalg.norm(valid_com - valid_hips, axis=1)

    mean_dev = np.mean(deviations)
    
    plt.figure(figsize=(12, 6))
    plt.plot(deviations, label='Deviation', linewidth=0.8, alpha=0.8)
    plt.axhline(mean_dev, color='r', linestyle='--', label=f'Mean: {mean_dev:.2f}')
    
    plt.title(f'CoM vs Mid-Hip Deviation (Player {player_id[:8]}...)')
    plt.xlabel('Frame Index (synced)')
    plt.ylabel('Distance (units)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / f"{player_id}_centroid_deviation.png"
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        transparent=False
    )
    plt.close()
    print(f"Plot saved to: {save_path}")

def visualise_bone_lengths(player_id: str):
    """
    Computes bone lengths, plots histograms, and flags outliers (IQR method).
    """
    print(f"Extracting data for player: {player_id}")
    
    base_path = Path("data/euro2024/2036198/20240629_SUI-2-0-ITA_2036198/2036198_Switzerland_Italy")
    joints_dir = base_path / "scrubbed.samples.joints"

    player_joints_list = get_player_sequence(joints_dir, player_id, 'joints')
    
    if not player_joints_list:
        print("No data found.")
        return

    print(f"Data loaded: {len(player_joints_list)} frames. Converting to NumPy matrix...")

    num_frames = len(player_joints_list)
    num_joints = max(BODY_PART_IDS.values()) + 1 
    joints_matrix = np.full((num_frames, num_joints, 3), np.nan)

    for t, frame_data in enumerate(player_joints_list):
        for name, coords in frame_data.items():
            if name in BODY_PART_IDS:
                idx = BODY_PART_IDS[name]
                if len(coords) >= 3:
                    joints_matrix[t, idx, :] = coords[:3]
                elif len(coords) == 2:
                    joints_matrix[t, idx, :2] = coords

    bone_plot_dir = OUTPUT_DIR / "bone_histograms" / player_id
    bone_plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a log file for outliers
    log_file_path = bone_plot_dir / "outlier_report.txt"
    
    # We open the file once and append to it inside the loop
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Outlier Report for Player {player_id}\n")
        log_file.write("="*50 + "\n\n")

        print("Matrix conversion complete. Computing lengths and analyzing...")

        for start_id, end_id in tqdm(BONES, desc="Analyzing Bones"):
            
            start_name = BODY_PART_MAPPING.get(start_id, str(start_id))
            end_name = BODY_PART_MAPPING.get(end_id, str(end_id))
            bone_label = f"{start_name}_to_{end_name}"

            p1 = joints_matrix[:, start_id, :] 
            p2 = joints_matrix[:, end_id, :]   

            lengths = np.linalg.norm(p1 - p2, axis=1)

            valid_mask = ~np.isnan(lengths)
            valid_lengths = lengths[valid_mask]
            valid_indices = np.arange(num_frames)[valid_mask]

            if len(valid_lengths) == 0:
                continue

            # --- STATISTICAL OUTLIER DETECTION (IQR Method) ---
            q1 = np.percentile(valid_lengths, 25)
            q3 = np.percentile(valid_lengths, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            
            # Find indices where length is outside bounds
            outlier_mask = (valid_lengths < lower_bound) | (valid_lengths > upper_bound)
            outlier_frames = valid_indices[outlier_mask]
            outlier_values = valid_lengths[outlier_mask]

            # --- LOGGING ---
            if len(outlier_frames) > 0:
                log_file.write(f"BONE: {bone_label}\n")
                log_file.write(f"  Total Outliers: {len(outlier_frames)} / {len(valid_lengths)} frames\n")
                log_file.write(f"  Valid Range: {lower_bound:.3f} to {upper_bound:.3f}\n")
                
                # Write details (first 20 only to keep file readable, or all if you prefer)
                for frame, val in zip(outlier_frames, outlier_values):
                    log_file.write(f"    Frame {frame}: {val:.4f}\n")
                log_file.write("-" * 30 + "\n")

            # --- PLOTTING ---
            plt.figure(figsize=(10, 6))
            
            n, bins, patches = plt.hist(
                valid_lengths, 
                bins=50, 
                color='skyblue', 
                edgecolor='black', 
                alpha=0.7, 
                label='Normal Data'
            )
            
            for patch, bin_left in zip(patches, bins[:-1]):
                if bin_left < lower_bound or bin_left > upper_bound:
                    patch.set_facecolor('salmon')

            plt.axvline(lower_bound, color='red', linestyle='-', linewidth=1.5, label='IQR Lower Bound')
            plt.axvline(upper_bound, color='red', linestyle='-', linewidth=1.5, label='IQR Upper Bound')
            plt.axvline(np.median(valid_lengths), color='green', linestyle='--', label='Median')

            plt.title(f"{bone_label} (Outliers: {len(outlier_frames)})")
            plt.xlabel("Length")
            plt.ylabel("Count")
            plt.legend()
            
            plt.savefig(bone_plot_dir / f"{bone_label}.png", dpi=150, bbox_inches='tight')
            plt.close()

    print(f"Analysis complete.")
    print(f"Plots saved to: {bone_plot_dir}")
    print(f"Detailed Outlier Report saved to: {log_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot CoM and bone deviations for a specific player.")
    
    parser.add_argument(
        "player_id", 
        type=str, 
        help="The UUID of the player (e.g., 1cc2530b-3412...)"
    )

    args = parser.parse_args()

    visualise_com_deviation(args.player_id)
    visualise_bone_lengths(args.player_id)