"""
Visualize a sample sequence from the processed ST-GCN tensors.

Usage:
    python visualize_tensor_sample.py <index> [--val] [--output FILENAME]

Example:
    python visualize_tensor_sample.py 0 --val
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import shutil
import json

# NTU RGB+D 25 Joint order (0-based)
BONES_NTU = [
    (0, 1), (1, 20), (20, 2), (2, 3), # Torso
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22), # Left Arm
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24), # Right Arm
    (0, 12), (12, 13), (13, 14), (14, 15), # Left Leg
    (0, 16), (16, 17), (17, 18), (18, 19) # Right Leg
]

# Native 29-Joint order (0-based)
# 0: midHip, 1: neck, 2: nose, 3: lEye, 4: rEye, 5: lEar, 6: rEar
# 7: lShoulder, 8: rShoulder, 9: lElbow, 10: rElbow, 11: lWrist, 12: rWrist
# 13: lThumb, 14: lPinky, 15: rThumb, 16: rPinky
# 17: lHip, 18: rHip, 19: lKnee, 20: rKnee, 21: lAnkle, 22: rAnkle
# 23: lBigToe, 24: lSmallToe, 25: lHeel, 26: rBigToe, 27: rSmallToe, 28: rHeel
BONES_NATIVE = [
    (0, 1), (0, 17), (0, 18),   # midHip -> neck, lHip, rHip
    (1, 2), (1, 7), (1, 8),     # neck -> nose, lShoulder, rShoulder
    (2, 3), (2, 4),             # nose -> lEye, rEye
    (3, 5),                     # lEye -> lEar
    (4, 6),                     # rEye -> rEar
    (7, 9),                     # lShoulder -> lElbow
    (9, 11),                    # lElbow -> lWrist
    (11, 13), (11, 14),         # lWrist -> lThumb, lPinky
    (8, 10),                    # rShoulder -> rElbow
    (10, 12),                   # rElbow -> rWrist
    (12, 15), (12, 16),         # rWrist -> rThumb, rPinky
    (17, 19),                   # lHip -> lKnee
    (19, 21),                   # lKnee -> lAnkle
    (21, 23), (21, 25),         # lAnkle -> lBigToe, lHeel
    (23, 24),                   # lBigToe -> lSmallToe
    (18, 20),                   # rHip -> rKnee
    (20, 22),                   # rKnee -> rAnkle
    (22, 26), (22, 28),         # rAnkle -> rBigToe, rHeel
    (26, 27)                    # rBigToe -> rSmallToe
]

def animate_skeleton(frames, output_path, bones, fps=20, swap_axis=True):
    """
    frames: numpy array of shape (T, V, C) i.e. (Time, Vertices, Channels/Coords)
    swap_axis: If True, swaps Y and Z axes (assuming Input Y is Up, mapping to Plot Z). 
               If False, assumes Input Z is Up (Native).
    """
    print(f"Initializing Animation with {fps:.2f} FPS...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Swap Y and Z axes for better visualization (Y is vertical in data, map to Z in plot)
    # Data: (X, Y, Z) -> Plot: (X, Z, Y)
    # This makes the skeleton stand "up" in the matplotlib 3d plot (where Z is up)
    frames_viz = frames.copy()
    if swap_axis:
        frames_viz[:, :, [1, 2]] = frames_viz[:, :, [2, 1]] 

    # 1. Calculate Body-Centric Limits
    # Instead of a global bounding box (which makes the person small if they move far),
    # we calculate the max extent of the body across all frames to define a fixed "zoom level" (radius),
    # and then center the camera on the person's centroid in every frame.
    
    # Calculate radius per frame -> max radius
    # For each frame, find bounding box of the body
    max_dims = np.zeros(3)
    
    for frame in frames_viz:
        if np.all(frame == 0): continue # Skip empty frames
        min_vals = frame.min(axis=0)
        max_vals = frame.max(axis=0)
        # Half-dimension of the body in this frame for each axis
        curr_dims = (max_vals - min_vals) / 2.0
        max_dims = np.maximum(max_dims, curr_dims)
    
    max_main_dim = max(max_dims[0], max_dims[2])
    padding_main = max_main_dim * 1.5 
    padding_depth = 0.5 
    
    padding = np.array([padding_main, padding_depth, padding_main])
    padding = np.maximum(padding, 0.1)

    # Initial limits
    ax.set_xlabel('X')
    if swap_axis:
        ax.set_ylabel('Data Z') 
        ax.set_zlabel('Data Y') 
    else:
        ax.set_ylabel('Data Y') 
        ax.set_zlabel('Data Z')
        
    # Calculate uniform scale for X (0) and Height (2) to keep aspect ratio
    max_main_dim = max(max_dims[0], max_dims[2])
    padding_main = max_main_dim * 1.5 
    
    padding_x = max_dims[0] * 1.5
    padding_y = max_dims[1] * 1.5
    padding_z = max_dims[2] * 1.0 # Tight padding for Vertical axis
    
    padding = np.array([padding_x, padding_y, padding_z])
    padding = np.maximum(padding, 0.1)

    # Initial limits
    ax.set_xlabel('X')
    if swap_axis:
        ax.set_ylabel('Data Z') 
        ax.set_zlabel('Data Y') 
    else:
        ax.set_ylabel('Data Y') 
        ax.set_zlabel('Data Z')

    ax.set_xlim(-padding[0], padding[0])
    ax.set_ylim(-padding[1], padding[1])
    ax.set_zlim(-padding[2], padding[2])

    # Restore axes labels
    ax.set_xlabel('X')
    if swap_axis:
        ax.set_ylabel('Z') # Plot Y is Data Z
        ax.set_zlabel('Y') # Plot Z is Data Y (Height)
    else:
        ax.set_ylabel('Y') # Plot Y is Data Y
        ax.set_zlabel('Z') # Plot Z is Data Z (Height)
        
    # Reduce axis granularity (fewer ticks)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    # Set view angle (Elevation, Azimuth)
    ax.view_init(elev=15, azim=45)
    
    # 2. Camera Smoothing Logic
    # Calculate the center of the bounding box for EVERY frame first
    centers = []
    for frame in frames_viz:
        if np.all(frame == 0):
            centers.append(np.array([0,0,0]))
            continue
        min_curr = frame.min(axis=0)
        max_curr = frame.max(axis=0)
        center = (min_curr + max_curr) / 2.0
        centers.append(center)
    centers = np.array(centers) # Shape (T, 3)
    
    # Apply a moving average filter to smooth the camera movement
    # Window size should be odd. 
    window_size = int(fps // 2) * 2 + 1 # e.g. ~1 second smoothing window or half-second
    kernel = np.ones(window_size) / window_size
    
    smoothed_centers = np.zeros_like(centers)
    for i in range(3): # For X, Y, Z coordinates
        # 'same' mode keeps output size, handling edges by zero-padding (not ideal for camera)
        # Using 'valid' shrinks it. 
        # Better: use a simple manual smooth or pad edges with edge value.
        padded = np.pad(centers[:, i], (window_size//2, window_size//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        # Ensure length match (convolve 'valid' on padded returns T)
        if len(smoothed) > len(centers): smoothed = smoothed[:len(centers)]
        smoothed_centers[:, i] = smoothed

    # 3. Setup Plot Elements
    scat = ax.scatter([], [], [], c='red', s=15)
    lines = [ax.plot([], [], [], 'k-', linewidth=2)[0] for _ in bones]

    def update(frame_idx):
        # Shape: (V, 3)
        current_frame = frames_viz[frame_idx]
        
        # Check for zeros/padding if any (though tensor is usually full)
        # Note: In NTU tensor, padded frames might be all zeros.
        if np.all(current_frame == 0):
            return lines + [scat]

        xs = current_frame[:, 0]
        ys = current_frame[:, 1]
        zs = current_frame[:, 2]
        
        # Use smoothed center for the camera
        cx, cy, cz = smoothed_centers[frame_idx]
        
        # Use per-axis padding
        ax.set_xlim(cx - padding[0], cx + padding[0])
        ax.set_ylim(cy - padding[1], cy + padding[1])
        ax.set_zlim(cz - padding[2], cz + padding[2])
        
        scat._offsets3d = (xs, ys, zs)

        for line, (start_idx, end_idx) in zip(lines, bones):
            p1 = current_frame[start_idx]
            p2 = current_frame[end_idx]

            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])

        return lines + [scat]

    interval = 1000 / fps
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=range(len(frames)), 
        interval=interval, 
        blit=False
    )
    
    save_location = Path(output_path)
    save_location.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving animation to {save_location}...")
    
    if shutil.which("ffmpeg") is None:
        print("\nERROR: ffmpeg not found in PATH.")
        return

    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(save_location, writer=writer)
    print("Done!")

def interpolate_sequence(sequence, target_length):
    T, V, C = sequence.shape
    if T == target_length:
        return sequence
        
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_length)
    
    # Flatten spatial dims to (T, V*C) so we can loop over features easily
    flat_seq = sequence.reshape(T, -1) 
    new_flat = np.zeros((target_length, flat_seq.shape[1]))
    
    # Interpolate each coordinate of each joint
    for i in range(flat_seq.shape[1]):
        new_flat[:, i] = np.interp(x_new, x_old, flat_seq[:, i])
        
    return new_flat.reshape(target_length, V, C)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize sample from tensor data")
    parser.add_argument("index", type=int, help="Index of the sequence to visualize")
    parser.add_argument("--val", action="store_true", help="Load from validation set (default: train)")
    parser.add_argument("--native", action="store_true", help="Use native 29-joint skeleton data (default: NTU 25-joint)")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    
    args = parser.parse_args()

    if args.native:
        bones = BONES_NATIVE
        print("Using Native 29-joint configuration.")
        data_dir_name = "tensor_native_old"
    else:
        bones = BONES_NTU
        print("Using NTU 25-joint configuration.")
        data_dir_name = "tensor"

    # Define paths
    data_dir = Path(f"../data/{data_dir_name}")
    if args.val:
        file_name = "val_position.npy"
        meta_name = "val_metadata.json"
    else:
        file_name = "train_position.npy"
        meta_name = "train_metadata.json"
        
    file_path = data_dir / file_name
    meta_path = data_dir / meta_name
    
    if not file_path.exists():
        print(f"Error: File {file_path} not found.")
        exit(1)
        
    print(f"Loading {file_name}...")
    # Shape: (N, C, T, V, M)
    data = np.load(file_path)
    print(f"Dataset shape: {data.shape}")
    
    if args.index < 0 or args.index >= data.shape[0]:
        print(f"Error: Index {args.index} is out of bounds (0-{data.shape[0]-1}).")
        exit(1)

    # Extract Sample
    # We want the first body (M=0), as M=1 is just padding zeros
    sample = data[args.index, :, :, :, 0] # (C, T, V)
    
    # Transpose to (T, V, C) for plotting
    sequence = sample.transpose(1, 2, 0) # (T, V, C) Original Frames

    # Handle Speed / Interpolation
    TARGET_FPS = 30  # Fixed video frame rate
    original_duration = 2.5  # Default
    clip_type = ""

    try:
        with open(meta_path, 'r') as f:
            metadata_list = json.load(f)
            sample_metadata = metadata_list[args.index]
            original_duration = sample_metadata.get('original_duration', 2.5)
            clip_type = sample_metadata.get('clip_type', 'unknown')
            print(f"Metadata found: Original duration {original_duration}s, Type: {clip_type}")
            
    except Exception as e:
        print(f"Metadata warning: {e}. Using default duration {original_duration}s.")

    # Calculate target number of frames to match original duration at TARGET_FPS
    target_length = int(original_duration * TARGET_FPS)
    
    # Avoid empty sequences just in case duration is 0
    if target_length < 10: 
        target_length = 10
    
    print(f"Interpolating from {sequence.shape[0]} frames to {target_length} frames (@ {TARGET_FPS} fps)...")
    sequence_interp = interpolate_sequence(sequence, target_length)

    print(f"Visualizing sample {args.index} with shape {sequence_interp.shape}...")
    
    # Save video
    if args.output is None:
        # Generate default name based on pattern: {split}_{index}_{clip_type}.mp4
        split_name = "val" if args.val else "train"
        safe_clip_type = clip_type if clip_type else "unknown"
        native_tag = "_native" if args.native else ""
        output_filename = f"{split_name}_{args.index}_{safe_clip_type}{native_tag}.mp4"
    else:
        # Use provided name, but append clip_type if not present
        output_filename = args.output
        if clip_type:
            stem = Path(output_filename).stem
            if not stem.endswith(f"_{clip_type}"): 
                suffix = Path(output_filename).suffix
                output_filename = f"{stem}_{clip_type}{suffix}"
            
    output_path = Path("../data/videos") / output_filename
    
    # Check if we need to swap axis (NTU uses Y-up, Native uses Z-up)
    # If Native (Z-up), we do NOT want to swap Y and Z, because Matplotlib Z is already up.
    # If NTU (Y-up), we WANT to swap Y and Z, to put Y data into Z plot axis.
    swap_axis = not args.native
    
    animate_skeleton(sequence_interp, output_path, bones, fps=TARGET_FPS, swap_axis=swap_axis)

