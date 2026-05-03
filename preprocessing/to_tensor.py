"""
Convert and preprocess filtered goalkeeper joint data into ST-GCN ready tensors.

This script performs the following steps:
    * Reads filtered JSON data from ``data/filtered``;
    * Splits data into Training and Validation sets based on predefined GK IDs;
    * Computes a canonical skeleton from all sequences for body size normalization;
    * Preprocesses each sequence: aligns to goal and normalizes body size;
    * Depending on mode:
        - Default: Maps to NTU-25 skeleton, resamples to 30fps, converts to Kinect coordinates;
        - Native (--native): Keeps native 29-joint skeleton, 25fps, standard coordinates;
    * Detects and extracts action segments (sequences with significant movement) and idle segments;
    * Resizes all extracted clips to a fixed length of 50 frames;
    * Generates two data streams:
        - Position: Rotated to face forward, centered on spine/hip;
        - Motion: Computed from coordinates (Kinect or Native) without rotation;
    * Formats data into (N, C, T, V, M) tensors and saves them along with metadata
      to ``data/tensor`` (or ``data/tensor_native``).

Output Shape: (N, C, T, V, M)
    where C=3 (dims), T=50 (frames), V=25 (or 29 for native) (joints), M=2 (or 1 for native) (max bodies)
"""

import json
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path

# Handle imports: if running as script vs module
if __name__ == "__main__" and __package__ is None:
    # Running as script (python preprocessing/to_tensor.py)
    sys.path.append(str(Path(__file__).parent.parent))
    from preprocessing.processing_utils import (
        align_to_goal, 
        compute_canonical_skeleton, 
        normalize_body_size,
        detect_activity_segments,
        extract_clips,
        map_to_ntu25,
        compute_motion,
        center_ntu_sequence,
        rotate_sequence_around_y_axis,
        convert_to_kinect_coords,
        resample_sequence,
        map_to_native,
        rotate_native_sequence_around_z
    )
else:
    # Running as module (python -m preprocessing.to_tensor)
    from preprocessing.processing_utils import (
        align_to_goal, 
        compute_canonical_skeleton, 
        normalize_body_size,
        detect_activity_segments,
        extract_clips,
        map_to_ntu25,
        compute_motion,
        center_ntu_sequence,
        rotate_sequence_around_y_axis,
        convert_to_kinect_coords,
        resample_sequence,
        map_to_native,
        rotate_native_sequence_around_z
    )

# NOTE: A GK sample is defined as the set of joint positions of a single game for a goalkeeper position of a team.
# FOR A 80-20 SPLIT, WE NEED ABOUT 21 GK SAMPLES OUT OF POSSIBLE 102 (51 games x 2 goalkeepers)
# We are using the data from the goalkeepers of teams that did not make it to the knockout stage to 
# constitute the validation set. Each of these teams played 3 matches, so we have 24 GK samples in total (~23.5%). 
# Note that multiple goalkeepers can be used throughout a single match (and thus we have multiple goalkeepers 
# being part of the same GK sample) and that different goalkeepers can be used in different games.

# List of specific Goalkeeper IDs to use for Validation
VALIDATION_GK_IDS = [
    "250047105",   # Angus Gunn, Scotland
    "102420",      # Péter Gulácsi, Hungary
    "250042625",   # Dominik Livaković, Croatia
    "250046901",   # Thomas Strakosha, Albania
    "250042207",   # Predrag Rajković, Serbia
    "108501",      # Wojciech Szczęsny, Poland
    "250040573",   # Łukasz Skorupski, Poland
    "250089824",   # Andriy Lunin, Ukraine
    "250113444",   # Anatoliy Trubin, Ukraine
    "250055112",   # Jindřich Staněk, Czechia
    "250117491",   # Matěj Kovář, Czechia
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', action='store_true', help="Keep native structure (25fps, 29 joints)")
    args = parser.parse_args()

    # Use script location to resolve paths robustly
    script_dir = Path(__file__).parent
    input_dir = script_dir / "../data/filtered"
    
    # Adjust output dir based on mode
    suffix = "_native" if args.native else "_ntu"
    output_dir = script_dir / f"../data/tensor{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        sys.exit(1)

    train_sequences = []
    val_sequences = []
    
    train_gks = set()
    val_gks = set()

    print("Loading filtered data...")
    # Iterate over files to capture Game ID
    for file_path in input_dir.glob("*_gk_joints.json"):
        # Filename format {game_id}_gk_joints.json
        game_id = file_path.name.split("_")[0]
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

                for gk_id, sequences in data.items():
                    if not sequences: continue
                    
                    # If GK is in VALIDATION_GK_IDS then Val, else Train
                    is_val = gk_id in VALIDATION_GK_IDS
                    
                    if is_val:
                        val_sequences.extend(sequences)
                        val_gks.add(gk_id)
                    else:
                        train_sequences.extend(sequences)
                        train_gks.add(gk_id)
                        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    if not train_sequences and not val_sequences:
        print("No data found.")
        sys.exit(1)

    print(f"Training Sequences: {len(train_sequences)} (from {len(train_gks)} GKs)")
    print(f"Validation Sequences: {len(val_sequences)} (from {len(val_gks)} GKs)")
    
    overlap = train_gks.intersection(val_gks)
    assert not overlap

    # We use ALL data to compute the canonical (average) skeleton
    print("Aligning sequences for canonical skeleton computation...")
    all_sequences = train_sequences + val_sequences
    all_aligned_sequences = []
    
    for seq in all_sequences:
        aligned_seq = [align_to_goal(frame) for frame in seq]
        all_aligned_sequences.append(aligned_seq)
            
    print("Computing canonical skeleton...")
    canonical_lengths = compute_canonical_skeleton(all_aligned_sequences)
    
    # Process Train and Val separately
    def process_dataset(sequences, name):
        print(f"Processing {name} dataset...")
        pos_segments = []
        motion_segments = []
        metadata_list = []
        
        for seq in sequences:
            # Align to Pitch (Goal Logic) - Keep for both
            aligned_seq = [align_to_goal(frame) for frame in seq]
            
            if args.native:
                # NATIVE: Normalize body size, raw mapping, 25fps
                normalized_seq = [normalize_body_size(frame, canonical_lengths) for frame in aligned_seq]
                seq_arr = np.array([map_to_native(frame) for frame in normalized_seq]) # (T, 29, 3)
                target_fps = 25
                
                # Detect & Extract
                action_segments, idle_segments = detect_activity_segments(seq_arr, fps=target_fps)
                clips_data = extract_clips(seq_arr, action_segments, idle_segments, fps=target_fps, extract_idle=True)
                
                for clip, meta in clips_data:
                    # Resize to 50 frames
                    T_clip, V, C = clip.shape
                    flat = clip.reshape(T_clip, V*C)
                    res = cv2.resize(flat.astype(np.float32), (V*C, 50), interpolation=cv2.INTER_LINEAR)
                    clip_50 = res.reshape(50, V, C)
                    
                    # NATIVE PROCESSING
                    
                    # Motion (Standard Coords)
                    motion_seq = compute_motion(clip_50)
                    
                    # Position (Absolute Coords, but Body Aligned)
                    # Isolate the Root (midHip is native index 0)
                    #    We want to rotate the body AROUND the root, not around the pitch origin (0,0).
                    #    This prevents "swinging" the GK to a different pitch location during rotation.
                    root = clip_50[:, 0:1, :]    # 0:1 to preserve dimensions of tensor
                    
                    body_centered = clip_50 - root    # center the body locally
                    body_rotated = rotate_native_sequence_around_z(body_centered)
                    pos_seq = body_rotated + root
                    
                    pos_segments.append(pos_seq)
                    motion_segments.append(motion_seq)
                    metadata_list.append(meta)

            else:
                # NTU ADAPTED: Normalize, Map 25, Resample 30fps
                normalized_seq = [normalize_body_size(frame, canonical_lengths) for frame in aligned_seq]
                ntu_seq_original = np.array([map_to_ntu25(frame) for frame in normalized_seq])
                
                # Resample to 30fps
                ntu_seq_30fps = resample_sequence(ntu_seq_original)
                target_fps = 30
                
                # Detect & Extract
                action_segments, idle_segments = detect_activity_segments(ntu_seq_30fps, fps=target_fps)
                clips_data = extract_clips(ntu_seq_30fps, action_segments, idle_segments, fps=target_fps, extract_idle=True)
                
                for clip, meta in clips_data:
                    # Resample to 50 frames
                    T_clip, V, C = clip.shape
                    flat = clip.reshape(T_clip, V*C)
                    res = cv2.resize(flat.astype(np.float32), (V*C, 50), interpolation=cv2.INTER_LINEAR)
                    clip_50 = res.reshape(50, V, C)
                    
                    # Convert to Kinect Coords (Y-up for NTU)
                    clip_kinect = convert_to_kinect_coords(clip_50)
                    
                    # Motion (Kinect Coords)
                    motion_seq = compute_motion(clip_kinect)
                    
                    # Position (Kinect Coords + Rotated + Centered)
                    clip_pos_kinect = rotate_sequence_around_y_axis(clip_kinect)
                    pos_seq = center_ntu_sequence(clip_pos_kinect)
                    
                    pos_segments.append(pos_seq)
                    motion_segments.append(motion_seq)
                    metadata_list.append(meta)
                
        if not pos_segments:
            return None, None, None
            
        # Convert to numpy arrays (N, T, V, C)
        pos_data = np.array(pos_segments)
        motion_data = np.array(motion_segments)
        
        # Transpose to (N, C, T, V)
        pos_data = pos_data.transpose(0, 3, 1, 2)
        motion_data = motion_data.transpose(0, 3, 1, 2)
        
        # Add Max Bodies Dimension
        if args.native:
            # Native Mode (Full Training from Scratch):
            # We only have 1 GK. Using M=1 saves memory/compute.
            pos_data = np.expand_dims(pos_data, axis=-1)       # (N, C, T, V, 1)
            motion_data = np.expand_dims(motion_data, axis=-1) # (N, C, T, V, 1)
        else:
            # NTU Mode (Finetuning):
            # Must match pre-trained model expectation (M=2).
            # Even if we only have 1 body, the weights expect this padding structure.
            pos_data = np.expand_dims(pos_data, axis=-1)
            motion_data = np.expand_dims(motion_data, axis=-1)
            pos_data = np.concatenate([pos_data, np.zeros_like(pos_data)], axis=-1)       # (N, C, T, V, 2)
            motion_data = np.concatenate([motion_data, np.zeros_like(motion_data)], axis=-1) # (N, C, T, V, 2)
        
        return pos_data, motion_data, metadata_list

    train_pos, train_motion, train_meta = process_dataset(train_sequences, "Train")
    val_pos, val_motion, val_meta = process_dataset(val_sequences, "Validation")
    
    print("Saving files...")
    if train_pos is not None:
        np.save(output_dir / "train_position.npy", train_pos.astype(np.float32))
        np.save(output_dir / "train_motion.npy", train_motion.astype(np.float32))
        with open(output_dir / "train_metadata.json", "w") as f:
            json.dump(train_meta, f, indent=2)
        print(f"Saved Train: {train_pos.shape}")
        
    if val_pos is not None:
        np.save(output_dir / "val_position.npy", val_pos.astype(np.float32))
        np.save(output_dir / "val_motion.npy", val_motion.astype(np.float32))
        with open(output_dir / "val_metadata.json", "w") as f:
            json.dump(val_meta, f, indent=2)
        print(f"Saved Val: {val_pos.shape}")
    
    print("Done!")