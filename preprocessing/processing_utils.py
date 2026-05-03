"""
Data preprocessing utils for goalkeeper skeletal data.
"""

import numpy as np
import cv2
from pathlib import Path
import numpy as np
from collections import defaultdict

SKELETON_TREE = {
    "midHip": ["neck", "lHip", "rHip"],
    "neck": ["nose", "lShoulder", "rShoulder"],
    "nose": ["lEye", "rEye"],
    "lEye": ["lEar"],
    "rEye": ["rEar"],
    "lShoulder": ["lElbow"],
    "lElbow": ["lWrist"],
    "lWrist": ["lThumb", "lPinky"],
    "rShoulder": ["rElbow"],
    "rElbow": ["rWrist"],
    "rWrist": ["rThumb", "rPinky"],
    "lHip": ["lKnee"],
    "lKnee": ["lAnkle"],
    "lAnkle": ["lBigToe", "lHeel"],
    "lBigToe": ["lSmallToe"],
    "rHip": ["rKnee"],
    "rKnee": ["rAnkle"],
    "rAnkle": ["rBigToe", "rHeel"],
    "rBigToe": ["rSmallToe"]
}


def align_to_goal(skeleton):
    """
    Align goalkeeper so:
    - Origin (0,0) is at the center of the goal line.
    - Y axis points UP the pitch (into the field).
    - X axis points LEFT to RIGHT along the goal line.
    - Z axis points UP (height).
    
    Original Data (assumed):
      - Pitch X: -52.5 (Left Goal) to 52.5 (Right Goal).
      - Pitch Y: Width.
      - Left Goal GK faces +X. Left is +Y.
      - Right Goal GK faces -X. Left is -Y.
    """
    mid_hip_x = skeleton["midHip"][0]
    is_right_goal = mid_hip_x > 0
    
    aligned_sk = {}
    
    for k, v in skeleton.items():
        if k == "time":
            aligned_sk[k] = v
            continue
            
        x, y, z = v
        
        if is_right_goal:
            # Right Goal GK (at X=52.5, facing -X)
            new_y = 52.5 - x
            new_x = y
            new_z = z
        else:
            # Left Goal GK (at X=-52.5, facing +X)
            new_y = x + 52.5
            new_x = -y
            new_z = z
            
        aligned_sk[k] = [new_x, new_y, new_z]
        
    return aligned_sk


def compute_canonical_skeleton(all_sequences):
    """
    Compute the dataset-wide mean length for each bone in the hierarchy.
    """
    bone_sums = defaultdict(float)
    bone_counts = defaultdict(int)
    
    for seq in all_sequences:
        for frame in seq:
            for parent, children in SKELETON_TREE.items():
                p_pos = np.array(frame[parent])
                for child in children:
                    c_pos = np.array(frame[child])
                    dist = np.linalg.norm(c_pos - p_pos)
                    bone_sums[(parent, child)] += dist
                    bone_counts[(parent, child)] += 1
                    
    canonical_lengths = {}
    for k, total in bone_sums.items():
        canonical_lengths[k] = total / bone_counts[k]
            
    return canonical_lengths


def normalize_body_size(frame, canonical_lengths):
    """
    Reconstruct the skeleton using original bone directions but canonical lengths.
    
    Parameters
    ----------
    frame : dict
        Input frame (dictionary of joints).
    canonical_lengths : dict
        Dictionary of average bone lengths.
    """
    new_frame = {}
    if "time" in frame:
        new_frame["time"] = frame["time"]
        
    # Root midHip (used to start the traversal of the bone tree)
    if "midHip" in frame:
        new_frame["midHip"] = frame["midHip"]
    
    # Traverse hierarchy to reconstruct children
    queue = ["midHip"]
    
    while queue:
        parent = queue.pop(0)
        if parent not in SKELETON_TREE: continue
        
        # Parent position in the NEW (normalized) skeleton
        p_pos_new = np.array(new_frame[parent])
        
        # Parent position in the OLD skeleton (for direction)
        p_pos_old = np.array(frame[parent])
        
        for child in SKELETON_TREE[parent]:
            if child not in frame: continue
            
            c_pos_old = np.array(frame[child])
            
            vec = c_pos_old - p_pos_old
            norm = np.linalg.norm(vec)
            direction = vec / norm
            
            length = canonical_lengths.get((parent, child), 0.0)
            
            # Calculate new child position
            c_pos_new = p_pos_new + direction * length
            new_frame[child] = c_pos_new.tolist()
            
            queue.append(child)
            
    return new_frame


def segment_sequence(sequence, window_size=50, stride=25):
    """Slice a sequence into overlapping windows."""
    segments = []
    seq_len = len(sequence)
    assert seq_len >= window_size
    
    for i in range(0, seq_len - window_size + 1, stride):
        window = sequence[i : i + window_size]
        segments.append(window)
        
    return segments


def compute_motion_energy(sequence_array):
    """
    Compute per-frame motion energy.
    sequence_array: (T, V, 3)
    Returns: (T,) array of energy values.
    """
    # Calculate velocity: ||P_t - P_{t-1}||
    # Pad first frame by copying first valid velocity (avoid zero boundary)
    diff = sequence_array[1:] - sequence_array[:-1]
    velocity = np.linalg.norm(diff, axis=2) # (T-1, V), norm of displacement vectors
    energy = np.sum(velocity, axis=1) # (T-1,), sum for all joints to get energy per frame
    return np.pad(energy, (1, 0), mode='edge') # ensure one energy value per frame


def detect_activity_segments(sequence_array, fps=30, min_action_duration=0.4, max_action_duration=6.0, min_gap=0.2, sigma_threshold=1.0, min_energy_threshold=1.65):
    """
    Detect action segments based on adaptive noise estimation (Robust Z-Score).
    
    Parameters:
    - sigma_threshold: How many deviations above the baseline 'idle' noise 
      constitutes an action. 
      Set to 1.0 to ensure that when shuffling (high energy), we only detect 
      significant outliers (saves) and ignore slightly faster shuffles.
    - min_energy_threshold: Minimum absolute energy required to be considered an action.
      Based on dataset analysis:
      - 50th percentile (Median Idle): ~0.96
      - 75th percentile (Active Idle/Walking): ~1.33
      - 95th percentile (Likely Action): ~2.50
      Setting to 1.65 filters out ~85% of all frames (walking/shuffling) while keeping saves.
    """
    energy = compute_motion_energy(sequence_array)
    
    # Smooth energy (0.4s window), ensures that a single noisy frame doesn't trigger 
    # an action, and a single still frame doesn't break an action.
    window_size = int(0.4 * fps)
    if window_size < 1: window_size = 1
    smoothed_energy = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
    
    # Median represents "idle" baseline.
    baseline_median = np.median(smoothed_energy)
    
    # Calculate MAD (Median Absolute Deviation)
    mad = np.median(np.abs(smoothed_energy - baseline_median))
    
    # Convert MAD to equivalent standard deviation (scale factor 1.4826 for normal dist)
    # This gives us a robust measure of "noise"
    sigma = mad * 1.4826
    
    # Enforce a minimum noise floor ("epsilon") to prevent over-sensitivity 
    # when the sequence is mostly constant (MAD approx 0).
    # 0.2 prevents triggering on tiny fluctuations in shuffling/walking.
    sigma = max(sigma, 0.2)
    
    if sigma < 1e-6: sigma = 1e-6   # Avoid division by zero
        
    # Calculate Threshold
    # Threshold is the baseline (standing) + X times the noise level
    adaptive_threshold = baseline_median + (sigma_threshold * sigma)
    
    # Enforce a minimum energy floor to avoid false positives in low-noise sequences
    threshold = max(adaptive_threshold, min_energy_threshold)
    
    is_active = smoothed_energy > threshold
    
    # Morphological closing (fill small gaps), prevents a single dive from being 
    # split into two separate clips
    min_gap_frames = int(min_gap * fps)
    active_indices = np.where(is_active)[0]
    
    raw_segments = []
    if len(active_indices) > 0:
        start = active_indices[0]
        prev = active_indices[0]
        
        for idx in active_indices[1:]:
            if idx - prev > min_gap_frames:
                # Gap detected, start new action
                raw_segments.append((start, prev + 1)) 
                start = idx
            prev = idx
        raw_segments.append((start, prev + 1))
        
    # Filter actions by duration and split long ones
    min_duration_frames = int(min_action_duration * fps)
    max_duration_frames = int(max_action_duration * fps)
    
    valid_action_segments = []
    
    for start, end in raw_segments:
        duration = end - start
        if duration < min_duration_frames:
            continue
            
        if duration > max_duration_frames:
            # Split into chunks
            curr = start
            while curr < end:
                chunk_end = min(curr + max_duration_frames, end)
                if (chunk_end - curr) >= min_duration_frames:
                    valid_action_segments.append((curr, chunk_end))
                curr = chunk_end
        else:
            valid_action_segments.append((start, end))
    
    idle_segments = []
    T = len(sequence_array)
    action_mask = np.zeros(T, dtype=bool)
    for start, end in valid_action_segments:
        action_mask[start:end] = True
    
    idle_indices = np.where(~action_mask)[0]
    
    if len(idle_indices) > 0:
        start = idle_indices[0]
        prev = idle_indices[0]
        
        for idx in idle_indices[1:]:
            if idx - prev > 1:
                idle_segments.append((start, prev + 1))
                start = idx
            prev = idx
        idle_segments.append((start, prev + 1))
    
    return valid_action_segments, idle_segments


def extract_clips(sequence_array, action_segments, idle_segments=None, fps=30, 
                  pre_context=1.0, post_context=1.5, max_clip_frames=300,
                  extract_idle=False, min_idle_duration=4.0, idle_sample_duration=4.0):
    """
    Extract clips with context around action segments AND representative idle clips.
    
    Parameters:
    - max_clip_frames: Maximum number of frames for an extracted clip (default 300 = 10s).
      If context makes it longer, it will be cropped centering on the action.
    
    Returns list of tuples: (clip_array, metadata_dict)
    metadata_dict keys:
        - "clip_type": "action" or "idle"
        - "original_duration": float (seconds)
        - "action_start_in_clip": int (frame index, action only)
        - "action_end_in_clip": int (frame index, action only)
    """
    clips = []
    
    # Pre-calculate energy for peak finding
    energy = compute_motion_energy(sequence_array)
    
    T = len(sequence_array)
    pre_frames = int(pre_context * fps)
    post_frames = int(post_context * fps)
    
    # 1. Extract Action Clips
    for start, end in action_segments:
        # Find the Peak Energy Frame within the detected segment to center alignment
        # This ensures the "Dive" or "Save" is the focal point, not the start of the shuffle.
        segment_energy = energy[start:end]
        if len(segment_energy) > 0:
            peak_offset = np.argmax(segment_energy)
            peak_frame = start + peak_offset
            action_center_idx = peak_frame
        else:
            action_center_idx = (start + end) // 2
            
        # Add context (centered effectively around the action components)
        clip_start = max(0, start - pre_frames)
        clip_end = min(T, end + post_frames)
        
        # Enforce max_clip_frames
        if (clip_end - clip_start) > max_clip_frames:
            # Center the window around the PEAK energy (Action Center)
            half_window = max_clip_frames // 2
            
            new_start = action_center_idx - half_window
            new_end = new_start + max_clip_frames
            
            # Handle boundary conditions
            if new_start < 0:
                new_start = 0
                new_end = min(T, max_clip_frames)
            elif new_end > T:
                new_end = T
                new_start = max(0, T - max_clip_frames)
                
            clip_start = int(new_start)
            clip_end = int(new_end)
        
        clip = sequence_array[clip_start:clip_end]
        
        # Indices of the action core within the clip
        # Clamp to clip bounds in case action was cropped
        action_start_rel = max(0, start - clip_start)
        action_end_rel = min(len(clip), end - clip_start)
        
        metadata = {
            "clip_type": "action",
            "original_duration": float(len(clip) / fps),
            "action_start_in_clip": int(action_start_rel),
            "action_end_in_clip": int(action_end_rel)
        }
        
        clips.append((clip, metadata))
        
    # 2. Extract Idle Clips (if requested)
    if extract_idle and idle_segments:
        min_idle_frames = int(min_idle_duration * fps)
        sample_frames = int(idle_sample_duration * fps)
        
        for start, end in idle_segments:
            duration = end - start
            
            # Only consider long enough idle periods
            if duration >= min_idle_frames:
                # Extract ONE representative sample from the middle
                # Center the sample window in the idle segment
                mid_point = start + duration // 2
                half_sample = sample_frames // 2
                
                sample_start = max(start, mid_point - half_sample)
                sample_end = min(end, sample_start + sample_frames)
                
                clip = sequence_array[sample_start:sample_end]
                
                if len(clip) > 0:
                    metadata = {
                        "clip_type": "idle",
                        "original_duration": float(len(clip) / fps)
                    }
                    clips.append((clip, metadata))
        
    return clips


def adaptive_resample_clip(clip, action_start, action_end, target_frames=50):
    """
    Resample variable length clip to exactly target_frames.
    Prioritizes the action core.
    """
    T, V, C = clip.shape
    
    # If short, simple upsample
    if T <= target_frames:
        flattened = clip.reshape(T, V*C)
        resampled = cv2.resize(flattened.astype(np.float32), (V*C, target_frames), interpolation=cv2.INTER_LINEAR)
        return resampled.reshape(target_frames, V, C)
        
    # If long, adaptive sampling
    # Budget: 60% to Action, 40% to Context (20% pre, 20% post)
    n_action = int(target_frames * 0.6)
    remaining = target_frames - n_action
    n_pre = int(remaining / 2)
    n_post = remaining - n_pre
    
    pre_segment = clip[:action_start]
    action_segment = clip[action_start:action_end]
    post_segment = clip[action_end:]
    
    def resample_part(part, n_target):
        if n_target <= 0: return np.zeros((0, V, C))
        if len(part) == 0: return np.zeros((0, V, C))
        flat = part.reshape(len(part), V*C)
        res = cv2.resize(flat.astype(np.float32), (V*C, n_target), interpolation=cv2.INTER_LINEAR)
        return res.reshape(n_target, V, C)

    res_pre = resample_part(pre_segment, n_pre)
    res_action = resample_part(action_segment, n_action)
    res_post = resample_part(post_segment, n_post)
    
    result = np.concatenate([res_pre, res_action, res_post], axis=0)
    
    # Handle potential rounding errors (if sum != target_frames)
    if len(result) != target_frames:
        # Resize the whole thing slightly to match exactly
        flat = result.reshape(len(result), V*C)
        res = cv2.resize(flat.astype(np.float32), (V*C, target_frames), interpolation=cv2.INTER_LINEAR)
        result = res.reshape(target_frames, V, C)
        
    return result


def map_to_ntu25(frame):
    """
    Map the 29-joint skeleton to the 25-joint NTU RGB+D format.
    """
    def get(joint_name):
        return np.array(frame[joint_name])

    def get_circumcenter(a, b, c):
        # Vectors relative to a
        u = b - a
        v = c - a
        # Cross product (Normal to plane)
        w = np.cross(u, v)
        w2 = np.dot(w, w)
        
        if w2 < 1e-8:
            # Collinear, fallback to centroid
            return (a + b + c) / 3.0
            
        # Circumcenter formula
        u2 = np.dot(u, u)
        v2 = np.dot(v, v)
        return a + (u2 * np.cross(v, w) + v2 * np.cross(w, u)) / (2 * w2)

    spine_base = get("midHip")
    neck = get("neck")
    spine_mid = (spine_base + neck) * 0.5
    head = get("nose")
    lShoulder = get("lShoulder")
    lElbow = get("lElbow")
    lWrist = get("lWrist")
    lThumb = get("lThumb")
    lPinky = get("lPinky")  # only used for computation, not actually a joint in the NTU model
    lHand = get_circumcenter(lWrist, lThumb, lPinky)
    rShoulder = get("rShoulder")
    rElbow = get("rElbow")
    rWrist = get("rWrist")
    rThumb = get("rThumb")
    rPinky = get("rPinky")  # only used for computation, not actually a joint in the NTU model
    rHand = get_circumcenter(rWrist, rThumb, rPinky)
    lHip = get("lHip")
    lKnee = get("lKnee")
    lAnkle = get("lAnkle")
    lFoot = get("lBigToe")
    rHip = get("rHip")
    rKnee = get("rKnee")
    rAnkle = get("rAnkle")
    rFoot = get("rBigToe")
    spine_shoulder = (lShoulder + rShoulder) * 0.5

    alpha = 0.9
    l_mid_fingers = (lThumb + lPinky) * 0.5
    lHandTip = l_mid_fingers + alpha * (l_mid_fingers - lWrist)
    r_mid_fingers = (rThumb + rPinky) * 0.5
    rHandTip = r_mid_fingers + alpha * (r_mid_fingers - rWrist)
    
    # NTU 25 Order (1-based index)
    joints = [
        spine_base,     # 1
        spine_mid,      # 2
        neck,           # 3
        head,           # 4
        lShoulder,      # 5
        lElbow,         # 6
        lWrist,         # 7
        lHand,          # 8
        rShoulder,      # 9
        rElbow,         # 10
        rWrist,         # 11
        rHand,          # 12
        lHip,           # 13
        lKnee,          # 14
        lAnkle,         # 15
        lFoot,          # 16
        rHip,           # 17
        rKnee,          # 18
        rAnkle,         # 19
        rFoot,          # 20
        spine_shoulder, # 21
        lHandTip,       # 22
        lThumb,         # 23
        rHandTip,       # 24
        rThumb          # 25
    ]
    
    return np.array(joints) # (25, 3)


def compute_motion(sequence_array):
    """
    Compute motion (displacement) between frames.
    M_t = P_{t+1} - P_t
    sequence_array: (T, V, 3)
    Returns: (T, V, 3)
    """
    motion_seq = np.zeros_like(sequence_array)
    motion_seq[:-1] = sequence_array[1:] - sequence_array[:-1]
    return motion_seq


def center_ntu_sequence(sequence_array):
    """
    Center the sequence by subtracting the root (Joint 21: Spine Shoulder) position from all joints.
    sequence_array: (T, V, 3)
    """
    T, V, C = sequence_array.shape
    centered_seq = np.zeros_like(sequence_array)
    
    # Joint 21 (Spine Shoulder) is at index 20 (0-based)
    root_index = 20 
    
    for t in range(T):
        root_pos = sequence_array[t, root_index, :]
        centered_seq[t] = sequence_array[t] - root_pos
        
    return centered_seq


def rotate_sequence_around_y_axis(sequence_array):
    """
    Rotate around Y-axis to align shoulders with X-axis. 
    Matches NTU convention. 
    sequence_array: (T, V, 3)
    """
    l_idx, r_idx = 4, 8
    
    l_shoulders = sequence_array[:, l_idx, :]
    r_shoulders = sequence_array[:, r_idx, :]
    vecs = r_shoulders - l_shoulders    # left to right shoulder vector
    
    # Angle in X-Z plane, how much the body is rotated around Y
    angles = np.arctan2(vecs[:, 2], vecs[:, 0])
    median_angle = np.median(angles)
    
    # Rotate by -median_angle to unwind the rotation of the shoulders
    theta = -median_angle
    c, s = np.cos(theta), np.sin(theta)
    
    R = np.array([
        [c,  0, -s],
        [0,  1,  0],
        [s,  0,  c]
    ])
    
    shape = sequence_array.shape
    reshaped = sequence_array.reshape(-1, 3)
    rotated = np.dot(reshaped, R.T)
    return rotated.reshape(shape)


def convert_to_kinect_coords(sequence_array):
    """
    Convert from Standard (X=Lat, Y=Depth, Z=Height) to Kinect (X=-Lat, Y=Height, Z=Depth).
    Based on: S_x = -K_x, S_y = K_z, S_z = K_y ==> K_x = -S_x, K_y = S_z, K_z = S_y
    """
    converted = np.zeros_like(sequence_array)
    converted[..., 0] = -sequence_array[..., 0] # X' = -X
    converted[..., 1] = sequence_array[..., 2]  # Y' = Z (Height becomes Up)
    converted[..., 2] = sequence_array[..., 1]  # Z' = Y (Depth becomes Forward)
    return converted


def resample_sequence(sequence_array, original_fps=25, target_fps=30):
    """
    Resample sequence from original_fps to target_fps using linear interpolation.
    sequence_array: (T, V, C)
    """
    T, V, C = sequence_array.shape
    if T < 2: return sequence_array
    
    new_T = int(np.round(T * (target_fps / original_fps)))
    
    if new_T == T:
        return sequence_array
    
    # Reshape to (T, V*C) for resizing (Time as Height, Features as Width)
    flattened = sequence_array.reshape(T, V * C)
    
    # cv2.resize expects (width, height) -> (V*C, new_T)
    resampled_flattened = cv2.resize(flattened.astype(np.float32), (V * C, new_T), interpolation=cv2.INTER_LINEAR)
    
    return resampled_flattened.reshape(new_T, V, C)


# Native skeleton joints order (29 joints)
NATIVE_JOINTS = [
    "midHip", "neck", "nose", "lEye", "rEye", "lEar", "rEar",
    "lShoulder", "rShoulder", "lElbow", "rElbow", "lWrist", "rWrist",
    "lThumb", "lPinky", "rThumb", "rPinky",
    "lHip", "rHip", "lKnee", "rKnee", "lAnkle", "rAnkle",
    "lBigToe", "lSmallToe", "lHeel", "rBigToe", "rSmallToe", "rHeel"
]


def map_to_native(frame):
    """Simple extraction of joints in fixed order without NTU mapping/interpolation."""
    return np.array([frame[k] for k in NATIVE_JOINTS])


def center_native_sequence(sequence_array):
    """Center sequence on 'midHip' (Root). Assumes midHip is index 0."""
    # Index 0 is "midHip" in NATIVE_JOINTS
    root_pos = sequence_array[:, 0:1, :] # (T, 1, 3)
    return sequence_array - root_pos


def rotate_native_sequence_around_z(sequence_array):
    """
    Rotate to align shoulders (lShoulder->rShoulder) along X-axis.
    CORRECTION: Native coordinates are Z-Up. We must rotate around Z to fix Yaw.
    (Function name kept for compatibility, but logic rotates around Z)
    """
    # lShoulder is index 7, rShoulder is index 8 (0-based) in NATIVE_JOINTS
    l_idx, r_idx = 7, 8
    
    l_shoulders = sequence_array[:, l_idx, :]
    r_shoulders = sequence_array[:, r_idx, :]
    vecs = r_shoulders - l_shoulders
    
    # Angle in X-Y plane (Yaw)
    # Ideally should be 0 (aligned with X axis)
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    median_angle = np.median(angles)
    theta = -median_angle
    
    # Rotation matrix around Z (UP)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    # Apply R to all points: (T, V, 3)
    # Reshape to (T*V, 3) -> dot -> reshape back
    T, V, C = sequence_array.shape
    flat = sequence_array.reshape(-1, 3)
    rotated = flat @ R.T
    return rotated.reshape(T, V, C)
