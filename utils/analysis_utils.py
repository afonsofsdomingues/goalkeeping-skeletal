
import torch
import numpy as np
import json
import os

data_joints_paths = {
    'ntu': {
        'train': '../data/NTU60_frame50/xview/train_position.npy',
        'val': '../data/NTU60_frame50/xview/val_position.npy',
    },
    'snfc': {
        'train': '../data/tensor/train_position.npy',
        'val': '../data/tensor/val_position.npy',
    },
    'snfc_native': {
        'train': '../data/tensor_native/train_position.npy',
        'val': '../data/tensor_native/val_position.npy',
    },
}

data_labels_paths = {
    'ntu': {
        'train': '../data/NTU60_frame50/xview/train_label.npy',
        'val': '../data/NTU60_frame50/xview/val_label.npy',
    },
    'snfc': None
}

def load_or_compute_embeddings(
    model,
    dataset_name,
    N=2000,
    embeddings_base_path='../models/CrosSCLR/embeddings/',
    views=['joint', 'motion', 'bone', 'all'],
    model_name='',
    split='val',
    return_indices=False,
):
    """
    Load or compute embeddings for a dataset.
    
    Args:
        model: The model with get_embeddings method.
        dataset_name: Name of the dataset (e.g., 'ntu', 'snfc').
        N: Number of samples to sample.
        embeddings_base_path: Base path for saving/loading embeddings.
        views: List of views to compute ('joint', 'motion', 'bone', 'all').
        model_name: Name of the model variant (e.g., '', 'pretrained', 'finetuned').
        split: Which split to use ('train' or 'val').
        return_indices: If True, also return the sampled dataset indices used to create the embeddings.
    
    Returns:
        embeddings: Dict of embeddings tensors.
        labels: Labels tensor or None.
    """
    data_joints_path = data_joints_paths[dataset_name][split]
    data_labels_path = data_labels_paths.get(dataset_name, None)
    if data_labels_path is not None:
        data_labels_path = data_labels_path[split]
    
    # Deterministic sampling so the same (dataset_name, split, N) always maps to the same rows.
    rng = np.random.default_rng(42)
    
    # Construct paths
    paths = {}
    for view in views:
        if model_name:
            paths[view] = f'{embeddings_base_path}{dataset_name}_{model_name}_embeddings_{split}_{view}_{N}.pt'
        else:
            paths[view] = f'{embeddings_base_path}{dataset_name}_embeddings_{split}_{view}_{N}.pt'
    
    labels_path = None
    if data_labels_path:
        if model_name:
            labels_path = f'{embeddings_base_path}{dataset_name}_{model_name}_embeddings_{split}_labels_{N}.pt'
        else:
            labels_path = f'{embeddings_base_path}{dataset_name}_embeddings_{split}_labels_{N}.pt'
    
    # Initialize
    embeddings = {view: None for view in views}
    labels = None
    sampled_indices = None
    
    # Load existing
    for view in views:
        if os.path.exists(paths[view]):
            print(f"Loading {view} embeddings from {paths[view]}")
            embeddings[view] = torch.load(paths[view])
    
    if labels_path and os.path.exists(labels_path):
        print(f"Loading labels from {labels_path}")
        labels = torch.load(labels_path)
    
    # Check missing
    missing_views = [view for view in views if embeddings[view] is None]
    need_compute = bool(missing_views) or (data_labels_path and labels is None)
    
    if need_compute:
        print("Some data missing, computing the missing ones...")
        joints = np.load(data_joints_path)
        print(f"Loaded data shape: {joints.shape}")
        total = joints.shape[0]
        sampled_indices = rng.choice(total, N, replace=False)
        joints = joints[sampled_indices]
        input_joints = torch.from_numpy(joints).float()
        
        if data_labels_path:
            labels_np = np.load(data_labels_path)
            labels_np = labels_np[sampled_indices]
            labels = torch.from_numpy(labels_np).float()
        
        with torch.no_grad():
            for view in missing_views:
                embeddings[view] = model.get_embeddings(input_joints, view=view, normalize=True)
                torch.save(embeddings[view].cpu(), paths[view])
                print(f"Saved {view} embeddings to {paths[view]}")
        
        if labels_path:
            torch.save(labels.cpu(), labels_path)
            print(f"Saved labels to {labels_path}")
    else:
        # Even if we load cached embeddings, we still define the sampled indices deterministically
        # so callers can align metadata.
        joints = np.load(data_joints_path)
        total = joints.shape[0]
        sampled_indices = rng.choice(total, N, replace=False)
        print("All data loaded successfully.")

    if return_indices:
        return embeddings, labels, sampled_indices
    return embeddings, labels


# ── Metadata helpers ──────────────────────────────────────────────────────────

data_meta_paths = {
    'ntu': None,
    'snfc': {
        'train': '../data/tensor/train_metadata.json',
        'val': '../data/tensor/val_metadata.json',
    },
    'snfc_native': {
        'train': '../data/tensor_native/train_metadata.json',
        'val': '../data/tensor_native/val_metadata.json',
    },
}


def load_metadata_for_indices(dataset_name, split, indices):
    """Load metadata entries for the given dataset indices.

    Args:
        dataset_name: Key into data_meta_paths (e.g. 'snfc_native').
        split: 'train' or 'val'.
        indices: Array-like of integer indices into the full dataset.

    Returns:
        List of metadata dicts (one per index).
    """
    meta_cfg = data_meta_paths.get(dataset_name)
    if meta_cfg is None:
        raise ValueError(f"No metadata path configured for dataset '{dataset_name}'")
    meta_path = meta_cfg[split]
    with open(meta_path, 'r') as f:
        all_meta = json.load(f)
    return [all_meta[i] for i in indices]


# ── Kinematic / motion-energy helpers ─────────────────────────────────────────

def compute_motion_energy(position_data):
    """Compute per-sample motion energy from raw skeletal position tensors.

    Args:
        position_data: numpy array of shape (N, C, T, V, M)
            C=3 (xyz), T=frames, V=joints, M=bodies.

    Returns:
        motion_energy: (N,) total squared velocity summed over joints, frames, coords.
        max_frame_energy: (N,) peak single-frame energy (useful for detecting bursts).
        displacement: (N,) Euclidean displacement of body centroid from first to last frame.
    """
    # Use first body only
    pos = position_data[:, :, :, :, 0]  # (N, C, T, V)

    # Frame-to-frame velocity
    vel = np.diff(pos, axis=2)  # (N, C, T-1, V)

    # Per-frame energy = sum of squared velocities over coords & joints
    frame_energy = (vel ** 2).sum(axis=(1, 3))  # (N, T-1)

    motion_energy = frame_energy.sum(axis=1)  # (N,)
    max_frame_energy = frame_energy.max(axis=1)  # (N,)

    # Centroid displacement (first vs last non-zero frame)
    centroid = pos.mean(axis=3)  # (N, C, T) - average over joints
    disp = centroid[:, :, -1] - centroid[:, :, 0]  # (N, C)
    displacement = np.linalg.norm(disp, axis=1)  # (N,)

    return motion_energy, max_frame_energy, displacement
