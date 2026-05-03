import subprocess
import sys
import os

"""
End-to-End Data Processing Pipeline

This script orchestrates the full data preparation workflow for Euro 2024 goalkeeper skeletal data.
It executes the following steps sequentially:
    1. preprocessing.filter_data: 
       - Extracts raw tracking data from 7z archives
       - Synchronizes ball and skeleton timestamps
       - Filters for goalkeeper interactions in the final third
       - Outputs: intermediate JSON files in data/filtered/
    
    2. preprocessing.to_tensor:
       - Loads filtered data and splits into Train/Val sets
       - Computes canonical skeleton for normalization
       - Aligns, normalizes, and resamples sequences (30fps -> 50 frames)
       - Generates Position and Motion tensors for ST-GCN
       - Outputs: .npy tensors and metadata in data/tensor/

Usage:
    Run from the project root or preprocessing/ directory:
    python -m preprocessing.run_pipeline
"""

def run_pipeline():
    """
    Executes the full pipeline:
    1. Filter Data (preprocessing/filter_data.py)
    2. To Tensor (preprocessing/to_tensor.py)
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Ensure we run from the project root so that relative paths 
    # (data/euro2024, data/filtered, etc.) resolve correctly.
    os.chdir(project_root)

    print(f"Starting pipeline from: {project_root}")

    # We use 'python -m <module>' to ensure the 'preprocessing' package is 
    # resolving correctly in imports (especially for to_tensor.py).
    # Step 1: Filter raw data
    # Step 2: Generate NTU tensors (default)
    # Step 3: Generate Native tensors (--native)
    steps = [
        ("preprocessing.filter_data", []),
        ("preprocessing.to_tensor", []),
        ("preprocessing.to_tensor", ["--native"])
    ]

    for module, args in steps:
        print(f"\n{'='*40}")
        print(f"Running: {module} {' '.join(args)}")
        print(f"{'='*40}")
        
        try:
            cmd = [sys.executable, "-m", module] + args
            subprocess.run(
                cmd, 
                check=True,
                cwd=project_root
            )
        except subprocess.CalledProcessError as e:
            print(f"\nPipeline failed at step: {module} {' '.join(args)}")
            print(f"Error: {e}")
            sys.exit(1)
            
    print("\n" + "="*40)
    print("Pipeline completed successfully!")
    print("="*40)

if __name__ == "__main__":
    run_pipeline()
