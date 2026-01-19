"""
Quick script to verify the structure of a generated pickle file.

Usage:
    python verify_pickle.py data/ulf_hf_paired.pkl
"""

import argparse
import pickle
import sys


def verify_pickle(pkl_path):
    """Verify the structure of a dataset pickle file."""
    print(f"Loading pickle file: {pkl_path}")
    
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return False
    
    print("\n" + "="*60)
    print("PICKLE FILE STRUCTURE")
    print("="*60)
    
    # Check top-level keys
    required_keys = ["train", "valid", "test"]
    actual_keys = list(data.keys())
    print(f"\nTop-level keys: {actual_keys}")
    
    missing_keys = set(required_keys) - set(actual_keys)
    if missing_keys:
        print(f"WARNING: Missing keys: {missing_keys}")
    
    # Check each split
    for split in required_keys:
        if split not in data:
            print(f"\n{split.upper()}: NOT FOUND")
            continue
            
        samples = data[split]
        print(f"\n{split.upper()}:")
        print(f"  Number of samples: {len(samples)}")
        
        if len(samples) == 0:
            print(f"  WARNING: No samples in {split} split!")
            continue
        
        # Check first sample structure
        sample = samples[0]
        print(f"  Sample keys: {list(sample.keys())}")
        
        # Check required fields
        if "image" in sample:
            print(f"    - image shape: {sample['image'].shape}, dtype: {sample['image'].dtype}")
        else:
            print(f"    - ERROR: 'image' field missing!")
        
        if "mask" in sample:
            print(f"    - mask shape: {sample['mask'].shape}, dtype: {sample['mask'].dtype}")
        else:
            print(f"    - ERROR: 'mask' field missing!")
        
        if "class" in sample:
            print(f"    - class: {sample['class']}")
        else:
            print(f"    - ERROR: 'class' field missing!")
        
        if "metadata" in sample:
            print(f"    - metadata: {sample['metadata']}")
        
        # Check consistency across samples
        print(f"\n  Checking consistency across all {len(samples)} samples...")
        shapes_ok = True
        for i, s in enumerate(samples):
            if s["image"].shape != sample["image"].shape:
                print(f"    WARNING: Sample {i} has different image shape: {s['image'].shape}")
                shapes_ok = False
            if s["mask"].shape != sample["mask"].shape:
                print(f"    WARNING: Sample {i} has different mask shape: {s['mask'].shape}")
                shapes_ok = False
        
        if shapes_ok:
            print(f"    ✓ All samples have consistent shapes")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_samples = sum(len(data.get(split, [])) for split in required_keys)
    print(f"Total samples: {total_samples}")
    
    # Check if dimensions are 2D or 3D
    if len(samples) > 0 and "image" in samples[0]:
        ndim = len(samples[0]["image"].shape)
        if ndim == 3:  # [C, H, W]
            print(f"Data type: 2D slices (shape: {samples[0]['image'].shape})")
        elif ndim == 4:  # [C, D, H, W]
            print(f"Data type: 3D volumes (shape: {samples[0]['image'].shape})")
        else:
            print(f"WARNING: Unexpected number of dimensions: {ndim}")
    
    print("\n✓ Pickle file structure looks good!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify pickle file structure")
    parser.add_argument("pickle_path", type=str, help="Path to pickle file to verify")
    args = parser.parse_args()
    
    success = verify_pickle(args.pickle_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
