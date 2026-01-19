"""
Script to create a paired dataset pickle file for ULF-to-HF MRI synthesis.

This script loads paired ULF and HF MRI images and creates a pickle file
following the MOTFM repository structure.

Usage:
    python create_paired_dataset.py \
        --hf_dir HF-ULF-SynthPair/HF \
        --ulf_dir HF-ULF-SynthPair/ULF-Synth-v1 \
        --output_pkl data/ulf_hf_paired.pkl \
        --train_split 0.7 \
        --val_split 0.15 \
        --test_split 0.15
"""

import argparse
import os
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch


def load_nifti_volume(file_path):
    """Load a NIfTI file and return the numpy array."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data


def normalize_volume(volume):
    """Normalize volume to [0, 1] range."""
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 0:
        return (volume - vmin) / (vmax - vmin)
    return volume


def extract_slices_2d(ulf_volume, hf_volume, axis=2):
    """
    Extract 2D slices from paired 3D volumes along the specified axis.
    
    Args:
        ulf_volume: 3D numpy array (ULF MRI)
        hf_volume: 3D numpy array (HF MRI)
        axis: Axis along which to slice (0, 1, or 2)
    
    Returns:
        List of tuples (ulf_slice, hf_slice)
    """
    slices = []
    num_slices = ulf_volume.shape[axis]
    
    for i in range(num_slices):
        if axis == 0:
            ulf_slice = ulf_volume[i, :, :]
            hf_slice = hf_volume[i, :, :]
        elif axis == 1:
            ulf_slice = ulf_volume[:, i, :]
            hf_slice = hf_volume[:, i, :]
        else:  # axis == 2
            ulf_slice = ulf_volume[:, :, i]
            hf_slice = hf_volume[:, :, i]
        
        # Skip slices with very low information content (e.g., background)
        if ulf_slice.mean() > 0.01 and hf_slice.mean() > 0.01:
            slices.append((ulf_slice, hf_slice))
    
    return slices


def create_paired_dataset(
    hf_dir,
    ulf_dir,
    output_pkl,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    slice_axis=2,
    use_3d=False,
):
    """
    Create a paired dataset pickle file.
    
    Args:
        hf_dir: Directory containing high-field MRI images
        ulf_dir: Directory containing ultra low-field MRI images
        output_pkl: Output pickle file path
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        slice_axis: Axis along which to extract 2D slices (0, 1, or 2)
        use_3d: If True, keep volumes as 3D; if False, extract 2D slices
    """
    print("Loading paired MRI datasets...")
    print(f"HF directory: {hf_dir}")
    print(f"ULF directory: {ulf_dir}")
    
    # Get list of files
    hf_files = sorted(Path(hf_dir).glob("*.nii*"))
    ulf_files = sorted(Path(ulf_dir).glob("*.nii*"))
    
    print(f"Found {len(hf_files)} HF files and {len(ulf_files)} ULF files")
    
    if len(hf_files) == 0 or len(ulf_files) == 0:
        raise ValueError("No NIfTI files found in one or both directories!")
    
    # Match pairs by filename
    hf_dict = {f.stem: f for f in hf_files}
    ulf_dict = {f.stem: f for f in ulf_files}
    
    common_names = set(hf_dict.keys()) & set(ulf_dict.keys())
    if len(common_names) == 0:
        print("\nWarning: No exact filename matches found. Pairing by index instead.")
        # Pair by index if no name matches
        min_len = min(len(hf_files), len(ulf_files))
        paired_files = [(hf_files[i], ulf_files[i]) for i in range(min_len)]
    else:
        print(f"Found {len(common_names)} matching pairs")
        paired_files = [(hf_dict[name], ulf_dict[name]) for name in sorted(common_names)]
    
    # Process all volumes
    all_samples = []
    
    for idx, (hf_path, ulf_path) in enumerate(paired_files):
        print(f"Processing pair {idx+1}/{len(paired_files)}: {hf_path.name}")
        
        # Load volumes
        hf_volume = load_nifti_volume(hf_path)
        ulf_volume = load_nifti_volume(ulf_path)
        
        # Normalize volumes
        hf_volume = normalize_volume(hf_volume)
        ulf_volume = normalize_volume(ulf_volume)
        
        if use_3d:
            # Keep as 3D volumes [D, H, W] -> [1, D, H, W]
            # In MOTFM, we need channel dimension
            sample = {
                "image": hf_volume[np.newaxis, :, :, :].astype(np.float32),  # Target (HF)
                "mask": ulf_volume[np.newaxis, :, :, :].astype(np.float32),   # Condition (ULF)
                "class": "HF-ULF-pair",
                "metadata": {
                    "hf_file": hf_path.name,
                    "ulf_file": ulf_path.name,
                    "volume_idx": idx,
                }
            }
            all_samples.append(sample)
        else:
            # Extract 2D slices
            slices = extract_slices_2d(ulf_volume, hf_volume, axis=slice_axis)
            
            for slice_idx, (ulf_slice, hf_slice) in enumerate(slices):
                sample = {
                    "image": hf_slice[np.newaxis, :, :].astype(np.float32),  # Target (HF)
                    "mask": ulf_slice[np.newaxis, :, :].astype(np.float32),  # Condition (ULF)
                    "class": "HF-ULF-pair",
                    "metadata": {
                        "hf_file": hf_path.name,
                        "ulf_file": ulf_path.name,
                        "volume_idx": idx,
                        "slice_idx": slice_idx,
                        "slice_axis": slice_axis,
                    }
                }
                all_samples.append(sample)
    
    print(f"\nTotal samples collected: {len(all_samples)}")
    
    # Split into train/val/test
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Split fractions must sum to 1.0"
    
    # First split: train vs (val+test)
    train_samples, temp_samples = train_test_split(
        all_samples,
        train_size=train_split,
        random_state=42,
        shuffle=True
    )
    
    # Second split: val vs test
    val_ratio = val_split / (val_split + test_split)
    val_samples, test_samples = train_test_split(
        temp_samples,
        train_size=val_ratio,
        random_state=42,
        shuffle=True
    )
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Create the data dictionary
    data_dict = {
        "train": train_samples,
        "valid": val_samples,
        "test": test_samples,
    }
    
    # Save to pickle
    os.makedirs(os.path.dirname(output_pkl) if os.path.dirname(output_pkl) else ".", exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(data_dict, f)
    
    print(f"\nDataset saved to: {output_pkl}")
    print(f"Mode: {'3D volumes' if use_3d else '2D slices'}")
    if not use_3d:
        print(f"Slice axis: {slice_axis}")
    
    # Print sample info
    print("\n=== Sample Information ===")
    sample = data_dict["train"][0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Class: {sample['class']}")
    print(f"Metadata: {sample['metadata']}")


def main():
    parser = argparse.ArgumentParser(
        description="Create paired ULF-HF MRI dataset pickle file"
    )
    parser.add_argument(
        "--hf_dir",
        type=str,
        required=True,
        help="Directory containing high-field MRI images"
    )
    parser.add_argument(
        "--ulf_dir",
        type=str,
        required=True,
        help="Directory containing ultra low-field MRI images"
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        default="data/ulf_hf_paired.pkl",
        help="Output pickle file path"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.7,
        help="Fraction of data for training (default: 0.7)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15)"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="Fraction of data for testing (default: 0.15)"
    )
    parser.add_argument(
        "--slice_axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Axis along which to extract 2D slices (0, 1, or 2; default: 2)"
    )
    parser.add_argument(
        "--use_3d",
        action="store_true",
        help="Keep volumes as 3D instead of extracting 2D slices"
    )
    
    args = parser.parse_args()
    
    create_paired_dataset(
        hf_dir=args.hf_dir,
        ulf_dir=args.ulf_dir,
        output_pkl=args.output_pkl,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        slice_axis=args.slice_axis,
        use_3d=args.use_3d,
    )


if __name__ == "__main__":
    main()
