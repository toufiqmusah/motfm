# ULF-to-HF MRI Synthesis with MOTFM

This guide explains how to use MOTFM for paired ULF (Ultra Low Field) to HF (High Field) MRI synthesis.

## Quick Answer to Your Questions

### 1. Does this repo support 3D?

**Yes, but with caveats:**
- The model uses `DiffusionModelUNet` from MONAI's `generative` library, which **supports both 2D and 3D** via the `spatial_dims` parameter
- Set `spatial_dims: 2` for 2D slices (recommended - lower memory, faster training)
- Set `spatial_dims: 3` for full 3D volumes (requires significant GPU memory)

**Recommendation:** Start with 2D slices extracted from your 3D volumes. This is more memory-efficient and trains faster. You can extract slices along any axis (axial, sagittal, or coronal).

### 2. How to create the pickle file for paired synthesis?

Use the provided `create_paired_dataset.py` script:

```bash
# For 2D slices (recommended)
python create_paired_dataset.py \
    --hf_dir HF-ULF-SynthPair/HF \
    --ulf_dir HF-ULF-SynthPair/ULF-Synth-v1 \
    --output_pkl data/ulf_hf_paired.pkl \
    --slice_axis 2 \
    --train_split 0.7 \
    --val_split 0.15 \
    --test_split 0.15

# For 3D volumes (if you have enough GPU memory)
python create_paired_dataset.py \
    --hf_dir HF-ULF-SynthPair/HF \
    --ulf_dir HF-ULF-SynthPair/ULF-Synth-v1 \
    --output_pkl data/ulf_hf_paired_3d.pkl \
    --use_3d
```

## Understanding the Data Structure

The pickle file contains:
```python
{
    "train": [
        {
            "image": HF_slice,      # Target: High Field MRI [1, H, W]
            "mask": ULF_slice,      # Condition: Ultra Low Field MRI [1, H, W]
            "class": "HF-ULF-pair", # Class label (single class for all pairs)
            "metadata": {...}       # Additional info (filenames, indices)
        },
        ...
    ],
    "valid": [...],
    "test": [...]
}
```

**Key concept:** In MOTFM's terminology:
- `"image"` = your **target** (what you want to generate) = **HF MRI**
- `"mask"` = your **conditioning** (what you provide as input) = **ULF MRI**

## Training

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Create your dataset:**
```bash
python create_paired_dataset.py \
    --hf_dir HF-ULF-SynthPair/HF \
    --ulf_dir HF-ULF-SynthPair/ULF-Synth-v1 \
    --output_pkl data/ulf_hf_paired.pkl
```

3. **Train the model:**
```bash
python trainer.py --config_path configs/ulf_hf_paired.yaml
```

## Inference

Generate HF MRI from ULF MRI:

```bash
python inferer.py \
    --config_path configs/ulf_hf_paired.yaml \
    --model_path checkpoints/ulf_hf_paired/latest \
    --num_inference_steps 10 \
    --num_samples 100
```

## Configuration Options

The `configs/ulf_hf_paired.yaml` file has been created for you with settings for paired synthesis:

### Important parameters to adjust:

1. **For 2D vs 3D:**
   - 2D: `spatial_dims: 2` (in `model_args`)
   - 3D: `spatial_dims: 3` (requires more GPU memory)

2. **Batch size:**
   - Adjust `batch_size` in `train_args` based on your GPU memory
   - For 2D: Try 8-16
   - For 3D: Try 1-4

3. **Model capacity:**
   - `num_channels: [32, 64, 128, 256, 512]` - increase for more capacity
   - `attention_levels` - enable/disable attention at different levels

4. **Training:**
   - `num_epochs: 200` - adjust based on convergence
   - `lr: 1.0e-4` - learning rate
   - `val_freq: 5` - validate every N epochs

## File Structure

```
motfm/
├── create_paired_dataset.py     # Script to create pickle from your data
├── configs/
│   └── ulf_hf_paired.yaml      # Configuration for ULF-HF synthesis
├── data/
│   └── ulf_hf_paired.pkl       # Generated pickle file (you create this)
├── trainer.py                   # Training script
├── inferer.py                   # Inference script
└── checkpoints/
    └── ulf_hf_paired/          # Saved model checkpoints
```

## Memory Requirements

### 2D Mode (spatial_dims: 2):
- Typical slice size: 256×256
- Batch size 8: ~6-8 GB GPU memory
- Batch size 16: ~12-16 GB GPU memory

### 3D Mode (spatial_dims: 3):
- Typical volume size: 128×128×128
- Batch size 1: ~16-24 GB GPU memory
- Batch size 2: ~32-48 GB GPU memory

## Tips for Better Results

1. **Data preprocessing:**
   - Ensure ULF and HF images are properly registered (aligned)
   - Normalize intensity ranges consistently
   - Remove slices with minimal brain tissue

2. **Training:**
   - Monitor validation loss to avoid overfitting
   - Use data augmentation if your dataset is small
   - Try different `num_inference_steps` during inference (5-20)

3. **Conditioning:**
   - The ULF image is used as a "mask" conditioning signal
   - This guides the flow matching to generate HF images that match the ULF structure

4. **3D considerations:**
   - If using 3D mode, consider reducing model capacity (smaller `num_channels`)
   - Use mixed precision training to save memory
   - Start with 2D and move to 3D only if needed

## Troubleshooting

**Q: Out of memory errors?**
- Reduce `batch_size`
- Use 2D mode instead of 3D
- Reduce `num_channels` in model config

**Q: Training is too slow?**
- Enable `use_flash_attention: true` (already set)
- Use smaller image dimensions
- Use fewer attention levels

**Q: Generated images are blurry?**
- Increase `num_inference_steps` (try 20-50)
- Train for more epochs
- Increase model capacity

**Q: Files don't match between HF and ULF directories?**
- The script will try to match by filename
- If no matches, it pairs by index order
- Check your filenames and ensure consistency
