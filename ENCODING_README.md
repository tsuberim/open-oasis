# Video Dataset Encoding

This directory contains scripts to encode video datasets using a trained VAE model and save the encoded tensors in H5 format.

## Files

- `encode_dataset.py` - Main script to encode videos using trained VAE
- `verify_encoded.py` - Script to verify and inspect encoded H5 files
- `ENCODING_README.md` - This documentation

## Usage

### 1. Encode Video Dataset

```bash
python encode_dataset.py --checkpoint ./checkpoints/best_vae_model.safetensors --videos-dir ./videos --output-dir ./encoded
```

**Arguments:**
- `--checkpoint` (required): Path to trained VAE checkpoint (.safetensors or .pth)
- `--videos-dir`: Directory containing video files (default: ./videos)
- `--output-dir`: Output directory for encoded H5 files (default: ./encoded)
- `--batch-size`: Batch size for encoding (default: 32)
- `--target-size`: Target frame size as height width (default: 360 640)
- `--target-fps`: Target frame rate (default: 20)

### 2. Verify Encoded Files

```bash
# List all encoded files and their metadata
python verify_encoded.py --encoded-dir ./encoded

# Load and inspect a specific encoded file
python verify_encoded.py --encoded-dir ./encoded --load-sample video_name_encoded.h5 --start-frame 0 --end-frame 10
```

## Output Format

Each video is encoded into a separate H5 file with the following structure:

```
video_name_encoded.h5
├── latents (dataset)
│   ├── shape: (target_frames, latent_dim)
│   ├── dtype: float32
│   └── compression: none
└── metadata (group)
    ├── video_path: str
    ├── original_fps: float
    ├── target_fps: int
    ├── original_frames: int
    ├── target_frames: int
    ├── duration: float
    ├── width: int
    ├── height: int
    ├── latent_dim: int
    └── encoding_timestamp: float
```

## Loading Encoded Data

```python
import h5py
import numpy as np

# Load encoded video
with h5py.File('video_name_encoded.h5', 'r') as f:
    latents = f['latents'][:]  # Shape: (target_frames, latent_dim)
    metadata = f['metadata']
    
    # Get video info
    target_fps = metadata.attrs['target_fps']
    duration = metadata.attrs['duration']
    latent_dim = metadata.attrs['latent_dim']

# Load specific frame range
with h5py.File('video_name_encoded.h5', 'r') as f:
    # Load frames 100-200
    latents_subset = f['latents'][100:200]  # Shape: (100, latent_dim)
```

## Features

- **Efficient Processing**: Uses the same optimized VideoDataset from training
- **Memory Efficient**: Processes videos in batches with GPU acceleration
- **Consistent Frame Rate**: All videos are resampled to 20 fps for consistency
- **Intelligent Frame Sampling**: Videos with higher fps are intelligently downsampled
- **Rich Metadata**: Stores video information alongside encoded data
- **Progress Tracking**: Shows encoding progress with tqdm
- **Error Handling**: Graceful handling of missing files and errors

## Requirements

- `h5py` for H5 file I/O
- `torch` for model inference
- `safetensors` for loading checkpoints
- `tqdm` for progress bars
- All other dependencies from `requirements.txt`

## Example Workflow

1. Train VAE model using `train_vae.py`
2. Encode dataset: `python encode_dataset.py --checkpoint ./checkpoints/best_vae_model.safetensors`
3. Verify encoding: `python verify_encoded.py`
4. Use encoded data for downstream tasks (e.g., action prediction, world modeling) 