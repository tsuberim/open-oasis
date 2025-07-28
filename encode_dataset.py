import os
import torch
import torch.nn as nn
import h5py
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import time
from safetensors.torch import load_file
from vae import VAE_models
from utils import get_device
from train_vae import VideoDataset

def load_trained_vae(checkpoint_path, device):
    """Load a trained VAE model from checkpoint."""
    print(f"Loading VAE model from {checkpoint_path}")
    
    # Create model (using the same architecture as training)
    model = VAE_models["vit-l-20-shallow-encoder"]()
    
    # Load checkpoint
    if checkpoint_path.endswith('.safetensors'):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel state dict
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print(f"VAE model loaded successfully")
    return model

def encode_video_dataset(model, dataset, device, output_dir="./encoded", batch_size=32, target_fps=20):
    """Encode the entire video dataset and save as H5 files with consistent frame rate."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get latent dimensions from model
    with torch.no_grad():
        # Create a dummy input to get latent dimensions
        dummy_input = torch.randn(1, 3, 360, 640).to(device)
        _, _, _, latent = model(dummy_input)
        latent_shape = latent.shape[1:]  # Remove batch dimension
        print(f"Latent shape: {latent_shape}")
    
    # Process each video separately
    video_metadata = dataset.video_metadata
    
    for video_idx, video_info in enumerate(tqdm(video_metadata, desc="Processing videos")):
        video_path = video_info['path']
        video_name = video_path.stem
        original_fps = video_info['fps']
        duration = video_info['duration']
        
        # Calculate target frame count for consistent 20 fps
        target_frame_count = int(duration * target_fps)
        
        print(f"\nProcessing video {video_idx + 1}/{len(video_metadata)}: {video_name}")
        print(f"  Original: {video_info['num_frames']} frames @ {original_fps:.1f} fps")
        print(f"  Target: {target_frame_count} frames @ {target_fps} fps")
        print(f"  Duration: {duration:.1f}s")
        
        # Create H5 file for this video
        h5_path = output_dir / f"{video_name}_encoded.h5"
        
        # Calculate frame sampling indices for consistent 20 fps using timestamp-based sampling
        if original_fps >= target_fps:
            # Need to skip frames (downsample)
            frame_indices = []
            target_frame_interval = 1.0 / target_fps  # Time between target frames
            
            # Get all frames for this video
            video_frame_indices = [(j, frame_idx) for j, (vid_idx, frame_idx) in enumerate(dataset.frame_indices) if vid_idx == video_idx]
            
            next_target_time = 0.0  # Next timestamp we want to sample
            
            for dataset_idx, frame_idx in video_frame_indices:
                frame_time = frame_idx / original_fps  # Current frame timestamp
                
                # If we've reached or passed the next target time, add this frame
                if frame_time >= next_target_time:
                    frame_indices.append(dataset_idx)
                    next_target_time += target_frame_interval
                    
                    # Stop if we have enough frames
                    if len(frame_indices) >= target_frame_count:
                        break
        else:
            # Need to duplicate frames (upsample) - raise error for now
            raise ValueError(f"Video {video_name} has {original_fps:.1f} fps which is below target {target_fps} fps. "
                           f"Please use videos with at least {target_fps} fps.")
        
        print(f"  Sampling {len(frame_indices)} frames from original video")
        print(f"  Target frame count: {target_frame_count}")
        if len(frame_indices) < target_frame_count:
            print(f"  Warning: Only got {len(frame_indices)} frames, expected {target_frame_count}")
        
        # Use actual number of frames we got
        actual_frame_count = len(frame_indices)
        
        with h5py.File(h5_path, 'w') as h5_file:
            # Create datasets (no compression)
            latents_dataset = h5_file.create_dataset(
                'latents', 
                shape=(actual_frame_count, *latent_shape), 
                dtype=np.float32,
                chunks=(min(100, actual_frame_count), *latent_shape)
            )
            
            # Store video metadata
            metadata_group = h5_file.create_group('metadata')
            metadata_group.attrs['video_path'] = str(video_path)
            metadata_group.attrs['original_fps'] = original_fps
            metadata_group.attrs['target_fps'] = target_fps
            metadata_group.attrs['original_frames'] = video_info['num_frames']
            metadata_group.attrs['target_frames'] = target_frame_count
            metadata_group.attrs['actual_frames'] = actual_frame_count
            metadata_group.attrs['duration'] = duration
            metadata_group.attrs['width'] = video_info['width']
            metadata_group.attrs['height'] = video_info['height']
            metadata_group.attrs['latent_shape'] = latent_shape
            metadata_group.attrs['encoding_timestamp'] = time.time()
            
            # Process frames in batches
            with torch.no_grad():
                for batch_start in tqdm(range(0, len(frame_indices), batch_size), 
                                      desc=f"Encoding {video_name}", leave=False):
                    batch_end = min(batch_start + batch_size, len(frame_indices))
                    batch_indices = frame_indices[batch_start:batch_end]
                    
                    # Get frames for this batch
                    frames = dataset.get_batch(batch_indices)
                    frames = frames.to(device)
                    
                    # Encode frames
                    _, _, _, latents = model(frames)
                    
                    # Store latents (keep original shape)
                    start_frame = batch_start
                    end_frame = batch_end
                    latents_dataset[start_frame:end_frame] = latents.cpu().numpy()
        
        print(f"  Saved encoded video to {h5_path}")
    
    print(f"\nEncoding completed! All videos saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Encode video dataset using trained VAE")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to VAE checkpoint (.safetensors or .pth)")
    parser.add_argument("--videos-dir", "-d", default="./videos", help="Directory with videos")
    parser.add_argument("--output-dir", "-o", default="./encoded", help="Output directory for encoded files")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--target-size", "-T", nargs=2, type=int, default=[360, 640], help="Target frame size (height width)")
    parser.add_argument("--target-fps", "-f", type=int, default=20, help="Target frame rate (default: 20)")
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load trained VAE model
    model = load_trained_vae(args.checkpoint, device)
    
    # Load dataset
    print("Loading video dataset...")
    dataset = VideoDataset(
        videos_dir=args.videos_dir,
        target_size=tuple(args.target_size)
    )
    
    print(f"Found {len(dataset.video_metadata)} videos with {len(dataset)} total frames")
    
    # Encode dataset
    encode_video_dataset(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        target_fps=args.target_fps
    )

if __name__ == "__main__":
    main() 