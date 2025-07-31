import torch
import numpy as np
import argparse
from pathlib import Path
import decord
from torch.utils.data import DataLoader
from vae import VAE_models
from utils import get_device

def generate_mock_embeddings(num_samples=1000, num_patches=576, latent_dim=32, batch_size=64, output_dir="./encoded"):
    """
    Generate mock embedding data that matches the VAE output shape.
    
    Args:
        num_samples: Number of embedding samples to generate
        num_patches: Number of patches per sample (576 for the VAE)
        latent_dim: Dimension of the latent space (32 for the VAE)
        batch_size: Batch size for processing
        output_dir: Directory to save the mock embeddings
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Generating {num_samples} mock embeddings with shape ({num_patches}, {latent_dim})")
    print(f"Output directory: {output_path}")
    
    # Generate mock embeddings
    # Using normal distribution to simulate realistic VAE latent vectors
    # Shape: (num_samples, num_patches, latent_dim)
    embeddings = torch.randn(num_samples, num_patches, latent_dim)
    
    # Save as a single file
    torch.save(embeddings, output_path / "embeddings.pt")
    
    # Generate metadata
    metadata = {
        "num_samples": num_samples,
        "num_patches": num_patches,
        "latent_dim": latent_dim,
        "shape": embeddings.shape,
        "mean": embeddings.mean().item(),
        "std": embeddings.std().item(),
        "min": embeddings.min().item(),
        "max": embeddings.max().item()
    }
    
    # Save metadata
    torch.save(metadata, output_path / "metadata.pt")
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Statistics: mean={metadata['mean']:.4f}, std={metadata['std']:.4f}")
    print(f"Range: [{metadata['min']:.4f}, {metadata['max']:.4f}]")
    print(f"Files saved to: {output_path}")
    
    return embeddings, metadata

def embed_video(video_path, checkpoint_path, output_dir="./encoded", batch_size=32):
    """
    Embed an entire video using a trained VAE checkpoint.
    
    Args:
        video_path: Path to the video file
        checkpoint_path: Path to the trained VAE checkpoint
        output_dir: Directory to save the embeddings
        batch_size: Batch size for processing
    """
    
    # Set up device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load VAE model
    print("Loading VAE model...")
    model = VAE_models["vit-l-20-shallow-encoder"](latent_dim=32)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Load video
    print(f"Loading video: {video_path}")
    video_reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
    num_frames = len(video_reader)
    fps = video_reader.get_avg_fps()
    
    print(f"Video info: {num_frames} frames, {fps:.2f} fps")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Process frames in batches
    all_embeddings = []
    
    print("Processing frames...")
    for i in range(0, num_frames, batch_size):
        batch_indices = list(range(i, min(i + batch_size, num_frames)))
        
        # Load batch of frames
        frames = video_reader.get_batch(batch_indices)
        
        # Convert to torch format and normalize
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        # Resize to target size (360, 640)
        if frames.shape[2:] != (360, 640):
            frames = torch.nn.functional.interpolate(frames, size=(360, 640), mode='bilinear', align_corners=False)
        
        frames = frames.to(device)
        
        # Get embeddings
        with torch.no_grad():
            _, _, _, latent = model(frames)
        
        all_embeddings.append(latent.cpu())
        
        print(f"Processed frames {i+1}-{min(i+batch_size, num_frames)}/{num_frames}")
    
    # Concatenate all embeddings
    embeddings = torch.cat(all_embeddings, dim=0)
    
    # Save embeddings
    torch.save(embeddings, output_path / "embeddings.pt")
    
    # Save metadata
    metadata = {
        "video_path": video_path,
        "num_frames": num_frames,
        "fps": fps,
        "duration_seconds": num_frames / fps,
        "shape": embeddings.shape,
        "checkpoint_path": checkpoint_path
    }
    torch.save(metadata, output_path / "metadata.pt")
    
    print(f"Embeddings saved to: {output_path}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Video duration: {metadata['duration_seconds']:.2f} seconds")
    
    return embeddings, metadata

def main():
    parser = argparse.ArgumentParser(description="Generate mock VAE embeddings or embed videos")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of embedding samples")
    parser.add_argument("--num-patches", type=int, default=576, help="Number of patches per sample")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--output-dir", default="./encoded", help="Output directory")
    parser.add_argument("--video", type=str, help="Path to video file to embed")
    parser.add_argument("--checkpoint", type=str, help="Path to VAE checkpoint for video embedding")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for video processing")
    
    args = parser.parse_args()
    
    if args.video and args.checkpoint:
        embed_video(
            video_path=args.video,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
    else:
        generate_mock_embeddings(
            num_samples=args.num_samples,
            num_patches=args.num_patches,
            latent_dim=args.latent_dim,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main() 