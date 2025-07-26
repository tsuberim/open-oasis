import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from pathlib import Path
import cv2
from vae import VAE_models
from utils import get_device
from safetensors.torch import load_model


def generate_random_frame(model, device, num_frames=1, save_path=None):
    """Generate random frames by sampling from the VAE latent space."""
    model.eval()
    
    with torch.no_grad():
        # Get the latent space dimensions from the model
        if hasattr(model, 'module'):
            seq_len = model.module.seq_len
            latent_dim = model.module.latent_dim
        else:
            seq_len = model.seq_len
            latent_dim = model.latent_dim
        
        # Create a diagonal normal distribution with learned parameters
        # For generation, we typically use mean=0 and std=1 (prior distribution)
        # But we can also use learned statistics from the VAE
        mean = torch.zeros(num_frames, seq_len, latent_dim).to(device)
        logvar = torch.zeros(num_frames, seq_len, latent_dim).to(device)
        
        # Create the diagonal normal distribution
        from vae import DiagonalGaussianDistribution
        posterior = DiagonalGaussianDistribution(torch.cat([mean, logvar], dim=2), dim=2)
        
        # Sample from the diagonal normal distribution
        z = posterior.sample()
        
        # Decode the latent vectors to generate frames
        generated_frames = model.module.decode(z) if hasattr(model, 'module') else model.decode(z)
        
        # Convert to images
        frames = []
        for i in range(num_frames):
            # Convert from (C, H, W) to (H, W, C) and normalize to [0, 255]
            frame = generated_frames[i].cpu().permute(1, 2, 0).numpy()
            
            # Debug: print the actual range of values
            print(f"Frame {i} - Min: {frame.min():.4f}, Max: {frame.max():.4f}, Mean: {frame.mean():.4f}")
            
            # Try different normalization approaches
            if frame.min() >= -1 and frame.max() <= 1:
                # If in [-1, 1] range, convert to [0, 1]
                frame = (frame + 1) / 2
            elif frame.min() >= 0 and frame.max() <= 1:
                # Already in [0, 1] range
                pass
            else:
                # Try sigmoid normalization
                frame = 1 / (1 + np.exp(-frame))
            
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frames.append(frame)
        
        # Save frames if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure we have a proper file extension
            if not save_path.suffix:
                save_path = save_path.with_suffix('.png')
            
            if num_frames == 1:
                cv2.imwrite(str(save_path), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
                print(f"Generated frame saved to: {save_path}")
            else:
                for i, frame in enumerate(frames):
                    frame_path = save_path.parent / f"{save_path.stem}_{i:03d}{save_path.suffix}"
                    cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"Generated {num_frames} frames saved to: {save_path.parent}")
        
        return frames


def main():
    parser = argparse.ArgumentParser(description="Generate random frames using trained VAE")
    parser.add_argument("--checkpoint", type=str, help="Path to VAE checkpoint (optional - will use random weights if not provided)")
    parser.add_argument("--model_type", type=str, default="vit-small-shallow-encoder", 
                       choices=["vit-small-shallow-encoder", "vit-l-20-shallow-encoder"],
                       help="VAE model type")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to generate")
    parser.add_argument("--output", type=str, default="generated_frames/frame.png", help="Output path for generated frame(s)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading VAE model: {args.model_type}")
    model_class = VAE_models[args.model_type]
    model = model_class()
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        if args.checkpoint.endswith(".pt"):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        elif args.checkpoint.endswith(".safetensors"):
            load_model(model, args.checkpoint)
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint provided - using random weights")
    
    model = model.to(device).eval()
    
    print(f"Model loaded successfully. Generating {args.num_frames} frame(s)...")
    
    # Generate frames
    frames = generate_random_frame(
        model=model,
        device=device,
        num_frames=args.num_frames,
        save_path=args.output
    )
    
    print("Generation complete!")


if __name__ == "__main__":
    main() 