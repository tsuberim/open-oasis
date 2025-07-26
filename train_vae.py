import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, get_worker_info
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import random
import wandb
import decord
from vae import VAE_models
from utils import get_device

# Set the backend to be thread-safe for DataLoader
decord.bridge.set_bridge('torch')

class VideoDataset(Dataset):
    """
    Optimal video dataset using decord for efficient random frame access.
    
    Key optimizations:
    - Worker-specific video reader caching
    - Direct frame indexing without seeking overhead
    - Batch frame loading for efficiency
    - Thread-safe design for multi-worker DataLoader
    - Automatic memory management
    """
    def __init__(self, videos_dir="./videos", target_size=(360, 640), 
                 ctx=None, num_threads=0, fault_tol=-1):
        self.videos_dir = Path(videos_dir)
        self.target_size = target_size
        
        # Set context for video decoding (CPU by default)
        if ctx is None:
            ctx = decord.cpu(0)
        self.ctx = ctx
        self.num_threads = num_threads
        self.fault_tol = fault_tol
        
        # Get all video files
        self.video_files = sorted(list(self.videos_dir.glob("*.mp4")) + 
                                 list(self.videos_dir.glob("*.avi")) + 
                                 list(self.videos_dir.glob("*.mov")))
        
        if not self.video_files:
            raise ValueError(f"No video files found in {videos_dir}")
        
        # Worker-specific video reader cache
        self._video_readers = {}
        
        # Build frame index mapping
        self.frame_indices = []
        self.video_metadata = []
        
        print(f"Scanning {len(self.video_files)} videos...")
        for video_idx, video_file in enumerate(tqdm(self.video_files, desc="Building frame index")):
            # Get video metadata without loading full video
            temp_reader = decord.VideoReader(
                str(video_file), 
                ctx=self.ctx
            )
            
            num_frames = len(temp_reader)
            fps = temp_reader.get_avg_fps()
            duration = num_frames / fps if fps > 0 else 0
            
            # Store metadata
            self.video_metadata.append({
                'path': video_file,
                'num_frames': num_frames,
                'fps': fps,
                'duration': duration,
                'width': temp_reader[0].shape[1],  # width from first frame
                'height': temp_reader[0].shape[0]  # height from first frame
            })
            
            # Create frame indices for this video
            for frame_idx in range(num_frames):
                self.frame_indices.append((video_idx, frame_idx))
            
            # Close temporary reader
            del temp_reader
        
        if not self.video_files:
            raise ValueError(f"No valid video files found in {videos_dir}. All videos failed to load.")
        
        print(f"Total frames available: {len(self.frame_indices):,}")
        print(f"Total duration: {sum(m['duration'] for m in self.video_metadata):.1f}s")
        
        # Print video summary
        for i, meta in enumerate(self.video_metadata):
            print(f"  Video {i}: {meta['num_frames']} frames, {meta['duration']:.1f}s, "
                  f"{meta['width']}x{meta['height']} @ {meta['fps']:.1f}fps")

    def __len__(self):
        return len(self.frame_indices)

    def _get_worker_readers(self):
        """Get worker-specific video reader cache."""
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        
        if worker_id not in self._video_readers:
            self._video_readers[worker_id] = {}
        
        return self._video_readers[worker_id]

    def _get_video_reader(self, video_idx):
        """Get or create video reader for specific video."""
        readers = self._get_worker_readers()
        
        if video_idx not in readers:
            video_file = self.video_metadata[video_idx]['path']
            
            # Create optimized video reader
            readers[video_idx] = decord.VideoReader(
                str(video_file),
                ctx=self.ctx
            )
        
        return readers[video_idx]

    def __getitem__(self, idx):
        """Get a single frame with optimal performance."""
        video_idx, frame_idx = self.frame_indices[idx]
        
        # Get cached video reader
        reader = self._get_video_reader(video_idx)
        
        # Direct frame access (decord handles seeking optimally)
        frame = reader[frame_idx]
        
        # Frame is already in torch format due to bridge setting
        # Shape: (H, W, C) -> convert to (C, H, W)
        if frame.dim() == 3:
            frame = frame.permute(2, 0, 1)
        
        # Normalize to [0, 1] if not already
        if frame.dtype == torch.uint8:
            frame = frame.float() / 255.0
        
        # Resize to target size
        if frame.shape[1:] != self.target_size:
            frame = F.interpolate(frame.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        
        return frame

    def get_batch(self, indices):
        """Get multiple frames efficiently using decord's batch loading."""
        if not indices:
            return torch.empty(0, 3, *self.target_size)
        
        # Group indices by video for efficient batch loading
        video_batches = {}
        for idx in indices:
            video_idx, frame_idx = self.frame_indices[idx]
            if video_idx not in video_batches:
                video_batches[video_idx] = []
            video_batches[video_idx].append((idx, frame_idx))
        
        # Load frames from each video
        frames = []
        for video_idx, batch_indices in video_batches.items():
            reader = self._get_video_reader(video_idx)
            frame_indices = [fi for _, fi in batch_indices]
            
            # Use decord's efficient batch loading
            batch_frames = reader.get_batch(frame_indices)
            
            # Convert to torch format
            if batch_frames.dim() == 3:
                batch_frames = batch_frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            elif batch_frames.dim() == 4:
                batch_frames = batch_frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            
            # Normalize if needed
            if batch_frames.dtype == torch.uint8:
                batch_frames = batch_frames.float() / 255.0
            
            # Resize to target size
            if batch_frames.shape[2:] != self.target_size:
                batch_frames = F.interpolate(batch_frames, size=self.target_size, mode='bilinear', align_corners=False)
            
            frames.append(batch_frames)
        
        # Concatenate all frames
        return torch.cat(frames, dim=0)

    def get_video_info(self, video_idx):
        """Get information about a specific video."""
        if 0 <= video_idx < len(self.video_metadata):
            return self.video_metadata[video_idx]
        return None

    def get_frame_info(self, idx):
        """Get information about a specific frame."""
        if 0 <= idx < len(self.frame_indices):
            video_idx, frame_idx = self.frame_indices[idx]
            video_info = self.video_metadata[video_idx]
            return {
                'video_idx': video_idx,
                'frame_idx': frame_idx,
                'video_path': video_info['path'],
                'timestamp': frame_idx / video_info['fps'] if video_info['fps'] > 0 else 0
            }
        return None

    def __del__(self):
        """Cleanup video readers."""
        for worker_readers in self._video_readers.values():
            for reader in worker_readers.values():
                del reader
        self._video_readers.clear()

def train_vae(model, train_loader, val_loader, device, num_epochs=100, lr=1e-4, beta=0.00005, checkpoint_dir="./checkpoints", resume_path=None):
    """Train the VAE model"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Loss function for reconstruction
    mse_loss = nn.MSELoss()
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if provided
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # Handle DataParallel state dict loading
        model_state_dict = checkpoint['model_state_dict']
        if isinstance(model, nn.DataParallel):
            # If current model is DataParallel, load into the module
            model.module.load_state_dict(model_state_dict)
        else:
            # If current model is not DataParallel, load directly
            model.load_state_dict(model_state_dict)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch-1}, best val loss: {best_val_loss:.4f}")
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as pbar:
            for batch_idx, frames in enumerate(pbar):
                frames = frames.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                
                # VAE forward pass
                recon, posterior_mean, posterior_logvar, latent = model(frames)
                
                # Calculate losses
                recon_loss = mse_loss(recon, frames)
                
                # Calculate KL divergence manually
                kl_loss = -0.5 * torch.sum(1 + posterior_logvar - posterior_mean.pow(2) - posterior_logvar.exp())
                kl_loss = kl_loss / frames.size(0)  # Normalize by batch size
                kl_loss = beta * kl_loss  # Apply beta coefficient
                
                total_loss = recon_loss + kl_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += total_loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })
                
                # Log sample reconstructions during training every 100 batches
                if batch_idx % 100 == 0:
                    with torch.no_grad():
                        # Generate reconstructions for current batch
                        batch_recon, _, _, _ = model(frames)
                        
                        # Convert to images for wandb
                        train_images = []
                        for i in range(min(4, len(frames))):
                            # Original image
                            orig = frames[i].cpu().permute(1, 2, 0).numpy()
                            orig = (orig * 255).astype(np.uint8)
                            
                            # Reconstructed image
                            rec = batch_recon[i].cpu().permute(1, 2, 0).numpy()
                            rec = np.clip(rec * 255, 0, 255).astype(np.uint8)
                            
                            # Combine original and reconstruction side by side
                            combined = np.concatenate([orig, rec], axis=1)
                            train_images.append(wandb.Image(combined, caption=f"Train Original | Reconstruction"))
                        
                        wandb.log({
                            'train_samples': train_images,
                            'train_batch': batch_idx,
                            'epoch': epoch + 1
                        })
                

        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        sample_images = []  # Store sample images for logging
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as pbar:
                for batch_idx, frames in enumerate(pbar):
                    frames = frames.to(device)
                    
                    # VAE forward pass
                    recon, posterior_mean, posterior_logvar, latent = model(frames)
                    
                    # Calculate losses
                    recon_loss = mse_loss(recon, frames)
                    
                    # Calculate KL divergence manually
                    kl_loss = -0.5 * torch.sum(1 + posterior_logvar - posterior_mean.pow(2) - posterior_logvar.exp())
                    kl_loss = kl_loss / frames.size(0)  # Normalize by batch size
                    kl_loss = beta * kl_loss  # Apply beta coefficient
                    
                    total_loss = recon_loss + kl_loss
                    
                    # Update metrics
                    val_loss += total_loss.item()
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_loss.item()
                    
                    # Capture sample images from first batch
                    if batch_idx == 0:
                        for i in range(min(4, len(frames))):
                            # Original image
                            orig = frames[i].cpu().permute(1, 2, 0).numpy()
                            orig = (orig * 255).astype(np.uint8)
                            
                            # Reconstructed image
                            rec = recon[i].cpu().permute(1, 2, 0).numpy()
                            rec = np.clip(rec * 255, 0, 255).astype(np.uint8)
                            
                            # Combine original and reconstruction side by side
                            combined = np.concatenate([orig, rec], axis=1)
                            sample_images.append(wandb.Image(combined, caption=f"Val Original | Reconstruction"))
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{total_loss.item():.4f}',
                        'recon': f'{recon_loss.item():.4f}',
                        'kl': f'{kl_loss.item():.4f}'
                    })
                    

        
        # Calculate averages
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_kl_loss /= len(val_loader)
        

        
        # Log to wandb
        log_dict = {
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'train/recon_loss': train_recon_loss,
            'train/kl_loss': train_kl_loss,
            'val/loss': val_loss,
            'val/recon_loss': val_recon_loss,
            'val/kl_loss': val_kl_loss,
            'lr': scheduler.get_last_lr()[0],
        }
        
        # Add validation sample images
        log_dict['val_samples'] = sample_images
        
        wandb.log(log_dict)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Recon: {val_recon_loss:.4f}, KL: {val_kl_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = checkpoint_dir / 'best_vae_model.pth'
            
            # Handle DataParallel state dict
            if isinstance(model, nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'best_val_loss': best_val_loss,
            }, best_model_path)
            print(f"  Saved best model with val_loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f'vae_checkpoint_epoch_{epoch+1}.pth'
            
            # Handle DataParallel state dict
            if isinstance(model, nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)

def main():
    parser = argparse.ArgumentParser(description="Train VAE on videos with on-the-fly preprocessing")
    parser.add_argument("--videos-dir", "-d", default="./videos", help="Directory with raw videos")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", "-l", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--test-ratio", "-t", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--beta", "-B", type=float, default=0.00005, help="Beta coefficient for KL divergence loss")
    parser.add_argument("--target-size", "-T", nargs=2, type=int, default=[360, 640], help="Target frame size (height width)")
    parser.add_argument("--checkpoint-dir", "-c", default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", "-r", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb-project", default="world-model", help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", default=None, help="Weights & Biases run name")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=f"{args.wandb_run_name}-vae-train",
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "test_ratio": args.test_ratio,
            "beta": args.beta,
            "target_size": args.target_size,
            "seed": args.seed,
        }
    )
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = VideoDataset(
        videos_dir=args.videos_dir,
        target_size=tuple(args.target_size)
    )
    
    # Split dataset
    test_size = int(len(dataset) * args.test_ratio)
    train_size = len(dataset) - test_size
    
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train set: {len(train_dataset)} frames")
    print(f"Test set: {len(test_dataset)} frames")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create VAE model
    print("Creating VAE model...")
    model = VAE_models["vit-small-shallow-encoder"]()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VAE model created with {total_params:,} parameters")
    
    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Train the model
    print("Starting training...")
    train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        checkpoint_dir=checkpoint_dir,
        resume_path=args.resume
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main() 