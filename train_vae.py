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
import time
import signal
import sys
import itertools
from safetensors.torch import save_file, load_file
from vae import VAE_models
from utils import get_device, GaussianPyramidLoss, LaplacianPyramidLoss

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

def save_checkpoint_on_signal(model, optimizer, scheduler, epoch, best_val_loss, current_beta, checkpoint_dir, global_batch_count, batch_idx=0):
    """Save checkpoint when receiving termination signal"""
    print(f"\nReceived termination signal. Saving checkpoint...")
    
    checkpoint_path = checkpoint_dir / 'vae_checkpoint_latest.safetensors'
    metadata_path = checkpoint_dir / 'vae_checkpoint_latest_metadata.pth'
    
    try:
        # Handle DataParallel state dict
        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        # Save model weights with safetensors
        print(f"  Saving model weights to {checkpoint_path}")
        save_file(model_state_dict, checkpoint_path)
        
        # Save metadata separately
        print(f"  Saving metadata to {metadata_path}")
        torch.save({
            'epoch': epoch,
            'batch_idx': batch_idx,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'timestamp': time.time(),
            'current_beta': current_beta,
            'global_batch_count': global_batch_count,
            'wandb_run_id': wandb.run.id if wandb.run else None,
        }, metadata_path)
        
        print(f"  Checkpoint saved successfully at epoch {epoch+1}, batch {global_batch_count}")
        
    except Exception as e:
        print(f"  Error saving checkpoint: {e}")
    
    print("Exiting gracefully...")
    sys.exit(0)

def train_vae(model, train_loader, val_loader, device, num_epochs=100, lr=1e-4, beta=0.00005, beta_annealing=True, checkpoint_dir="./checkpoints"):
    """Train the VAE model"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=1500)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Loss functions for reconstruction
    #gaussian_loss = GaussianPyramidLoss(max_level=3, loss_func=nn.L1Loss())
    laplacian_loss = LaplacianPyramidLoss(max_level=3, loss_func=nn.L1Loss())
    best_val_loss = float('inf')
    start_epoch = 0
    last_checkpoint_time = time.time()
    checkpoint_interval = 600  # 10 minutes in seconds
    global_batch_count = 0
    
    # Try to load latest checkpoint for resuming
    latest_checkpoint_path = checkpoint_dir / 'vae_checkpoint_latest.safetensors'
    latest_metadata_path = checkpoint_dir / 'vae_checkpoint_latest_metadata.pth'
    
    if latest_checkpoint_path.exists():
        print(f"Found existing checkpoint: {latest_checkpoint_path}")
        
        # Try to load model weights first
        try:
            print(f"Loading model weights from {latest_checkpoint_path}")
            model_state_dict = load_file(latest_checkpoint_path)
            
            # Handle DataParallel state dict loading
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(model_state_dict)
            else:
                model.load_state_dict(model_state_dict)
            print("Model weights loaded successfully")
            
            # Try to load metadata if it exists
            if latest_metadata_path.exists():
                try:
                    print(f"Loading metadata from {latest_metadata_path}")
                    metadata = torch.load(latest_metadata_path, map_location=device)
                    optimizer.load_state_dict(metadata['optimizer_state_dict'])
                    scheduler.load_state_dict(metadata['scheduler_state_dict'])
                    start_epoch = metadata['epoch']
                    best_val_loss = metadata['best_val_loss']
                    
                    # Check if we need to resume from a specific batch within the epoch
                    if 'batch_idx' in metadata:
                        resume_batch_idx = metadata['batch_idx'] + 1
                        print(f"Resuming from epoch {start_epoch}, batch {resume_batch_idx} within epoch")
                    else:
                        resume_batch_idx = 0
                        print(f"Resuming from epoch {start_epoch}, starting from beginning of epoch")
                    
                    # Recalculate global_batch_count based on start_epoch and batch_idx
                    global_batch_count = start_epoch * len(train_loader) + resume_batch_idx
                    
                    print("Metadata loaded successfully")
                    print(f"Resumed from epoch {start_epoch}, batch {global_batch_count}, best val loss: {best_val_loss:.4f}")
                    
                except Exception as e:
                    print(f"Error loading metadata: {e}")
                    print("Continuing with loaded model weights but fresh training state...")
                    # Keep the loaded model but reset training state
                    start_epoch = 0
                    best_val_loss = float('inf')
                    global_batch_count = 0
                    resume_batch_idx = 0
            else:
                print("No metadata file found, continuing with loaded model weights but fresh training state...")
                start_epoch = 0
                best_val_loss = float('inf')
                global_batch_count = 0
                resume_batch_idx = 0
                
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Starting training from scratch...")
            # Reset to initial values
            start_epoch = 0
            best_val_loss = float('inf')
            global_batch_count = 0
            resume_batch_idx = 0
    
    # Override learning rate from function parameter
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4
        print(f"Overrode learning rate to {param_group['lr']}")

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        save_checkpoint_on_signal(model, optimizer, scheduler, epoch, best_val_loss, current_beta, checkpoint_dir, global_batch_count, actual_batch_idx if 'actual_batch_idx' in locals() else 0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Beta annealing setup
    if beta_annealing:
        initial_beta = 1e-5
        final_beta = beta
        beta_warmup_batches = 500  # Warm up over first 5000 batches
        current_beta = initial_beta  # Initialize current_beta
    else:
        current_beta = beta
    

    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", total=len(train_loader)) as pbar:
            for batch_idx, frames in enumerate(pbar):
                actual_batch_idx = batch_idx
                global_batch_count += 1
                
                # Calculate current beta for annealing (every batch)
                if beta_annealing:
                    if global_batch_count < beta_warmup_batches:
                        current_beta = initial_beta + (final_beta - initial_beta) * (global_batch_count / beta_warmup_batches)
                    else:
                        current_beta = final_beta
                else:
                    current_beta = beta
                
                frames = frames.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                
                # VAE forward pass
                recon, posterior_mean, posterior_logvar, latent = model(frames)
                
                # Calculate losses
                #recon_loss = gaussian_loss(recon, frames)  # Use only Gaussian pyramid loss
                recon_loss = laplacian_loss(recon, frames)  # Use only Gaussian pyramid loss
                
                # Calculate KL divergence manually
                kl_loss = -0.5 * torch.sum(1 + posterior_logvar - posterior_mean.pow(2) - posterior_logvar.exp())
                kl_loss = kl_loss / frames.size(0)  # Normalize by batch size
                kl_loss = current_beta * kl_loss  # Apply current beta coefficient
                
                total_loss = recon_loss + kl_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Step scheduler with current batch loss
                scheduler.step(total_loss.item())
                
                # Save checkpoint every 10 minutes during training
                current_time = time.time()
                if current_time - last_checkpoint_time >= checkpoint_interval:
                    checkpoint_path = checkpoint_dir / 'vae_checkpoint_latest.safetensors'
                    metadata_path = checkpoint_dir / 'vae_checkpoint_latest_metadata.pth'
                    
                    # Handle DataParallel state dict
                    if isinstance(model, nn.DataParallel):
                        model_state_dict = model.module.state_dict()
                    else:
                        model_state_dict = model.state_dict()
                    
                    # Save model weights with safetensors
                    print(f"  Saving model weights to {checkpoint_path}")
                    save_file(model_state_dict, checkpoint_path)
                    
                    # Save metadata separately (optimizer, scheduler, etc.)
                    print(f"  Saving metadata to {metadata_path}")
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': actual_batch_idx,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss if 'val_loss' in locals() else None,
                        'train_loss': train_loss / (actual_batch_idx + 1) if actual_batch_idx > 0 else total_loss.item(),
                        'best_val_loss': best_val_loss,
                        'timestamp': current_time,
                        'current_beta': current_beta,
                        'wandb_run_id': wandb.run.id if wandb.run else None,
                    }, metadata_path)
                    
                    last_checkpoint_time = current_time
                    print(f"  Saved checkpoint at batch {global_batch_count}, epoch {epoch+1} - {time.strftime('%H:%M:%S', time.localtime(current_time))}")
                
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
                
                # Log training metrics every batch
                try:
                    wandb.log({
                        'train_batch_loss': total_loss.item(),
                        'train_batch_recon_loss': recon_loss.item(),
                        'train_batch_kl_loss': kl_loss.item(),
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'beta': current_beta,
                        'epoch': epoch + 1,
                        'global_step': global_batch_count
                    }, step=epoch * len(train_loader) + actual_batch_idx)
                except Exception as e:
                    print(f"Warning: Failed to log to wandb: {e}")
                
                # Log sample reconstructions during training every 100 batches
                if actual_batch_idx % 50 == 0:
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
                        
                        try:
                            wandb.log({
                                'train_samples': train_images,
                                'train_batch': actual_batch_idx,
                                'epoch': epoch + 1
                            })
                        except Exception as e:
                            print(f"Warning: Failed to log train samples to wandb: {e}")
                

        
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
                    #recon_loss = gaussian_loss(recon, frames)  # Use only Gaussian pyramid loss
                    recon_loss = laplacian_loss(recon, frames)  # Use only Gaussian pyramid loss
                    
                    # Calculate KL divergence manually
                    kl_loss = -0.5 * torch.sum(1 + posterior_logvar - posterior_mean.pow(2) - posterior_logvar.exp())
                    kl_loss = kl_loss / frames.size(0)  # Normalize by batch size
                    kl_loss = current_beta * kl_loss  # Apply current beta coefficient
                    
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
            'train_loss': train_loss,
            'train_recon_loss': train_recon_loss,
            'train_kl_loss': train_kl_loss,
            'val_loss': val_loss,
            'val_recon_loss': val_recon_loss,
            'val_kl_loss': val_kl_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'beta': current_beta,
            'global_step': global_batch_count,
            'current_beta': current_beta,
            'recon_kl_ratio': train_recon_loss / (train_kl_loss + current_beta),
            'val_recon_kl_ratio': val_recon_loss / (val_kl_loss + current_beta),
        }
        
        # Add validation sample images
        log_dict['val_samples'] = sample_images
        
        try:
            wandb.log(log_dict, step=(epoch + 1) * len(train_loader))
        except Exception as e:
            print(f"Warning: Failed to log epoch metrics to wandb: {e}")
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Recon: {val_recon_loss:.4f}, KL: {val_kl_loss:.4f}")
        
        # Update best validation loss (but don't save checkpoint)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best validation loss: {val_loss:.4f}")
        


def main():
    parser = argparse.ArgumentParser(description="Train VAE on videos with on-the-fly preprocessing")
    parser.add_argument("--videos-dir", "-d", default="./videos", help="Directory with raw videos")
    parser.add_argument("--batch-size", "-b", type=int, default=96, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", "-l", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--test-ratio", "-t", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--beta", "-B", type=float, default=1e-4, help="Beta coefficient for KL divergence loss")
    parser.add_argument("--no-beta-annealing", action="store_true", help="Disable beta annealing")
    parser.add_argument("--target-size", "-T", nargs=2, type=int, default=[360, 640], help="Target frame size (height width)")
    parser.add_argument("--checkpoint-dir", "-c", default="./checkpoints", help="Directory to save checkpoints")
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
    
    # Check for existing wandb run ID in checkpoint metadata
    wandb_run_id = None
    latest_metadata_path = checkpoint_dir / 'vae_checkpoint_latest_metadata.pth'
    if latest_metadata_path.exists():
        try:
            metadata = torch.load(latest_metadata_path, map_location='cpu')
            wandb_run_id = metadata.get('wandb_run_id')
            if wandb_run_id:
                print(f"Found wandb run ID in checkpoint: {wandb_run_id}")
        except Exception as e:
            print(f"Error loading wandb run ID from checkpoint: {e}")
    
    # Initialize wandb
    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name and f"{args.wandb_run_name}-vae-train",
            id=wandb_run_id,  # Resume existing run if available
            resume="allow" if wandb_run_id else None,  # Allow resuming if run ID exists
            config={
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "test_ratio": args.test_ratio,
                "beta": args.beta,
                "beta_annealing": not args.no_beta_annealing,
                "target_size": args.target_size,
                "seed": args.seed,
            }
        )
        print(f"Wandb initialized successfully. Run URL: {wandb.run.url if wandb.run else 'N/A'}")
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        print("Training will continue without wandb logging")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = VideoDataset(
        videos_dir=args.videos_dir,
        target_size=tuple(args.target_size)
    )
    
    # Calculate and display total video hours
    total_duration_seconds = sum(m['duration'] for m in dataset.video_metadata)
    total_hours = total_duration_seconds / 3600
    print(f"\n{'='*50}")
    print(f"TOTAL VIDEO HOURS: {total_hours:.2f} hours")
    print(f"{'='*50}\n")
    
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
    model = VAE_models["vit-l-20-shallow-encoder"](latent_dim=32)
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
        beta_annealing=not args.no_beta_annealing,
        checkpoint_dir=checkpoint_dir
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main() 