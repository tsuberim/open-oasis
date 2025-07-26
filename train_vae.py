import os
import torch
import torch.nn as nn
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
    Efficiently loads video frames from files using decord.
    Each worker process maintains its own video reader object.
    """
    def __init__(self, videos_dir="./videos", target_size=(360, 640)):
        self.videos_dir = Path(videos_dir)
        self.target_size = target_size
        
        # Get all video files
        self.video_files = sorted(list(self.videos_dir.glob("*.mp4"))) + \
                           sorted(list(self.videos_dir.glob("*.avi"))) + \
                           sorted(list(self.videos_dir.glob("*.mov")))
        
        if not self.video_files:
            raise ValueError(f"No video files found in {videos_dir}")
            
        # This dictionary will cache the video reader objects in each worker process
        self.video_readers = {}
        
        self.frame_indices = []
        print("Scanning video lengths...")
        for i, video_file in enumerate(tqdm(self.video_files, desc="Scanning Videos")):
            # Open with decord to get frame count
            vr = decord.VideoReader(str(video_file))
            num_frames = len(vr)
            # Create a tuple of (video_index, frame_index) for each frame
            for frame_idx in range(num_frames):
                self.frame_indices.append((i, frame_idx))

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        # Get the video and frame index for the current item
        video_idx, frame_idx = self.frame_indices[idx]
        
        # Get worker-specific information
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1

        # Each worker process maintains its own cache of video readers
        if worker_id not in self.video_readers:
            self.video_readers[worker_id] = {}

        if video_idx not in self.video_readers[worker_id]:
            # If the video reader for this video is not open in this worker, open it
            video_path = self.video_files[video_idx]
            # You can also specify ctx=decord.gpu(0) to use GPU for decoding
            self.video_readers[worker_id][video_idx] = decord.VideoReader(str(video_path))
        
        # Read the frame directly using the cached video reader
        frame = self.video_readers[worker_id][video_idx][frame_idx].numpy()
        
        # Preprocess frame
        frame = self.preprocess_frame(frame)
        
        # Convert to torch tensor (C, H, W)
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1)
        
        return frame_tensor

    def preprocess_frame(self, frame):
        """Preprocess a single frame (RGB)"""
        # Resize to target size (cv2.resize expects width, height)
        frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        return frame

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
                recon, posterior, latent = model.autoencode(frames, sample_posterior=True)
                
                # Calculate losses
                recon_loss = mse_loss(recon, frames)
                
                # Calculate KL divergence manually
                kl_loss = -0.5 * torch.sum(1 + posterior.logvar - posterior.mean.pow(2) - posterior.logvar.exp())
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
                        batch_recon, _, _ = model.autoencode(frames, sample_posterior=False)
                        
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
                    recon, posterior, latent = model.autoencode(frames, sample_posterior=False)
                    
                    # Calculate losses
                    recon_loss = mse_loss(recon, frames)
                    
                    # Calculate KL divergence manually
                    kl_loss = -0.5 * torch.sum(1 + posterior.logvar - posterior.mean.pow(2) - posterior.logvar.exp())
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
    parser.add_argument("--beta", "-b", type=float, default=0.00005, help="Beta coefficient for KL divergence loss")
    
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