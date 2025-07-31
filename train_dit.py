import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import random
import wandb
import time
from safetensors.torch import save_file, load_file
from dit import DiT_models
from utils import get_device, sigmoid_beta_schedule

# Set the backend to be thread-safe for DataLoader
import decord
decord.bridge.set_bridge('torch')

class EncodedFramesDataset(Dataset):
    """
    Dataset for loading encoded frames from the ./encoded directory.
    """
    def __init__(self, encoded_dir="./encoded", sequence_length=32):
        self.encoded_dir = Path(encoded_dir)
        self.sequence_length = sequence_length
        
        # Load embeddings
        embeddings_path = self.encoded_dir / "embeddings.pt"
        if not embeddings_path.exists():
            raise ValueError(f"Embeddings file not found: {embeddings_path}")
        
        self.embeddings = torch.load(embeddings_path)
        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        
        # Calculate number of possible sequences
        self.num_sequences = max(0, len(self.embeddings) - sequence_length + 1)
        print(f"Number of possible sequences: {self.num_sequences}")
        
        if self.num_sequences == 0:
            raise ValueError(f"Not enough frames ({len(self.embeddings)}) for sequence length {sequence_length}")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Get contiguous sequence of frames
        start_idx = idx
        end_idx = start_idx + self.sequence_length
        
        # Shape: (sequence_length, num_patches, latent_dim)
        sequence = self.embeddings[start_idx:end_idx]
        
        return sequence

def train_dit(model, train_loader, val_loader, device, num_epochs=100, lr=1e-4, checkpoint_dir="./dit_checkpoints"):
    """Train the DiT model"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    best_val_loss = float('inf')
    start_epoch = 0
    last_checkpoint_time = time.time()
    checkpoint_interval = 600  # 10 minutes in seconds
    global_batch_count = 0
    
    # Try to load latest checkpoint for resuming
    latest_checkpoint_path = checkpoint_dir / 'dit_checkpoint_latest.safetensors'
    latest_metadata_path = checkpoint_dir / 'dit_checkpoint_latest_metadata.pth'
    
    if latest_checkpoint_path.exists():
        print(f"Found existing checkpoint: {latest_checkpoint_path}")
        
        try:
            print(f"Loading model weights from {latest_checkpoint_path}")
            model_state_dict = load_file(latest_checkpoint_path)
            model.load_state_dict(model_state_dict)
            print("Model weights loaded successfully")
            
            if latest_metadata_path.exists():
                try:
                    print(f"Loading metadata from {latest_metadata_path}")
                    metadata = torch.load(latest_metadata_path, map_location=device)
                    optimizer.load_state_dict(metadata['optimizer_state_dict'])
                    scheduler.load_state_dict(metadata['scheduler_state_dict'])
                    start_epoch = metadata['epoch']
                    best_val_loss = metadata['best_val_loss']
                    global_batch_count = metadata.get('global_batch_count', 0)
                    print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
                    
                except Exception as e:
                    print(f"Error loading metadata: {e}")
                    print("Continuing with loaded model weights but fresh training state...")
                    start_epoch = 0
                    best_val_loss = float('inf')
                    global_batch_count = 0
            else:
                print("No metadata file found, continuing with loaded model weights but fresh training state...")
                start_epoch = 0
                best_val_loss = float('inf')
                global_batch_count = 0
                
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Starting training from scratch...")
            start_epoch = 0
            best_val_loss = float('inf')
            global_batch_count = 0
    
    # Set up noise schedule
    max_noise_level = 1000
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    print(f"Starting training from epoch {start_epoch + 1}")
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as pbar:
            for batch_idx, frames in enumerate(pbar):
                global_batch_count += 1
                frames = frames.to(device)  # Shape: (batch_size, sequence_length, num_patches, latent_dim)
                
                batch_size, sequence_length, num_patches, latent_dim = frames.shape
                
                # Randomize noise levels for each frame in the sequence
                t = torch.randint(0, max_noise_level, (batch_size, sequence_length), device=device)
                
                # Add noise to frames
                noise = torch.randn_like(frames)
                alphas_t = alphas_cumprod[t].view(batch_size, sequence_length, 1, 1)
                noisy_frames = alphas_t.sqrt() * frames + (1 - alphas_t).sqrt() * noise
                
                # Forward pass
                optimizer.zero_grad()
                
                # Reshape for DiT input: (batch_size, sequence_length, num_patches, latent_dim) -> (batch_size, sequence_length, latent_dim, H, W)
                # Assuming num_patches = H * W, we need to reshape
                H = int(np.sqrt(num_patches))  # Assuming square patches
                W = num_patches // H
                
                if H * W != num_patches:
                    # If not perfect square, pad or crop
                    H = W = int(np.sqrt(num_patches))
                    if H * W < num_patches:
                        H += 1
                    W = num_patches // H
                
                # Reshape to (batch_size, sequence_length, latent_dim, H, W)
                frames_reshaped = frames.view(batch_size, sequence_length, latent_dim, H, W)
                noisy_frames_reshaped = noisy_frames.view(batch_size, sequence_length, latent_dim, H, W)
                
                # Get model predictions
                predicted_noise = model(noisy_frames_reshaped, t)
                
                # Compute loss (MSE on the predicted noise vs actual noise)
                loss = nn.MSELoss()(predicted_noise, noise.view(batch_size, sequence_length, latent_dim, H, W))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # Log training metrics
                wandb.log({
                    'train_batch_loss': loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch + 1,
                    'global_step': global_batch_count
                }, step=global_batch_count)
                
                # Save checkpoint every 10 minutes
                current_time = time.time()
                if current_time - last_checkpoint_time >= checkpoint_interval:
                    checkpoint_path = checkpoint_dir / 'dit_checkpoint_latest.safetensors'
                    metadata_path = checkpoint_dir / 'dit_checkpoint_latest_metadata.pth'
                    
                    # Save model weights
                    print(f"  Saving model weights to {checkpoint_path}")
                    save_file(model.state_dict(), checkpoint_path)
                    
                    # Save metadata
                    print(f"  Saving metadata to {metadata_path}")
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': train_loss / (batch_idx + 1),
                        'best_val_loss': best_val_loss,
                        'timestamp': current_time,
                        'global_batch_count': global_batch_count,
                        'wandb_run_id': wandb.run.id if wandb.run else None,
                    }, metadata_path)
                    
                    last_checkpoint_time = current_time
                    print(f"  Saved checkpoint at batch {global_batch_count}, epoch {epoch+1}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as pbar:
                for batch_idx, frames in enumerate(pbar):
                    frames = frames.to(device)
                    
                    batch_size, sequence_length, num_patches, latent_dim = frames.shape
                    
                    # Randomize noise levels
                    t = torch.randint(0, max_noise_level, (batch_size, sequence_length), device=device)
                    
                    # Add noise to frames
                    noise = torch.randn_like(frames)
                    alphas_t = alphas_cumprod[t].view(batch_size, sequence_length, 1, 1)
                    noisy_frames = alphas_t.sqrt() * frames + (1 - alphas_t).sqrt() * noise
                    
                    # Reshape for DiT input
                    H = int(np.sqrt(num_patches))
                    W = num_patches // H
                    if H * W != num_patches:
                        H = W = int(np.sqrt(num_patches))
                        if H * W < num_patches:
                            H += 1
                        W = num_patches // H
                    
                    frames_reshaped = frames.view(batch_size, sequence_length, latent_dim, H, W)
                    noisy_frames_reshaped = noisy_frames.view(batch_size, sequence_length, latent_dim, H, W)
                    
                    # Get model predictions
                    predicted_noise = model(noisy_frames_reshaped, t)
                    
                    # Compute loss
                    loss = nn.MSELoss()(predicted_noise, noise.view(batch_size, sequence_length, latent_dim, H, W))
                    
                    val_loss += loss.item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}'
                    })
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Step scheduler
        scheduler.step(val_loss)
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'global_step': global_batch_count,
        }, step=global_batch_count)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Update best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best validation loss: {val_loss:.4f}")
            
            # Save best model
            best_model_path = checkpoint_dir / 'dit_checkpoint_best.safetensors'
            save_file(model.state_dict(), best_model_path)
            print(f"  Saved best model to {best_model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train DiT model on encoded frames")
    parser.add_argument("--encoded-dir", default="./encoded", help="Directory with encoded frames")
    parser.add_argument("--sequence-length", type=int, default=32, help="Length of frame sequences")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", default="./dit_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--wandb-project", default="dit-training", help="Weights & Biases project name")
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
        name=args.wandb_run_name and f"{args.wandb_run_name}-dit-train",
        config={
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
        }
    )
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = EncodedFramesDataset(
        encoded_dir=args.encoded_dir,
        sequence_length=args.sequence_length
    )
    
    # Split dataset
    test_size = int(len(dataset) * args.test_ratio)
    train_size = len(dataset) - test_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train set: {len(train_dataset)} sequences")
    print(f"Test set: {len(test_dataset)} sequences")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    # Create DiT model
    print("Creating DiT model...")
    model = DiT_models["DiT-S/2"]()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"DiT model created with {total_params:,} parameters")
    
    model = model.to(device)
    
    # Train the model
    print("Starting training...")
    train_dit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=checkpoint_dir
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main() 