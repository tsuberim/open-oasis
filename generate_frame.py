import torch
import torchvision.utils as vutils
from vae import VAE_models
from utils import get_device
from safetensors.torch import load_model
import argparse


def generate_frame(model, device, output_path="generated_frame.png"):
    """Generate a single frame from random latent encoding."""
    model.eval()
    
    # Get the required shape parameters from the model
    latent_dim = model.latent_dim
    sequence_length = model.seq_len  # This is (H/patch_size) * (W/patch_size)
    batch_size = 1  # Generate 1 image
    
    print(f"Model parameters:")
    print(f"  latent_dim: {latent_dim}")
    print(f"  sequence_length: {sequence_length}")
    print(f"  batch_size: {batch_size}")
    
    # Create a random latent code from the prior distribution (N(0, 1))
    # Shape: (batch_size, sequence_length, latent_dim)
    random_z = torch.randn(batch_size, sequence_length, latent_dim).to(device)
    print(f"Random latent shape: {random_z.shape}")
    
    # Decode the random latent code to generate an image
    with torch.no_grad():
        generated_images = model.decode(random_z)
    
    # The output is a tensor of shape (batch_size, 3, height, width)
    print(f"Shape of generated images: {generated_images.shape}")
    
    # The output is typically in the range [0, 1] or [-1, 1].
    # If your model doesn't use a final tanh/sigmoid, you might need to clamp it.
    generated_images = torch.clamp(generated_images, 0, 1)
    
    # Save the generated image
    vutils.save_image(generated_images, output_path, nrow=1)
    print(f"Saved generated image to {output_path}")
    
    return generated_images


def main():
    parser = argparse.ArgumentParser(description="Generate a single frame using VAE")
    parser.add_argument("--model_type", type=str, default="vit-l-20-shallow-encoder", 
                       choices=["vit-small-shallow-encoder", "vit-l-20-shallow-encoder"],
                       help="VAE model type")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to VAE safetensors checkpoint")
    parser.add_argument("--output", type=str, default="generated_frame.png", help="Output image path")
    
    args = parser.parse_args()
    
    # Set the device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize the VAE model
    print(f"Loading VAE model: {args.model_type}")
    model = VAE_models[args.model_type]()
    
    # Load the safetensors checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_model(model, args.checkpoint)
    
    # Set the device and put the model in evaluation mode
    model.to(device)
    model.eval()
    
    # Generate the frame
    generate_frame(model, device, args.output)


if __name__ == "__main__":
    main() 