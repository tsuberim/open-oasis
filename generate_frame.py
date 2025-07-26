import torch
import torchvision.utils as vutils
from vae import VAE_models
from utils import get_device
from safetensors.torch import load_file
import argparse
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib
try:
    matplotlib.use('MacOSX')  # Use MacOSX backend for macOS
except:
    try:
        matplotlib.use('Qt5Agg')  # Fallback to Qt5
    except:
        matplotlib.use('Agg')  # Non-interactive fallback
        print("Warning: Using non-interactive backend. Interactive features may not work.")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch.nn as nn



def load_and_preprocess_image(image_path, target_size=(360, 640)):
    """Load and preprocess an image for the VAE."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def generate_frame_from_prompt(model, device, prompt_image_path, noise_scale=0.1, output_path="generated_frame.png"):
    """Generate a frame by encoding a prompt image, adding noise, and decoding."""
    model.eval()
    
    try:
        # Load and preprocess the prompt image
        print(f"Loading prompt image: {prompt_image_path}")
        prompt_image = load_and_preprocess_image(prompt_image_path)
        prompt_image = prompt_image.to(device)
        print(f"Prompt image shape: {prompt_image.shape}")
        
        # Encode the prompt image to get latent representation
        print("Encoding prompt image...")
        with torch.no_grad():
            # Get the encoder output (posterior mean and logvar)
            _, posterior_mean, posterior_logvar, _ = model(prompt_image)
            
            # Sample from the posterior distribution
            std = torch.exp(0.5 * posterior_logvar)
            eps = torch.randn_like(std)
            latent_z = posterior_mean + eps * std
            
        print(f"Encoded latent shape: {latent_z.shape}")
        
        # Add noise to the latent representation
        print(f"Adding noise with scale: {noise_scale}")
        noise = torch.randn_like(latent_z) * noise_scale
        noisy_latent = latent_z + noise
        print(f"Noisy latent shape: {noisy_latent.shape}")
        
        # Decode the noisy latent to generate an image
        print("Decoding noisy latent...")
        with torch.no_grad():
            generated_images = model.decode(noisy_latent)
        
        # The output is a tensor of shape (batch_size, 3, height, width)
        print(f"Shape of generated images: {generated_images.shape}")
        
        # The output is typically in the range [0, 1] or [-1, 1].
        # If your model doesn't use a final tanh/sigmoid, you might need to clamp it.
        generated_images = torch.clamp(generated_images, 0, 1)
        
        # Save the generated image
        vutils.save_image(generated_images, output_path, nrow=1)
        print(f"Saved generated image to {output_path}")
        
        return generated_images
        
    except Exception as e:
        print(f"Error during generation: {e}")
        print("This might be due to model architecture mismatch or incomplete training.")
        return None

def generate_frame_random(model, device, output_path="generated_frame.png"):
    """Generate a single frame from random latent encoding."""
    model.eval()
    
    try:
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
        
    except Exception as e:
        print(f"Error during generation: {e}")
        print("This might be due to model architecture mismatch or incomplete training.")
        return None


def interactive_generation(model, device, prompt_image_path, output_path="interactive_generation.png"):
    """Interactive generation with sliders controlling latent space."""
    # Check if we have an interactive backend
    if matplotlib.get_backend() == 'Agg':
        print("Error: Non-interactive backend detected. Cannot launch interactive visualization.")
        print("Please install a GUI backend like:")
        print("  pip install tkinter  # or")
        print("  pip install PyQt5    # or")
        print("  brew install python-tk  # on macOS")
        return
    
    model.eval()
    
    # Test if the model actually generates different images
    print("Testing model generation...")
    with torch.no_grad():
        test_noise1 = torch.randn(1, model.seq_len, model.latent_dim).to(device) * 0.1
        test_noise2 = torch.randn(1, model.seq_len, model.latent_dim).to(device) * 1.0
        
        test_img1 = model.decode(test_noise1)
        test_img2 = model.decode(test_noise2)
        
        print(f"Test images - img1 range: [{test_img1.min():.3f}, {test_img1.max():.3f}], img2 range: [{test_img2.min():.3f}, {test_img2.max():.3f}]")
        print(f"Test images different: {torch.abs(test_img1 - test_img2).mean():.3f}")
    
    if torch.abs(test_img1 - test_img2).mean() < 0.01:
        print("Warning: Model seems to generate very similar images regardless of input!")
        print("This might indicate the model is not properly trained or loaded.")
        return
    
    # Load and preprocess the prompt image
    print(f"Loading prompt image: {prompt_image_path}")
    prompt_image = load_and_preprocess_image(prompt_image_path)
    prompt_image = prompt_image.to(device)
    
    # Encode the prompt image to get base latent representation
    with torch.no_grad():
        _, posterior_mean, posterior_logvar, _ = model(prompt_image)
        std = torch.exp(0.5 * posterior_logvar)
        base_latent = posterior_mean
    
    # Create 3 random orthogonal directions in latent space
    latent_dim = model.latent_dim
    sequence_length = model.seq_len
    
    print("Creating 3 random orthogonal directions in latent space...")
    
    # Generate 3 random vectors in latent_dim space
    directions = torch.randn(3, latent_dim, device=device)
    
    # Gram-Schmidt orthogonalization to make them orthogonal
    directions[0] = directions[0] / directions[0].norm()
    directions[1] = directions[1] - (directions[1] @ directions[0]) * directions[0]
    directions[1] = directions[1] / directions[1].norm()
    directions[2] = directions[2] - (directions[2] @ directions[0]) * directions[0] - (directions[2] @ directions[1]) * directions[1]
    directions[2] = directions[2] / directions[2].norm()
    
    # Scale the directions to have larger magnitudes for more visible effects
    directions = directions * 100000.0
    
    print("Orthogonal directions created!")
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(bottom=0.3)  # Make room for sliders
    
    # Display original prompt image
    prompt_display = prompt_image[0].cpu().permute(1, 2, 0).numpy()
    prompt_display = np.clip(prompt_display, 0, 1)
    ax1.imshow(prompt_display)
    ax1.set_title('Original Prompt Image')
    ax1.axis('off')
    
    # Initialize generated image display
    img_plot = ax2.imshow(np.zeros_like(prompt_display))
    ax2.set_title('Generated Image')
    ax2.axis('off')
    
    def update(val):
        nonlocal directions, posterior_mean, std, sequence_length, latent_dim
        print(f"Updating with params: {slider1.val:.3f}, {slider2.val:.3f}, {slider3.val:.3f}, noise_scale: {slider4.val:.3f}")
        # Get slider values
        param1 = slider1.val
        param2 = slider2.val
        param3 = slider3.val
        noise_scale = slider4.val
        
        # Generate noise using the orthogonal directions
        with torch.no_grad():
            # Linear combination of the three orthogonal directions
            direction_combination = param1 * directions[0] + param2 * directions[1] + param3 * directions[2]
            # Apply to each position in the sequence
            noise = direction_combination.unsqueeze(0).unsqueeze(0).expand(1, sequence_length, latent_dim)
            print(f"Slider values: {param1:.3f}, {param2:.3f}, {param3:.3f}")
            print(f"Direction magnitudes: {directions[0].norm():.3f}, {directions[1].norm():.3f}, {directions[2].norm():.3f}")
            print(f"Direction combination - mean: {noise.mean():.3f}, std: {noise.std():.3f}")
            

            print(noise)
            # Apply noise to the posterior mean (using std for proper VAE sampling)
            noise_scaled = noise * std * noise_scale
            sampled_latent = posterior_mean + noise_scaled
            print(f"Posterior mean shape: {posterior_mean.shape}")
            print(f"Noise shape: {noise.shape}")
            print(f"Scaled noise shape: {noise_scaled.shape}")
            print(f"Posterior mean stats - mean: {posterior_mean.mean():.3f}, std: {posterior_mean.std():.3f}")
            print(f"Scaled noise stats - mean: {noise_scaled.mean():.3f}, std: {noise_scaled.std():.3f}")
            print(f"Sampled latent stats - mean: {sampled_latent.mean():.3f}, std: {sampled_latent.std():.3f}")
            print(f"Noise stats - mean: {noise.mean():.3f}, std: {noise.std():.3f}, scale: {noise_scale:.3f}")
            
            # Decode to generate image
            generated_images = model.decode(sampled_latent)
            print(f"Raw model output - min: {generated_images.min():.3f}, max: {generated_images.max():.3f}, mean: {generated_images.mean():.3f}")
            
            # Apply basic normalization to ensure proper range
            generated_images = torch.clamp(generated_images, 0, 1)
            print("Applied clamp to [0,1] range")
            
            print(f"Final output - min: {generated_images.min():.3f}, max: {generated_images.max():.3f}, mean: {generated_images.mean():.3f}")
            
            # Update display
            generated_display = generated_images[0].cpu().permute(1, 2, 0).numpy()
            print(f"Generated image shape: {generated_display.shape}, range: [{generated_display.min():.3f}, {generated_display.max():.3f}]")
            
            # Clear and redraw the image completely
            ax2.clear()
            ax2.imshow(generated_display)
            ax2.set_title('Generated Image')
            ax2.axis('off')
            
            # Force complete redraw
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Small pause to ensure the update is processed
            plt.pause(0.01)
    
    # Create sliders
    ax_slider1 = plt.axes([0.2, 0.20, 0.6, 0.03])
    ax_slider2 = plt.axes([0.2, 0.15, 0.6, 0.03])
    ax_slider3 = plt.axes([0.2, 0.10, 0.6, 0.03])
    ax_slider4 = plt.axes([0.2, 0.05, 0.6, 0.03])
    
    slider1 = Slider(ax_slider1, 'Param 1', -3.0, 3.0, valinit=0.0)
    slider2 = Slider(ax_slider2, 'Param 2', -3.0, 3.0, valinit=0.0)
    slider3 = Slider(ax_slider3, 'Param 3', -3.0, 3.0, valinit=0.0)
    slider4 = Slider(ax_slider4, 'Noise Scale', 0.0, 10.0, valinit=.2)
    
    # Connect sliders to update function
    slider1.on_changed(update)
    slider2.on_changed(update)
    slider3.on_changed(update)
    slider4.on_changed(update)
    
    # Initial update
    print("Performing initial update...")
    update(None)
    
    # Use non-blocking show with more aggressive updates
    plt.ion()  # Turn on interactive mode
    plt.show(block=False)
    
    # Keep the window open with more frequent updates
    try:
        while plt.get_fignums():  # While any figure is open
            plt.pause(0.05)  # More frequent updates
            # Force a redraw every iteration
            fig.canvas.draw()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("Interactive session ended by user")
    finally:
        plt.ioff()  # Turn off interactive mode
    
    # Save the final generated image
    with torch.no_grad():
        # Linear combination of the three orthogonal directions
        direction_combination = slider1.val * directions[0] + slider2.val * directions[1] + slider3.val * directions[2]
        # Apply to each position in the sequence
        noise = direction_combination.unsqueeze(0).unsqueeze(0).expand(1, sequence_length, latent_dim)
        sampled_latent = posterior_mean + noise * std * slider4.val
        generated_images = model.decode(sampled_latent)
        generated_images = torch.clamp(generated_images, 0, 1)
        vutils.save_image(generated_images, output_path, nrow=1)
        print(f"Saved interactive generation result to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a single frame using VAE")
    parser.add_argument("--checkpoint", type=str, default="vit-l-20.safetensors", help="Path to VAE safetensors checkpoint file (default: vit-l-20.safetensors)")
    parser.add_argument("--output", type=str, default="generated_frame.png", help="Output image path")
    parser.add_argument("--prompt", type=str, default="sample_data/sample_image_0.png", help="Path to prompt image for guided generation (default: sample_data/sample_image_0.png)")
    parser.add_argument("--noise_scale", type=float, default=0.1, help="Scale of noise to add to latent (default: 0.1)")
    parser.add_argument("--random", action="store_true", help="Generate from random latent (ignores prompt)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive visualization with sliders")
    
    args = parser.parse_args()
    
    # Set the device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize the large VAE model
    print("Loading VAE model: vit-l-20-shallow-encoder")
    model = VAE_models["vit-l-20-shallow-encoder"]()
    
    # Load the safetensors checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please provide a valid path to the safetensors checkpoint file.")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    # Load to CPU first since safetensors doesn't support MPS directly
    model_state_dict = load_file(checkpoint_path, device="cpu")
    model.load_state_dict(model_state_dict)
    print("Model weights loaded successfully")
    
    # Set the device and put the model in evaluation mode
    model.to(device)
    model.eval()
    
    # Generate the frame
    if args.interactive:
        if not Path(args.prompt).exists():
            print(f"Error: Prompt image not found at {args.prompt}")
            return
        print("Launching interactive visualization...")
        interactive_generation(model, device, args.prompt, args.output)
    elif args.random:
        print("Generating frame from random latent...")
        generate_frame_random(model, device, args.output)
    else:
        if not Path(args.prompt).exists():
            print(f"Error: Prompt image not found at {args.prompt}")
            return
        print(f"Generating frame from prompt image with noise scale {args.noise_scale}...")
        generate_frame_from_prompt(model, device, args.prompt, args.noise_scale, args.output)


if __name__ == "__main__":
    main() 