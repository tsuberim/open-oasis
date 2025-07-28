import h5py
import numpy as np
from pathlib import Path
import argparse

def verify_encoded_files(encoded_dir="./encoded"):
    """Verify and display information about encoded H5 files."""
    encoded_dir = Path(encoded_dir)
    
    if not encoded_dir.exists():
        print(f"Error: Directory {encoded_dir} does not exist")
        return
    
    h5_files = list(encoded_dir.glob("*.h5"))
    
    if not h5_files:
        print(f"No H5 files found in {encoded_dir}")
        return
    
    print(f"Found {len(h5_files)} encoded video files:")
    print("=" * 60)
    
    total_frames = 0
    total_size_mb = 0
    
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            # Get dataset info
            latents = f['latents']
            metadata = f['metadata']
            
            # Calculate file size
            file_size_mb = h5_file.stat().st_size / (1024 * 1024)
            total_size_mb += file_size_mb
            
            # Get metadata
            video_path = metadata.attrs.get('video_path', 'Unknown')
            original_frames = metadata.attrs.get('original_frames', 0)
            target_frames = metadata.attrs.get('target_frames', 0)
            actual_frames = metadata.attrs.get('actual_frames', 0)
            original_fps = metadata.attrs.get('original_fps', 0)
            target_fps = metadata.attrs.get('target_fps', 0)
            duration = metadata.attrs.get('duration', 0)
            latent_shape = metadata.attrs.get('latent_shape', (0,))
            
            total_frames += actual_frames
            
            print(f"File: {h5_file.name}")
            print(f"  Video: {Path(video_path).name}")
            print(f"  Original: {original_frames:,} frames @ {original_fps:.1f} fps")
            print(f"  Target: {target_frames:,} frames @ {target_fps} fps")
            print(f"  Actual: {actual_frames:,} frames")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Latent shape: {latent_shape}")
            print(f"  File size: {file_size_mb:.1f} MB")
            print(f"  Latents shape: {latents.shape}")
            print(f"  Latents dtype: {latents.dtype}")
            print(f"  Compression: {latents.compression}")
            print()
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Total videos: {len(h5_files)}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Total size: {total_size_mb:.1f} MB")
    print(f"  Average size per video: {total_size_mb/len(h5_files):.1f} MB")
    print(f"  Average frames per video: {total_frames/len(h5_files):.0f}")

def load_encoded_video(h5_path, start_frame=0, end_frame=None):
    """Load encoded video data from H5 file."""
    with h5py.File(h5_path, 'r') as f:
        latents = f['latents']
        metadata = f['metadata']
        
        # Get frame range
        if end_frame is None:
            end_frame = latents.shape[0]
        
        # Load latents
        video_latents = latents[start_frame:end_frame]
        
        # Get metadata
        video_info = {
            'video_path': metadata.attrs.get('video_path', 'Unknown'),
            'original_frames': metadata.attrs.get('original_frames', 0),
            'target_frames': metadata.attrs.get('target_frames', 0),
            'actual_frames': metadata.attrs.get('actual_frames', 0),
            'original_fps': metadata.attrs.get('original_fps', 0),
            'target_fps': metadata.attrs.get('target_fps', 0),
            'duration': metadata.attrs.get('duration', 0),
            'latent_shape': metadata.attrs.get('latent_shape', (0,)),
            'loaded_frames': end_frame - start_frame
        }
        
        return video_latents, video_info

def main():
    parser = argparse.ArgumentParser(description="Verify encoded H5 files")
    parser.add_argument("--encoded-dir", "-d", default="./encoded", help="Directory with encoded H5 files")
    parser.add_argument("--load-sample", "-l", help="Load and display sample from specific H5 file")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame for sample loading")
    parser.add_argument("--end-frame", type=int, default=10, help="End frame for sample loading")
    
    args = parser.parse_args()
    
    if args.load_sample:
        # Load sample from specific file
        h5_path = Path(args.encoded_dir) / args.load_sample
        if not h5_path.exists():
            print(f"Error: File {h5_path} does not exist")
            return
        
        print(f"Loading sample from {h5_path}")
        latents, video_info = load_encoded_video(h5_path, args.start_frame, args.end_frame)
        
        print(f"Video: {Path(video_info['video_path']).name}")
        print(f"Original: {video_info['original_frames']} frames @ {video_info['original_fps']:.1f} fps")
        print(f"Target: {video_info['target_frames']} frames @ {video_info['target_fps']} fps")
        print(f"Actual: {video_info['actual_frames']} frames")
        print(f"Loaded frames: {args.start_frame} to {args.end_frame} ({video_info['loaded_frames']} frames)")
        print(f"Latents shape: {latents.shape}")
        print(f"Latents dtype: {latents.dtype}")
        print(f"Latents stats:")
        print(f"  Mean: {np.mean(latents):.6f}")
        print(f"  Std: {np.std(latents):.6f}")
        print(f"  Min: {np.min(latents):.6f}")
        print(f"  Max: {np.max(latents):.6f}")
        
        # Show first few latent vectors
        print(f"\nFirst 3 latent vectors (first 10 dimensions):")
        for i in range(min(3, latents.shape[0])):
            print(f"  Frame {args.start_frame + i}: {latents[i, :10]}")
    
    else:
        # Verify all files
        verify_encoded_files(args.encoded_dir)

if __name__ == "__main__":
    main() 