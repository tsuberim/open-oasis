import os
import re
import subprocess

VIDEOS_DIR = "videos"
URLS_FILE = "urls.txt"
COOKIES_FILE = "cookies.txt"

os.makedirs(VIDEOS_DIR, exist_ok=True)

with open(URLS_FILE, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

def extract_hash(url):
    # Handles standard and short YouTube URLs
    match = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
    return match.group(1) if match else None

for url in urls:
    vid_hash = extract_hash(url)
    if not vid_hash:
        print(f"Could not extract hash from: {url}")
        continue
    out_path = os.path.join(VIDEOS_DIR, f"{vid_hash}.mp4")
    if os.path.exists(out_path):
        print(f"Already downloaded: {out_path}")
        continue
    print(f"Downloading {url} to {out_path} (video only, 480p mp4)...")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=480][vcodec^=avc1]",
        "--no-audio",
        "--recode-video", "mp4",
        "-o", out_path,
    ]
    
    # Add cookies if cookies.txt exists
    if os.path.exists(COOKIES_FILE):
        cmd.extend(["--cookies", COOKIES_FILE])
        print(f"Using cookies from {COOKIES_FILE}")
    
    cmd.append(url)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to download: {url}") 