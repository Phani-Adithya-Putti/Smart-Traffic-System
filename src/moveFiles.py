import os
import random
import shutil
from pathlib import Path


def split_videos(video_dir, output_dir, split_ratio=0.8):
    video_path = Path(video_dir)
    out_path = Path(output_dir)

    # Video extensions
    extensions = {'.avi', '.mp4', '.mov', '.mkv'}

    # Get all videos
    videos = [
        f for f in video_path.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]

    # Shuffle videos
    random.seed(42)
    random.shuffle(videos)

    # Calculate split
    split_idx = int(len(videos) * split_ratio)

    train_videos = videos[:split_idx]
    val_videos = videos[split_idx:]

    # Create folders
    train_dir = out_path / "train"
    val_dir = out_path / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Copy videos
    for video in train_videos:
        shutil.copy(video, train_dir / video.name)

    for video in val_videos:
        shutil.copy(video, val_dir / video.name)

    print("Done!")
    print(f"Total videos: {len(videos)}")
    print(f"Training videos: {len(train_videos)}")
    print(f"Validation videos: {len(val_videos)}")


if __name__ == "__main__":
    split_videos(
        video_dir="data/videos",
        output_dir="video_dataset",
        split_ratio=0.8
    )
