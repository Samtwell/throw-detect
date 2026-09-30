"""Prepare a YOLO Pose dataset split by video.

This script copies extracted frames from pose_dataset/raw_frames into
train/val/test folders while keeping all frames from the same source video
inside the same split. That prevents temporal leakage between splits.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split extracted frames into train/val/test by source video."
    )
    parser.add_argument(
        "--raw-frames-dir",
        type=Path,
        default=Path("pose_dataset/raw_frames"),
        help="Directory containing extracted frame images.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("pose_dataset"),
        help="Root pose dataset directory.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of videos assigned to training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of videos assigned to validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of videos assigned to testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible video splitting.",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing train/val/test images before copying new files.",
    )
    return parser.parse_args()


def get_frame_files(raw_frames_dir: Path) -> list[Path]:
    """Return all supported extracted frame images."""
    if not raw_frames_dir.exists():
        print(f"Raw frame directory does not exist: {raw_frames_dir}")
        return []

    return sorted(
        path for path in raw_frames_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_video_key_from_frame(frame_path: Path) -> str | None:
    """Recover the original video stem from a frame filename.

    Expected format:
    video_name_frame_000001.jpg
    """
    if "_frame_" not in frame_path.stem:
        return None

    return frame_path.stem.rsplit("_frame_", 1)[0]


def group_frames_by_video(frame_files: list[Path]) -> dict[str, list[Path]]:
    """Group frame image paths by their source video name."""
    grouped_frames: dict[str, list[Path]] = defaultdict(list)

    for frame_path in frame_files:
        video_key = get_video_key_from_frame(frame_path)
        if video_key is None:
            print(f"[WARN] Skipping unexpected filename: {frame_path.name}")
            continue
        grouped_frames[video_key].append(frame_path)

    return dict(grouped_frames)


def compute_split_counts(
    total_videos: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """Convert split ratios into video counts while keeping totals consistent."""
    if total_videos == 0:
        return 0, 0, 0
    if total_videos == 1:
        return 1, 0, 0
    if total_videos == 2:
        return 1, 1, 0

    train_count = int(total_videos * train_ratio)
    val_count = int(total_videos * val_ratio)
    test_count = total_videos - train_count - val_count

    if total_videos >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        test_count = total_videos - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if train_count >= val_count and train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1

    return train_count, val_count, test_count


def clear_split_directories(dataset_dir: Path) -> None:
    """Remove existing files from split directories."""
    for split_name in ("train", "val", "test"):
        for subdir in ("images", "labels"):
            split_dir = dataset_dir / subdir / split_name
            if not split_dir.exists():
                continue
            for path in split_dir.iterdir():
                if path.is_file():
                    path.unlink()


def copy_split_images(split_name: str, video_names: list[str], grouped_frames: dict[str, list[Path]], dataset_dir: Path) -> int:
    """Copy grouped frame images into a split folder."""
    destination_dir = dataset_dir / "images" / split_name
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for video_name in video_names:
        for frame_path in sorted(grouped_frames[video_name]):
            shutil.copy2(frame_path, destination_dir / frame_path.name)
            copied_count += 1

    return copied_count


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Ensure split ratios are valid."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios cannot be negative.")
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0.")


def main() -> None:
    """Split extracted frames into train/val/test by video."""
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    if args.clear_existing:
        clear_split_directories(args.dataset_dir)

    frame_files = get_frame_files(args.raw_frames_dir)
    grouped_frames = group_frames_by_video(frame_files)

    if not grouped_frames:
        print("No extracted frames found to split.")
        return

    video_names = sorted(grouped_frames.keys())
    random.Random(args.seed).shuffle(video_names)

    train_count, val_count, test_count = compute_split_counts(
        total_videos=len(video_names),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_videos = video_names[:train_count]
    val_videos = video_names[train_count : train_count + val_count]
    test_videos = video_names[train_count + val_count : train_count + val_count + test_count]

    split_mapping = {
        "train": train_videos,
        "val": val_videos,
        "test": test_videos,
    }

    image_counts: dict[str, int] = {}
    for split_name, split_videos in split_mapping.items():
        image_counts[split_name] = copy_split_images(
            split_name=split_name,
            video_names=split_videos,
            grouped_frames=grouped_frames,
            dataset_dir=args.dataset_dir,
        )

    print("Dataset split complete.")
    for split_name in ("train", "val", "test"):
        print(
            f"{split_name}: "
            f"{len(split_mapping[split_name])} video(s), "
            f"{image_counts[split_name]} image(s)"
        )


if __name__ == "__main__":
    main()
