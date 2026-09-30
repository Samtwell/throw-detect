"""Extract frames from videos for YOLO Pose dataset creation.

This script reads every supported video in pose_dataset/videos and writes
sampled frames into pose_dataset/raw_frames. Frame filenames keep the source
video name so later dataset splitting can happen by video instead of by image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract every Nth frame from videos for YOLO Pose training."
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("pose_dataset/videos"),
        help="Directory containing source videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pose_dataset/raw_frames"),
        help="Directory where extracted frames will be saved.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=5,
        help="Save one frame for every N frames read.",
    )
    parser.add_argument(
        "--image-format",
        default="jpg",
        choices=["jpg", "png"],
        help="Image format for extracted frames.",
    )
    return parser.parse_args()


def get_video_files(videos_dir: Path) -> list[Path]:
    """Return supported video files recursively from the input directory tree."""
    if not videos_dir.exists():
        print(f"Video directory does not exist: {videos_dir}")
        return []

    return sorted(
        path for path in videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    frame_interval: int,
    image_format: str,
    videos_dir: Path | None = None,
) -> int:
    """Extract every Nth frame from one video.

    Returns the number of frames written. Corrupted videos are skipped safely.
    """
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"[SKIP] Could not open video: {video_path.name}")
        return 0

    source_frame_index = 0
    saved_frame_count = 0

    if videos_dir is not None:
        try:
            rel = video_path.relative_to(videos_dir)
            video_stem = str(rel.with_suffix("")).replace("\\", "_").replace("/", "_")
        except ValueError:
            video_stem = video_path.stem
    else:
        video_stem = video_path.stem

    while True:
        success, frame = capture.read()
        if not success:
            break

        if source_frame_index % frame_interval == 0:
            saved_frame_count += 1
            output_name = f"{video_stem}_frame_{saved_frame_count:06d}.{image_format}"
            output_path = output_dir / output_name
            write_ok = cv2.imwrite(str(output_path), frame)

            if not write_ok:
                print(f"[WARN] Failed to save frame: {output_path.name}")
                saved_frame_count -= 1

        source_frame_index += 1

    capture.release()
    return saved_frame_count


def main() -> None:
    """Run the extraction pipeline for all videos."""
    args = parse_args()

    if args.frame_interval <= 0:
        raise ValueError("--frame-interval must be greater than 0.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_files = get_video_files(args.videos_dir)

    if not video_files:
        print("No supported videos found in pose_dataset/videos.")
        return

    total_saved = 0

    for video_path in video_files:
        try:
            extracted_count = extract_frames_from_video(
                video_path=video_path,
                output_dir=args.output_dir,
                frame_interval=args.frame_interval,
                image_format=args.image_format,
                videos_dir=args.videos_dir,
            )
            total_saved += extracted_count
            print(f"[OK] {video_path.name}: extracted {extracted_count} frames")
        except Exception as exc:  # noqa: BLE001 - continue batch processing safely
            print(f"[SKIP] Error while processing {video_path.name}: {exc}")

    print(f"Finished. Extracted {total_saved} frames from {len(video_files)} video(s).")


if __name__ == "__main__":
    main()
