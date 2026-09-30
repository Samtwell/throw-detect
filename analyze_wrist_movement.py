"""Extract arm keypoint motion data from a video using a YOLO Pose model.

This script is intentionally future-ready. It does not classify "throwing".
Instead, it runs pose inference frame by frame, records shoulder/elbow/wrist
coordinates, computes wrist displacement between consecutive frames, and writes
the results to CSV for later motion-analysis work.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


# Standard COCO human-pose indices used by Ultralytics pose models.
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLO Pose on a video and export wrist movement data to CSV."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolo26n-pose.pt"),
        help="Path to a trained YOLO Pose model.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Input video to analyze.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("pose_dataset/wrist_movement.csv"),
        help="CSV path for exported movement data.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for pose predictions.",
    )
    return parser.parse_args()


def point_to_xy(point_row: Any) -> tuple[float | None, float | None]:
    """Convert one keypoint row into x/y floats."""
    return float(point_row[0]), float(point_row[1])


def calculate_displacement(
    previous_point: tuple[float | None, float | None] | None,
    current_point: tuple[float | None, float | None] | None,
) -> float | None:
    """Return Euclidean distance between two points when both are available."""
    if previous_point is None or current_point is None:
        return None

    if None in previous_point or None in current_point:
        return None

    return math.hypot(current_point[0] - previous_point[0], current_point[1] - previous_point[1])


def get_primary_person_keypoints(result: Any) -> Any | None:
    """Return keypoints for the highest-confidence detected person.

    This keeps the first version simple and stable. If you later want robust
    multi-person motion analysis, replace this with pose tracking logic.
    """
    if result.boxes is None or result.keypoints is None:
        return None
    if len(result.boxes) == 0 or len(result.keypoints) == 0:
        return None

    confidences = result.boxes.conf.tolist()
    best_index = max(range(len(confidences)), key=confidences.__getitem__)
    return result.keypoints.xy[best_index]


def main() -> None:
    """Run frame-by-frame pose inference and export movement data."""
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.video.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    capture = cv2.VideoCapture(str(args.video))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    previous_left_wrist: tuple[float | None, float | None] | None = None
    previous_right_wrist: tuple[float | None, float | None] | None = None
    frame_number = 0

    fieldnames = [
        "frame_number",
        "left_shoulder_x",
        "left_shoulder_y",
        "right_shoulder_x",
        "right_shoulder_y",
        "left_elbow_x",
        "left_elbow_y",
        "right_elbow_x",
        "right_elbow_y",
        "left_wrist_x",
        "left_wrist_y",
        "right_wrist_x",
        "right_wrist_y",
        "left_wrist_displacement",
        "right_wrist_displacement",
    ]

    with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            success, frame = capture.read()
            if not success:
                break

            result = model.predict(frame, conf=args.conf, verbose=False)[0]
            keypoints = get_primary_person_keypoints(result)

            if keypoints is None:
                writer.writerow({"frame_number": frame_number})
                frame_number += 1
                previous_left_wrist = None
                previous_right_wrist = None
                continue

            left_shoulder = point_to_xy(keypoints[LEFT_SHOULDER])
            right_shoulder = point_to_xy(keypoints[RIGHT_SHOULDER])
            left_elbow = point_to_xy(keypoints[LEFT_ELBOW])
            right_elbow = point_to_xy(keypoints[RIGHT_ELBOW])
            left_wrist = point_to_xy(keypoints[LEFT_WRIST])
            right_wrist = point_to_xy(keypoints[RIGHT_WRIST])

            row = {
                "frame_number": frame_number,
                "left_shoulder_x": left_shoulder[0],
                "left_shoulder_y": left_shoulder[1],
                "right_shoulder_x": right_shoulder[0],
                "right_shoulder_y": right_shoulder[1],
                "left_elbow_x": left_elbow[0],
                "left_elbow_y": left_elbow[1],
                "right_elbow_x": right_elbow[0],
                "right_elbow_y": right_elbow[1],
                "left_wrist_x": left_wrist[0],
                "left_wrist_y": left_wrist[1],
                "right_wrist_x": right_wrist[0],
                "right_wrist_y": right_wrist[1],
                "left_wrist_displacement": calculate_displacement(previous_left_wrist, left_wrist),
                "right_wrist_displacement": calculate_displacement(previous_right_wrist, right_wrist),
            }
            writer.writerow(row)

            previous_left_wrist = left_wrist
            previous_right_wrist = right_wrist
            frame_number += 1

    capture.release()
    print(f"Saved movement data to {args.output_csv}")


if __name__ == "__main__":
    main()
