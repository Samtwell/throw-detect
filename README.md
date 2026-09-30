# YOLO Pose Dataset Pipeline for Garbage-Throwing Motion Analysis

This project prepares a **YOLO Pose** dataset for learning human keypoints from video frames.

Important:
- This pipeline is for **pose/keypoint detection**, not direct throw classification from a single image.
- The goal is to detect stable human pose keypoints first, especially:
  - left shoulder
  - right shoulder
  - left elbow
  - right elbow
  - left wrist
  - right wrist
- Later, those coordinates can be analyzed across video frames to detect throwing motion.

## Project Structure

```text
pose_dataset/
├── videos/
├── raw_frames/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

## 1. Put Your Videos Here

Copy your source throwing-action videos into:

```text
pose_dataset/videos/
```

Example:

```text
pose_dataset/videos/throw_01.mp4
pose_dataset/videos/throw_02.mp4
pose_dataset/videos/throw_03.mp4
```

## 2. Extract Frames from Videos

Run:

```bash
python extract_frames.py --frame-interval 5
```

This will:
- read every supported video from `pose_dataset/videos/`
- extract every 5th frame
- save frames into `pose_dataset/raw_frames/`
- preserve the original video name in the frame filename

Example output filenames:

```text
throw_01_frame_000001.jpg
throw_01_frame_000002.jpg
throw_02_frame_000001.jpg
```

You can change the interval:

```bash
python extract_frames.py --frame-interval 10
```

## 3. Prepare the Dataset Split

Run:

```bash
python prepare_dataset.py --clear-existing
```

This script:
- reads extracted frames from `pose_dataset/raw_frames/`
- groups frames by original video name
- splits by **video**, not by random image
- copies images into:
  - `pose_dataset/images/train/`
  - `pose_dataset/images/val/`
  - `pose_dataset/images/test/`

Default split:
- 70% train
- 20% validation
- 10% test

Why split by video?
- Because frames from the same video are very similar.
- If the same video appears in both training and validation, model evaluation becomes misleading.

## 4. Annotate Pose Keypoints

After splitting the images, annotate the images inside:

```text
pose_dataset/images/train/
pose_dataset/images/val/
pose_dataset/images/test/
```

Save the YOLO Pose label files into:

```text
pose_dataset/labels/train/
pose_dataset/labels/val/
pose_dataset/labels/test/
```

Use a labeling tool that supports **human pose / keypoint annotation** and can export in **YOLO Pose format** or a format you can convert to YOLO Pose.

Recommended keypoint standard:
- Use the **standard 17 human keypoints** expected by Ultralytics pretrained human pose models.
- This matches the common COCO human pose layout.

Standard keypoint order:

```text
0  nose
1  left_eye
2  right_eye
3  left_ear
4  right_ear
5  left_shoulder
6  right_shoulder
7  left_elbow
8  right_elbow
9  left_wrist
10 right_wrist
11 left_hip
12 right_hip
13 left_knee
14 right_knee
15 left_ankle
16 right_ankle
```

The most important arm chain for your later movement analysis is:

```text
shoulder -> elbow -> wrist
```

### Annotation Notes

- Annotate the person as class `person`
- Keep the 17-keypoint order consistent for every image
- Even though shoulder/elbow/wrist matter most, keep the full standard pose format so it matches the pretrained YOLO Pose model
- Use visibility flags correctly if your labeling tool supports them:
  - `0` = not labeled
  - `1` = labeled but not visible
  - `2` = visible

## 5. Train YOLO Pose

Use your pretrained pose model as the starting point:

```bash
yolo pose train model=yolo26n-pose.pt data=pose_dataset/data.yaml epochs=50 imgsz=640
```

If `yolo26n-pose.pt` is not in the project root, either:
- copy it into this folder, or
- replace `model=yolo26n-pose.pt` with the full path to your checkpoint

## 6. Test the Trained Pose Model

After training, test inference on a video:

```bash
yolo pose predict model=runs/pose/train/weights/best.pt source=pose_dataset/videos/throw_01.mp4
```

You can also test on a folder of images:

```bash
yolo pose predict model=runs/pose/train/weights/best.pt source=pose_dataset/images/test
```

## 7. Export Wrist Movement Data

The provided analysis script does **not** classify throwing yet.
It only extracts pose coordinates and wrist motion data frame by frame.

Run:

```bash
python analyze_wrist_movement.py --model runs/pose/train/weights/best.pt --video pose_dataset/videos/throw_01.mp4 --output-csv pose_dataset/wrist_movement_throw_01.csv
```

This script currently:
- loads a trained YOLO Pose model
- processes the video frame by frame
- reads shoulder, elbow, and wrist coordinates
- calculates wrist displacement between consecutive frames
- writes the results to CSV

This CSV can later be used to build your throw-detection logic.

## 8. Suggested Workflow

1. Put source videos into `pose_dataset/videos/`
2. Extract frames with `extract_frames.py`
3. Split frames by video with `prepare_dataset.py`
4. Annotate `person` keypoints in the split image folders
5. Train the pose model with `yolo pose train`
6. Run `analyze_wrist_movement.py` on test videos
7. Later add motion rules or a temporal classifier for throw detection

## 9. Notes

- This pipeline does **not** create normal object-detection labels.
- This is a **YOLO Pose / keypoint** project.
- The pose system is kept separate so it can later be combined with your existing trash/person detector.
