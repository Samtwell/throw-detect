import os
import random
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent.resolve()
DATASET = ROOT / "pose_dataset"
RAW_FRAMES = DATASET / "raw_frames"
IMAGES_TRAIN = DATASET / "images" / "train"
IMAGES_VAL = DATASET / "images" / "val"
LABELS_TRAIN = DATASET / "labels" / "train"
LABELS_VAL = DATASET / "labels" / "val"

TRAIN_N = 500
VAL_N = 100
IMGZ = 640
EPOCHS = 20

for d in (IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL):
    d.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        if f.is_file():
            f.unlink()

all_frames = sorted([p for p in RAW_FRAMES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
print(f"[INFO] {len(all_frames)} raw frames available")

sample_n = min(TRAIN_N + VAL_N, len(all_frames))
sampled = random.Random(42).sample(all_frames, sample_n)
random.Random(7).shuffle(sampled)

train_imgs = sampled[:TRAIN_N] if len(sampled) >= TRAIN_N else sampled[:max(1, int(len(sampled) * 0.8))]
val_imgs = sampled[len(train_imgs):]

print(f"[INFO] Sampling {len(train_imgs)} train, {len(val_imgs)} val")

det_model = YOLO(str(ROOT / "yolo26n.pt"))
try:
    pose_model = YOLO("yolo11n-pose.pt")
except Exception:
    pose_model = YOLO("yolov8n-pose.pt")

def yolo_pose_label(img_path, dest_label):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    results = pose_model.predict(img, verbose=False, conf=0.25)
    if not results or not results[0].keypoints:
        return False
    kpts = results[0].keypoints
    boxes = results[0].boxes
    lines = []
    nk = len(kpts) if kpts else 0
    for i in range(len(boxes) if boxes else 0):
        try:
            xyxy = boxes.xyxy[i].cpu().numpy()
        except Exception:
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        cls = 0
        kpt_strs = []
        for kidx in range(17):
            if kpts is not None and i < len(kpts) and kidx < kpts.shape[1]:
                try:
                    kdata = kpts.xy[i][kidx].cpu().numpy()
                    conf_k = float(kpts.conf[i][kidx].cpu().numpy()) if hasattr(kpts, 'conf') else 1.0
                    kx = float(kdata[0]) / w
                    ky = float(kdata[1]) / h
                    vis = 2 if conf_k > 0.5 else 1
                    kpt_strs.append(f"{kx:.6f} {ky:.6f} {vis}")
                except Exception:
                    kpt_strs.append("0 0 0")
            else:
                kpt_strs.append("0 0 0")
        line = f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(kpt_strs)
        lines.append(line)
    if not lines:
        return False
    dest_label.write_text("\n".join(lines) + "\n")
    return True

def process_split(img_paths, img_dir, lbl_dir, name):
    ok = 0
    for p in img_paths:
        dest_img = img_dir / p.name
        dest_lbl = (lbl_dir / p.stem).with_suffix(".txt")
        shutil.copy2(p, dest_img)
        if yolo_pose_label(p, dest_lbl):
            ok += 1
        else:
            dest_lbl.write_text("")
    print(f"[INFO] {name}: {ok}/{len(img_paths)} images annotated")

process_split(train_imgs, IMAGES_TRAIN, LABELS_TRAIN, "train")
process_split(val_imgs, IMAGES_VAL, LABELS_VAL, "val")

print("[INFO] Starting training...")
train_model = YOLO("yolo11n-pose.pt")
results = train_model.train(
    data=str(DATASET / "data.yaml"),
    epochs=EPOCHS,
    imgsz=IMGZ,
    batch=8,
    device="cpu",
    workers=0,
    patience=5,
    project=str(ROOT / "runs" / "pose"),
    name="train",
    exist_ok=True,
)
print("[DONE] Training complete.")
