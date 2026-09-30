import sys, os, shutil
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

ROOT = Path(__file__).parent.resolve()
from ultralytics import YOLO
import cv2
import numpy as np

POSE = ROOT / 'yolo11n-pose.pt'
model = YOLO(str(POSE))

for split in ('train', 'val'):
    img_dir = ROOT / 'pose_dataset' / 'images' / split
    lbl_dir = ROOT / 'pose_dataset' / 'labels' / split
    lbl_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')])
    ok = 0
    for p in imgs:
        label_path = (lbl_dir / p.stem).with_suffix('.txt')
        try:
            r = model.predict(str(p), verbose=False, conf=0.3, device='cpu')[0]
            h, w = r.orig_shape
            lines = []
            if r.boxes is not None and r.keypoints is not None:
                nb = len(r.boxes)
                for i in range(nb):
                    try:
                        b = r.boxes.xyxy[i].cpu().numpy().astype(float)
                        x1,y1,x2,y2 = b.tolist()
                        bw = (x2-x1)/w; bh = (y2-y1)/h
                        cx = (x1+x2)/2/w; cy = (y1+y2)/2/h
                        parts = [f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}']
                        kxy = r.keypoints.xy[i].cpu().numpy().astype(float)
                        kc = r.keypoints.conf[i].cpu().numpy().astype(float) if hasattr(r.keypoints, 'conf') and r.keypoints.conf is not None else np.ones(17)
                        for k in range(17):
                            if k < len(kxy):
                                kx = kxy[k,0]/w; ky = kxy[k,1]/h
                                vis = 2 if (k < len(kc) and kc[k] > 0.5) else 1
                            else:
                                kx=0; ky=0; vis=0
                            parts.append(f'{kx:.6f} {ky:.6f} {vis}')
                        lines.append(' '.join(parts))
                    except Exception:
                        continue
            if lines:
                label_path.write_text('\n'.join(lines)+'\n')
                ok += 1
            else:
                label_path.write_text('')
        except Exception as e:
            label_path.write_text('')
    print(f'[ANNOT] {split}: {ok}/{len(imgs)}')

print('[ANNOT] Done. Starting train.')
tm = YOLO(str(POSE))
tm.train(
    data=str(ROOT / 'pose_dataset' / 'data.yaml'),
    epochs=15, imgsz=640, batch=8, device='cpu', workers=0,
    patience=3, project=str(ROOT / 'runs' / 'pose'), name='train', exist_ok=True
)
print('[TRAIN] DONE.')
