import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.resolve()
try:
    import shutil
    shutil.rmtree(ROOT / 'runs' / 'pose' / 'train', ignore_errors=True)
except Exception:
    pass

model = YOLO(str(ROOT / 'yolo11n-pose.pt'))
model.train(
    data=str(ROOT / 'pose_dataset' / 'data.yaml'),
    epochs=8, imgsz=416, batch=4, device='cpu', workers=0,
    patience=2, project=str(ROOT / 'runs' / 'pose'), name='train',
    exist_ok=True, verbose=True, optimizer='SGD', lr0=0.001
)
print('ALL_DONE')
