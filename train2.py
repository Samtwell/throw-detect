import os, sys, shutil
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from pathlib import Path
ROOT = Path(__file__).parent.resolve()

FONT_DIR = ROOT / 'assets' / 'fonts'
FONT_DIR.mkdir(parents=True, exist_ok=True)

from PIL import ImageFont
import matplotlib as mpl
mpl_font = Path(mpl.get_data_path()) / 'fonts' / 'ttf' / 'DejaVuSans.ttf'
if mpl_font.exists():
    target = FONT_DIR / 'Arial.ttf'
    if not target.exists():
        shutil.copy2(mpl_font, target)
    os.environ['YOLO_FONT_DIR'] = str(FONT_DIR)
    local_font_dir = FONT_DIR
else:
    local_font_dir = None

import ultralytics.utils.checks as uc
_orig_check_font = uc.check_font
def patched_check_font(font='Arial.ttf'):
    candidates = []
    if local_font_dir:
        candidates.append(local_font_dir / font)
    candidates.append(ROOT / 'assets' / 'fonts' / font)
    candidates.append(ROOT / font)
    for c in candidates:
        if c and c.exists():
            return str(c)
    try:
        return _orig_check_font(font)
    except Exception:
        for c in candidates:
            if c and c.exists():
                return str(c)
        if mpl_font.exists():
            return str(mpl_font)
        raise
uc.check_font = patched_check_font

import ultralytics.data.utils as du
if hasattr(du, 'check_font'):
    du.check_font = patched_check_font

shutil.rmtree(ROOT / 'runs' / 'pose' / 'train', ignore_errors=True)

from ultralytics import YOLO
model = YOLO(str(ROOT / 'yolo11n-pose.pt'))
try:
    model.train(
        data=str(ROOT / 'pose_dataset' / 'data.yaml'),
        epochs=8, imgsz=416, batch=4, device='cpu', workers=0,
        patience=2, project=str(ROOT / 'runs' / 'pose'), name='train',
        exist_ok=True, verbose=True, optimizer='SGD', lr0=0.001, plots=False
    )
except SystemExit:
    pass
print('ALL_DONE_OK')
