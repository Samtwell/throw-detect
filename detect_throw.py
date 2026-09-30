import os, math, sys
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).parent.resolve()
MODEL = ROOT / 'runs' / 'pose' / 'train' / 'weights' / 'best.pt'
if len(sys.argv) > 1:
    VIDEO = Path(sys.argv[1])
else:
    VIDEO = ROOT / 'pose_dataset' / 'videos' / 'vive_day1' / '100_Vive_1' / '0.mp4'

LS, RS, LE, RE, LW, RW = 5, 6, 7, 8, 9, 10
VEL_WIN = 3
MIN_VEL = 60
MIN_REL_VEL = 1.8
COOLDOWN = 10

def pxy(k, i):
    try:
        a = k.xy[0][i].cpu().numpy()
        return float(a[0]), float(a[1])
    except Exception:
        return None

def dist(a, b):
    if a is None or b is None: return 0.0
    return math.hypot(a[0]-b[0], a[1]-b[1])

model = YOLO(str(MODEL))
cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    print(f'ERROR: cannot open {VIDEO}'); sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f'Video: {VIDEO.name}  FPS={fps:.1f}')

prev_lw, prev_rw = None, None
vel_hist_l, vel_hist_r = [], []
cooldown = 0
frame_i = 0
throw_count = 0

while True:
    ok, frame = cap.read()
    if not ok: break
    frame_i += 1
    r = model.predict(frame, verbose=False, conf=0.3, device='cpu')[0]
    k = r.keypoints
    if k is None or len(k) == 0:
        prev_lw = prev_rw = None
        vel_hist_l.clear(); vel_hist_r.clear()
        continue

    lw, rw = pxy(k, LW), pxy(k, RW)
    ls, rs = pxy(k, LS), pxy(k, RS)
    le, re = pxy(k, LE), pxy(k, RE)
    shoulder_w = max(dist(ls, rs), 1.0)
    frame_dt = 1.0 / fps

    vl = dist(prev_lw, lw) / frame_dt if prev_lw else 0.0
    vr = dist(prev_rw, rw) / frame_dt if prev_rw else 0.0

    vel_hist_l.append(vl); vel_hist_r.append(vr)
    if len(vel_hist_l) > VEL_WIN: vel_hist_l.pop(0)
    if len(vel_hist_r) > VEL_WIN: vel_hist_r.pop(0)
    va_l = sum(vel_hist_l)/len(vel_hist_l) if vel_hist_l else 0.0
    va_r = sum(vel_hist_r)/len(vel_hist_r) if vel_hist_r else 0.0

    elbow_wrist_l = dist(le, lw) / max(shoulder_w, 1)
    elbow_wrist_r = dist(re, rw) / max(shoulder_w, 1)
    ext_l = elbow_wrist_l > 0.85
    ext_r = elbow_wrist_r > 0.85

    trigger = False
    side = None
    abs_thresh_px = MIN_VEL * shoulder_w / 40.0
    if va_l > abs_thresh_px and ext_l:
        baseline = max(va_r, 1.0)
        if va_l / baseline > MIN_REL_VEL or va_r == 0:
            trigger, side = True, 'LEFT'
    if va_r > abs_thresh_px and ext_r:
        baseline = max(va_l, 1.0)
        if va_r / baseline > MIN_REL_VEL or va_l == 0:
            trigger, side = True, 'RIGHT'
    if va_l > abs_thresh_px * 1.6 and va_r > abs_thresh_px * 1.6:
        trigger = True
        side = 'BOTH' if abs(va_l-va_r)/max(va_l,va_r,1) < 0.4 else ('LEFT' if va_l>va_r else 'RIGHT')

    if trigger and cooldown <= 0:
        throw_count += 1
        sec = frame_i / fps
        tag = '#'
        print(f'\n{tag*50}')
        print(f'  ⚡ THROW DETECTED [{side}]  frame={frame_i}  time={sec:.2f}s')
        print(f'     wrist_vel L={va_l:.0f} R={va_r:.0f} px/s  extension L={elbow_wrist_l:.2f} R={elbow_wrist_r:.2f}')
        print(f'{tag*50}\n')
        cooldown = COOLDOWN
    elif trigger:
        pass

    if cooldown > 0: cooldown -= 1
    prev_lw, prev_rw = lw, rw

cap.release()
if throw_count == 0:
    print(f'\n{"#"*50}')
    print(f'  ✅  THROW NOT DETECTED')
    print(f'     scanned {frame_i} frames, no throwing motion found')
    print(f'{"#"*50}')
else:
    print(f'\nDone. Scanned {frame_i} frames. {throw_count} throw event(s) detected.')
