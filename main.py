"""
main.py — FILE CHINH: Hand Magic (ML gesture + hieu ung 3D)
============================================================
Ghep cac manh:
  1. Webcam    -> lam background 3D
  2. MediaPipe -> lay 21 landmarks ban tay
  3. ML model  -> nhan dien chieu (gesture_model.joblib)
  4. Fireball3D (file phu) -> hieu ung cau lua / khoi / vet lua

Chay:
    python main.py

Dieu khien:
    - Chieu 1 "Xoe ban tay"  -> ban cau lua lien tuc ve phia man hinh
    - Cac chieu khac         -> da nhan dien, chua gan hieu ung
    - Phim 'b'               -> bat/tat bloom
    - Phim ESC               -> thoat

File demo nhan dien thuan (khong hieu ung): gesture_demo.py
"""

import math
import random
import time
import os
from collections import deque, Counter

import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from ursina import (
    Ursina, Entity, camera, window, color, time as utime, Vec3,
    application, Texture,
)

from gesture_utils import GESTURES, normalize_landmarks
from Fireball3D import bloom_shader, init_effects, Fireball

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# 1. WEBCAM + MEDIAPIPE
# ----------------------------------------------------------------------
CAM_W, CAM_H = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")

_options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)
detector = mp_vision.HandLandmarker.create_from_options(_options)
_t0 = time.time()

# ----------------------------------------------------------------------
# 2. ML GESTURE MODEL (pipeline giong gesture_demo.py)
# ----------------------------------------------------------------------
clf = joblib.load(os.path.join(BASE_DIR, "gesture_model.joblib"))

CONF_THRESHOLD = 0.8       # duoi nguong nay -> khong ro chieu
SMOOTH_FRAMES = 7          # vote trong N frame gan nhat
_history = deque(maxlen=SMOOTH_FRAMES)

FIREBALL_GESTURE = 0       # "Chieu 1 - Xoe ban tay"


def flip_landmark_vector(vec):
    """Lat ngang cac moc ban tay quanh co tay (moc 0)."""
    flipped = vec.reshape(21, 2).copy()
    flipped[:, 0] *= -1
    return flipped.reshape(-1)


def predict_with_mirror(clf, vec):
    """Du doan ca dang goc va dang lat, chon ket qua tin cay hon."""
    variants = np.stack((vec, flip_landmark_vector(vec)))
    probabilities = clf.predict_proba(variants)
    best_each = probabilities.max(axis=1)
    variant_index = int(best_each.argmax())
    class_index = int(probabilities[variant_index].argmax())
    label = int(clf.classes_[class_index])
    confidence = float(probabilities[variant_index, class_index])
    return label, confidence


def current_gesture(landmarks):
    """Tra ve index chieu on dinh (da smoothing), hoac None neu chua ro."""
    vec = normalize_landmarks(landmarks)
    pred, conf = predict_with_mirror(clf, vec)
    _history.append(pred if conf >= CONF_THRESHOLD else -1)
    vote, n = Counter(_history).most_common(1)[0]
    if vote != -1 and n >= SMOOTH_FRAMES // 2 + 1:
        return vote
    return None


# ----------------------------------------------------------------------
# 3. URSINA SCENE
# ----------------------------------------------------------------------
app = Ursina()
window.title = "Hand Magic"
window.color = color.black

camera.shader = bloom_shader
# Meo HDR: webcam giam sang xuong duoi threshold -> nen khong bi bloom/nhoe,
# chi lua (sang toi da) moi glow.
camera.set_shader_input("threshold", 0.90)
camera.set_shader_input("intensity", 1.7)
camera.set_shader_input("blur_size", 0.013)
BG_BRIGHTNESS = 0.85     # do sang webcam (phai < threshold)

init_effects()           # tao texture hieu ung (sau khi Ursina() khoi dong)

# Background quad: dan webcam len
aspect = CAM_W / CAM_H
BG_Z = 50
fov_rad = math.radians(camera.fov)
bg_h = 2 * BG_Z * math.tan(fov_rad / 2)
bg_w = bg_h * aspect
background = Entity(parent=camera, model="quad", scale=(bg_w, bg_h),
                    z=BG_Z, double_sided=True)

# Mat phang spawn hieu ung
SPAWN_Z = 16
spawn_h = 2 * SPAWN_Z * math.tan(fov_rad / 2)
spawn_w = spawn_h * aspect


def hand_to_world(hx, hy):
    """Toa do tay chuan hoa (0..1) -> toa do the gioi tren mat phang spawn."""
    x = (hx - 0.5) * spawn_w
    y = (0.5 - hy) * spawn_h
    return Vec3(x, y, SPAWN_Z)


# ----------------------------------------------------------------------
# 4. VONG LAP CHINH
# ----------------------------------------------------------------------
_last_fire_time = 0.0
FIRE_COOLDOWN = 0.15     # giay giua 2 qua cau lua


def cast_fireball(lm):
    """Ban 1 qua cau lua tu long ban tay theo huong tay."""
    pos = hand_to_world(lm[9].x, lm[9].y)
    pos += Vec3(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0)
    dx = lm[12].x - lm[0].x
    dy = lm[0].y - lm[12].y
    n = math.hypot(dx, dy) or 1
    aim = Vec3(dx / n * 2.5, dy / n * 2.5, -10)
    Fireball(pos, aim, power=random.uniform(0.5, 0.8))


def update():
    global _last_fire_time

    ok, frame = cap.read()
    if not ok:
        return
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Giam sang nen de nen khong bi bloom (ML van dung rgb goc)
    bg_rgb = (rgb.astype(np.uint16) * int(BG_BRIGHTNESS * 256) >> 8).astype(np.uint8)
    background.texture = None
    tex = _np_to_texture(bg_rgb)
    if tex:
        background.texture = tex

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms = int((time.time() - _t0) * 1000)
    res = detector.detect_for_video(mp_image, ts_ms)

    if not res.hand_landmarks:
        _history.clear()
        return

    lm = res.hand_landmarks[0]
    gesture = current_gesture(lm)

    if gesture == FIREBALL_GESTURE and (time.time() - _last_fire_time > FIRE_COOLDOWN):
        cast_fireball(lm)
        _last_fire_time = time.time()
    # >>> GAN CHIEU MOI TAI DAY, vi du:
    # elif gesture == 1:
    #     cast_skill_2(lm)


def _np_to_texture(rgb):
    try:
        return Texture(Image.fromarray(rgb))
    except Exception as e:
        print("Texture error:", e)
        return None


def input(key):
    if key == "escape":
        application.quit()
    elif key == "b":
        camera.shader = None if camera.shader else bloom_shader


if __name__ == "__main__":
    print(f"Chieu ban lua: {GESTURES[FIREBALL_GESTURE]}. ESC de thoat.")
    try:
        app.run()
    finally:
        # Thoat "cung" bang os._exit de tranh loi destructor MediaPipe
        # tren Python 3.13.
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            detector.close()
        except Exception:
            pass
        os._exit(0)