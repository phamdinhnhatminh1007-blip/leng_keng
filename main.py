"""
main.py — FILE CHINH: Hand Magic
================================
Chi lo phan "khung": webcam, nhan dien tay, ML doan chieu, dung scene 3D,
va DIEU PHOI chieu -> hieu ung. Toan bo hieu ung + logic tung chieu nam o
cac module rieng:

    Fireball3D.py  -> chieu "Xoe ban tay" (cau lua)
    lightning.py   -> chieu "Nam dam"    (set)

Muon them chieu moi: viet module hieu ung co ham cast(lm, hand_to_world),
roi them 1 dong vao bang SKILLS ben duoi. Khong can sua gi khac.

Chay: python main.py   (phim 'b' bat/tat bloom, ESC de thoat)
"""

import os
import time
import math
from collections import deque, Counter

import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from pathlib import Path
from ursina import Ursina, Entity, camera, window, color, application, Texture, Vec3, Text, destroy

from gesture_utils import GESTURES, normalize_landmarks
import Fireball3D
import lightning
import earth
import air
import ice

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAM_W, CAM_H = 1280, 720

# Ursina tim texture/am thanh trong asset_folder -> tro ve thu muc du an
# (de sounds/*.wav luon tim thay du chay tu cwd nao).
application.asset_folder = Path(BASE_DIR)


# ======================================================================
# 1. WEBCAM + MEDIAPIPE (lay 21 landmarks ban tay)
# ======================================================================
# CAP_DSHOW (DirectShow) mo cam nhanh hon nhieu tren Windows so voi MSMF mac dinh
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # bot do tre khung hinh

_hand_model = os.path.join(BASE_DIR, "hand_landmarker.task")
if not os.path.exists(_hand_model):
    raise FileNotFoundError(f"Khong tim thay model: {_hand_model}")

detector = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_hand_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,          # nhan dien ca 2 tay
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
)
_t0 = time.time()


# ======================================================================
# 2. ML MODEL — doan chieu tu landmarks (co lam muot qua nhieu frame)
# ======================================================================
clf = joblib.load(os.path.join(BASE_DIR, "gesture_model.joblib"))
CONF_THRESHOLD = 0.8       # duoi nguong nay coi nhu khong ro chieu
SMOOTH_FRAMES = 7          # so frame de binh chon (chong nhap nhay)
_histories = {}            # lich su vote RIENG cho tung tay: {hand_id: deque}


def current_gesture(lm, hand_id):
    """Tra ve index chieu on dinh cho 1 tay, hoac None neu chua ro."""
    vec = normalize_landmarks(lm)
    # thu ca ban goc va ban lat guong, chon ket qua tin cay hon
    mirrored = vec.reshape(21, 2).copy()
    mirrored[:, 0] *= -1
    variants = np.stack((vec, mirrored.reshape(-1)))
    probs = clf.predict_proba(variants)
    vi = int(probs.max(axis=1).argmax())
    ci = int(probs[vi].argmax())
    label = int(clf.classes_[ci])
    conf = float(probs[vi, ci])

    hist = _histories.setdefault(hand_id, deque(maxlen=SMOOTH_FRAMES))
    hist.append(label if conf >= CONF_THRESHOLD else -1)
    vote, n = Counter(hist).most_common(1)[0]
    return vote if (vote != -1 and n >= SMOOTH_FRAMES // 2 + 1) else None


# ======================================================================
# 3. SCENE 3D (Ursina) — webcam lam nen, bat glow bloom
# ======================================================================
app = Ursina()
window.title = "Hand Magic"
window.color = color.black

camera.shader = Fireball3D.bloom_shader
camera.set_shader_input("threshold", 0.90)
camera.set_shader_input("intensity", 1.7)
camera.set_shader_input("blur_size", 0.013)
BG_BRIGHTNESS = 0.85       # giam sang nen de nen khong bi bloom (phai < threshold)

Fireball3D.init_effects()  # tao texture hieu ung (sau khi Ursina() khoi dong)

# Nen 3D dan webcam
aspect = CAM_W / CAM_H
_fov = math.radians(camera.fov)
_bg_h = 2 * 50 * math.tan(_fov / 2)
background = Entity(parent=camera, model="quad",
                    scale=(_bg_h * aspect, _bg_h), z=50, double_sided=True)

# Mat phang xuat hien hieu ung
SPAWN_Z = 16
_spawn_h = 2 * SPAWN_Z * math.tan(_fov / 2)
_spawn_w = _spawn_h * aspect


def hand_to_world(hx, hy):
    """Toa do tay chuan hoa (0..1) -> toa do 3D tren mat phang spawn."""
    return Vec3((hx - 0.5) * _spawn_w, (0.5 - hy) * _spawn_h, SPAWN_Z)


# Chu bao dang khoi dong (tu an khi co khung hinh dau tien)
loading_text = Text("Dang khoi dong camera...", origin=(0, 0), scale=1.8,
                    color=color.white, background=True)
_started = False


# ======================================================================
# 4. BANG CHIEU: gesture index -> ham tung chieu
#    (moi module tu lo hieu ung + cooldown ben trong ham cast)
# ======================================================================
SKILLS = {
    0: Fireball3D.cast,    # Chieu 1 - Xoe ban tay  -> cau lua (Hoa)
    1: earth.cast,         # Chieu 2 - Nam dam      -> khien da (Tho)
    3: lightning.cast,     # Chieu 4 - Do ngon tro     -> set giang tu troi
    4: air.cast,           # Chieu 5 - Do ngon cai     -> loc xoay (Khi)
    5: ice.cast,           # Chieu 6 - Do tro + ut     -> cot bang chui len (Thuy)
    # them chieu moi: <index>: <module>.cast
}


# ======================================================================
# 5. VONG LAP CHINH
# ======================================================================
def update():
    global _started

    ok, frame = cap.read()
    if not ok:
        return
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Cap nhat nen (giam sang de khong bi bloom; ML van dung rgb goc)
    bg = (rgb.astype(np.uint16) * int(BG_BRIGHTNESS * 256) >> 8).astype(np.uint8)
    background.texture = _to_texture(bg)

    # Co khung hinh dau tien -> an chu "dang khoi dong"
    if not _started:
        destroy(loading_text)
        _started = True

    # Nhan dien tay
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect_for_video(mp_img, int((time.time() - _t0) * 1000))

    hands = res.hand_landmarks or []
    handed = res.handedness or []
    if not hands:
        _histories.clear()
        return

    # Xu ly TUNG TAY doc lap
    seen = set()
    for i, lm in enumerate(hands):
        # khoa phan biet tay: dung Left/Right; neu trung thi them chi so
        label = handed[i][0].category_name if i < len(handed) and handed[i] else str(i)
        key = label if label not in seen else f"{label}{i}"
        seen.add(key)

        gesture = current_gesture(lm, key)
        skill = SKILLS.get(gesture)      # moi module tu xu ly cooldown theo key
        if skill:
            skill(lm, hand_to_world, key)

    # Xoa lich su cua tay khong con thay
    for k in list(_histories):
        if k not in seen:
            _histories[k].clear()


def _to_texture(rgb):
    try:
        return Texture(Image.fromarray(rgb))
    except Exception as e:
        print("Texture error:", e)
        return None


def input(key):
    if key == "escape":
        application.quit()
    elif key == "b":
        camera.shader = None if camera.shader else Fireball3D.bloom_shader


if __name__ == "__main__":
    print("Xoe=lua | Dam=khien | Tro=set | Cai=loc | Tro+Ut=bang | ESC thoat.")
    try:
        app.run()
    finally:
        # Thoat "cung" bang os._exit de tranh loi destructor MediaPipe (Python 3.13)
        for fn in (cap.release, cv2.destroyAllWindows, detector.close):
            try:
                fn()
            except Exception:
                pass
        os._exit(0)