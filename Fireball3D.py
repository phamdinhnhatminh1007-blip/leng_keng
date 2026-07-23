"""
Fireball Hand Magic — xoe tay de tha qua cau lua 3D (Ban Lien Tuc)
====================================================
Ghep 3 mang:
  1. Webcam  -> lam background 3D
  2. MediaPipe -> lay toa do ban tay
  3. Ursina  -> spawn fireball 3D (glow + particle) khi xoe tay

Cai dat:
    pip install ursina mediapipe opencv-python pillow numpy

Chay:
    python fireball_3d.py

Dieu khien:
    - Xoe ca 5 ngon tay  -> tha qua cau lua bay theo huong tay (ban lien tuc)
    - Nam dam / thu tay   -> khong spawn
    - Phim ESC / dong cua so -> thoat
"""

import math
import random
import time
import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from ursina import (
    Ursina, Entity, camera, window, color, destroy, time as utime, Vec3,
    application,
)

# ----------------------------------------------------------------------
# 1. WEBCAM + MEDIAPIPE
# ----------------------------------------------------------------------
CAM_W, CAM_H = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

# --- Tasks API (giong cv.py). Can file hand_landmarker.task cung thu muc ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Khong tim thay model: {MODEL_PATH}\n"
        "Copy file hand_landmarker.task vao cung thu muc voi file nay."
    )

_options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)
detector = mp_vision.HandLandmarker.create_from_options(_options)
_t0 = time.time()


def fingers_up(lm):
    """lm = list 21 landmark. Tra ve [thumb, index, middle, ring, pinky]."""
    f = [1 if lm[4].x < lm[3].x else 0]          # ngon cai (gia dinh tay phai, da flip)
    for tip in (8, 12, 16, 20):
        f.append(1 if lm[tip].y < lm[tip - 2].y else 0)
    return f


# ----------------------------------------------------------------------
# 2. URSINA SCENE
# ----------------------------------------------------------------------
app = Ursina()
window.title = "Fireball Hand Magic - Rapid Fire"
window.color = color.black

# --- Background quad: dan webcam len, parent vao camera de luon lap day khung hinh
aspect = CAM_W / CAM_H
BG_Z = 50                                   # khoang cach nen toi camera
fov_rad = math.radians(camera.fov)
bg_h = 2 * BG_Z * math.tan(fov_rad / 2)
bg_w = bg_h * aspect
background = Entity(
    parent=camera,
    model="quad",
    scale=(bg_w, bg_h),
    z=BG_Z,
    double_sided=True,
)

# Mat phang spawn fireball (gan camera hon nen -> fireball noi len truoc canh)
SPAWN_Z = 12
spawn_h = 2 * SPAWN_Z * math.tan(fov_rad / 2)
spawn_w = spawn_h * aspect


def hand_to_world(hx, hy):
    """Toa do tay chuan hoa (0..1) -> toa do the gioi tren mat phang spawn."""
    x = (hx - 0.5) * spawn_w
    y = (0.5 - hy) * spawn_h         # lat truc y
    return Vec3(x, y, SPAWN_Z)


# ----------------------------------------------------------------------
# 3. FIREBALL + PARTICLE
# ----------------------------------------------------------------------
class Ember(Entity):
    """Tan lua nho bay ra tu qua cau."""

    def __init__(self, position):
        super().__init__(
            model="sphere",
            color=random.choice([color.orange, color.yellow, color.red]),
            scale=random.uniform(0.15, 0.4),
            position=position,
            unlit=True,
        )
        self.velocity = Vec3(
            random.uniform(-3, 3),
            random.uniform(-1, 4),
            random.uniform(-3, 3),
        )
        self.life = random.uniform(0.3, 0.7)

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.scale *= 0.90
        self.alpha = max(0.0, self.life * 1.5)
        if self.life <= 0:
            destroy(self)


class Fireball(Entity):
    """Qua cau lua: loi sang + 2 lop hao quang + tan lua."""

    def __init__(self, position, velocity):
        super().__init__(position=position)
        self.velocity = velocity
        self.life = 2.2
        self.timer = 0.0

        # 3 lop long nhau tao chieu sau va do rung
        self.core = Entity(parent=self, model="sphere", color=color.yellow,
                           scale=0.6, unlit=True)
        self.mid = Entity(parent=self, model="sphere", color=color.orange,
                          scale=1.0, unlit=True)
        self.outer = Entity(parent=self, model="sphere", color=color.red,
                            scale=1.6, unlit=True)
        self.mid.alpha = 0.55
        self.outer.alpha = 0.30

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.timer += utime.dt

        # Rung / flicker cho co cam giac lua chay
        flick = 1 + 0.12 * math.sin(self.timer * 40)
        self.core.scale = 0.6 * flick
        self.mid.scale = 1.0 * flick
        self.outer.scale = 1.6 * flick

        self.scale *= 1.012                     # lon dan -> cam giac bay ve phia nguoi xem

        # Nha tan lua
        if random.random() < 0.7:
            Ember(self.world_position + Vec3(
                random.uniform(-0.3, 0.3),
                random.uniform(-0.3, 0.3),
                0))

        # Mo dan roi bien mat
        if self.life < 0.6:
            a = self.life / 0.6
            self.core.alpha = a
            self.mid.alpha = 0.55 * a
            self.outer.alpha = 0.30 * a
        if self.life <= 0:
            destroy(self)


# ----------------------------------------------------------------------
# 4. VONG LAP CHINH (Ursina goi update() moi frame)
# ----------------------------------------------------------------------

# Khai bao bien de quan ly thoi gian ban (tranh viec ban qua nhieu qua cung luc)
_last_fire_time = 0.0
# Thoi gian hoi chieu (cooldown) giua 2 qua cau lua (tinh bang giay)
# Giam so nay de ban nhanh hon, tang so nay de ban cham lai
FIRE_COOLDOWN = 0.15 

def update():
    global _last_fire_time

    ok, frame = cap.read()
    if not ok:
        return

    frame = cv2.flip(frame, 1)                 # lat guong cho tu nhien
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- cap nhat background bang frame webcam
    background.texture = None
    # background.model.uvs = background.model.uvs   # giu uv
    tex = _np_to_texture(rgb)
    if tex:
        background.texture = tex

    # --- nhan dien tay (Tasks API, che do VIDEO)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms = int((time.time() - _t0) * 1000)
    res = detector.detect_for_video(mp_image, ts_ms)
    
    palm_open = False
    
    if res.hand_landmarks:
        lm = res.hand_landmarks[0]
        f = fingers_up(lm)
        palm_open = sum(f) == 5                 # xoe ca 5 ngon

        # KIEM TRA DIEU KIEN BAN LIEN TUC: 
        # Chi can tay dang xoe va da qua thoi gian cooldown
        if palm_open and (time.time() - _last_fire_time > FIRE_COOLDOWN):
            # Vi tri spawn = long ban tay (landmark 9)
            pos = hand_to_world(lm[9].x, lm[9].y)
            
            # Them mot chut sai so (jitter) de cac qua cau khong bay cung 1 diem
            jitter_x = random.uniform(-0.5, 0.5)
            jitter_y = random.uniform(-0.5, 0.5)
            pos += Vec3(jitter_x, jitter_y, 0)

            # Huong = tu co tay (0) toi ngon giua (12)
            dx = lm[12].x - lm[0].x
            dy = lm[0].y - lm[12].y             # lat y
            n = math.hypot(dx, dy) or 1
            vel = Vec3(dx / n, dy / n, 0) * 14  # toc do bay
            vel.z = -3                          # huong nhe ve phia camera
            
            # Tao qua cau lua
            Fireball(pos, vel)
            
            # Cap nhat lai thoi gian ban gan nhat
            _last_fire_time = time.time()


# --- chuyen numpy array -> Ursina Texture ---
def _np_to_texture(rgb):
    try:
        from ursina import Texture
        img = Image.fromarray(rgb)
        return Texture(img)
    except Exception as e:
        print("Texture error:", e)
        return None


def input(key):
    if key == "escape":
        application.quit()


if __name__ == "__main__":
    print("Xoe ca 5 ngon tay de ban cau lua lien tuc. ESC de thoat.")
    try:
        app.run()
    finally:
        # Don dep thu cong roi thoat cung "cung" bang os._exit de Python
        # KHONG chay destructor cua MediaPipe (tranh loi 'NoneType' luc shutdown
        # tren Python 3.13).
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
