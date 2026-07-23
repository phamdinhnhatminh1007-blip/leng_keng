"""
Fireball Hand Magic — xoe tay de tha qua cau lua 3D
====================================================
Ghep 3 mang:
  1. Webcam    -> lam background 3D
  2. MediaPipe -> lay toa do ban tay
  3. Ursina    -> spawn fireball 3D (shader lua + khoi + vet lua) khi xoe tay

Cai dat:
    pip install ursina mediapipe opencv-python pillow numpy

Chay:
    python Fireball3D.py

Dieu khien (BAN LIEN TUC):
    - Xoe ca 5 ngon tay          -> ban cau lua lien tuc ve phia man hinh
    - Nam tay / thu tay          -> ngung ban
    - Phim 'b'                   -> bat/tat bloom de so sanh
    - Phim ESC / dong cua so     -> thoat
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
    Ursina, Entity, camera, window, color, destroy, time as utime, Vec3, Vec2,
    application, Shader, Texture,
)

# ----------------------------------------------------------------------
# BLOOM SHADER (screen-space, tu viet vi Ursina khong co san)
# ----------------------------------------------------------------------
bloom_shader = Shader(fragment='''
#version 430
uniform sampler2D tex;
in vec2 uv;
out vec4 color;

uniform float threshold;   // nguong sang: chi pixel sang hon moi glow
uniform float intensity;   // do manh cua glow
uniform float blur_size;   // ban kinh lan toa (theo uv)

vec3 bright(vec3 c) {
    float b = max(c.r, max(c.g, c.b));
    return c * clamp((b - threshold) / (1.0 - threshold), 0.0, 1.0);
}

void main() {
    vec3 original = texture(tex, uv).rgb;
    vec3 glow = vec3(0.0);
    float total = 0.0;
    for (int i = 0; i < 16; i++) {
        float a = float(i) * 0.39269908;      // = 2*pi/16
        for (int r = 1; r <= 2; r++) {
            vec2 off = vec2(cos(a), sin(a)) * blur_size * float(r);
            float w = 1.0 / float(r);
            glow  += bright(texture(tex, uv + off).rgb) * w;
            total += w;
        }
    }
    glow /= total;
    color = vec4(original + glow * intensity, 1.0);
}
''',
default_input=dict(
    threshold=0.72,
    intensity=1.7,
    blur_size=0.013,
))


# ----------------------------------------------------------------------
# FIRE SHADER (procedural - buoc 5): sinh lua bang noise + time,
# mau tuoi sang (tu do -> cam -> vang -> trang), canh wispy nhu khoi.
# Dua tren cau truc unlit_shader cua Ursina (#130 vertex / #140 fragment).
# ----------------------------------------------------------------------
fire_shader = Shader(name='fire_shader', language=Shader.GLSL,
vertex='''
#version 130
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
uniform vec2 texture_scale;
uniform vec2 texture_offset;
out vec2 texcoords;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    texcoords = (p3d_MultiTexCoord0 * texture_scale) + texture_offset;
}
''',
fragment='''
#version 140
uniform vec4 p3d_ColorScale;
in vec2 texcoords;
out vec4 fragColor;

uniform float time;       // thoi gian -> lua chuyen dong
uniform float heat;       // 0..1: cao = nong/sang/trang, thap = do
uniform float softness;   // 0..1: cao = canh tan ra thanh khoi

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * vnoise(p); p *= 2.0; a *= 0.5; }
    return v;
}
vec3 firePalette(float t) {
    vec3 c1 = vec3(0.6, 0.0, 0.0);    // do tham
    vec3 c2 = vec3(1.0, 0.25, 0.0);   // cam do
    vec3 c3 = vec3(1.0, 0.6, 0.05);   // cam
    vec3 c4 = vec3(1.0, 0.9, 0.35);   // vang
    vec3 c5 = vec3(1.0, 1.0, 0.92);   // trang nong
    t = clamp(t, 0.0, 1.0);
    if (t < 0.25) return mix(c1, c2, t / 0.25);
    if (t < 0.5)  return mix(c2, c3, (t - 0.25) / 0.25);
    if (t < 0.75) return mix(c3, c4, (t - 0.5) / 0.25);
    return mix(c4, c5, (t - 0.75) / 0.25);
}
void main() {
    vec2 uv = texcoords;
    float n1 = fbm(uv * 4.0 + vec2(time * 0.25, -time * 1.1));
    float n2 = fbm(uv * 8.0 - vec2(0.0, time * 1.9));
    float f = n1 * 0.6 + n2 * 0.4;

    float t = clamp(f * 0.75 + heat * 0.55, 0.0, 1.0);
    vec3 col = firePalette(t) * (1.35 + heat * 0.7);   // > 1 de bloom bat sang

    float a = p3d_ColorScale.a;
    a *= mix(1.0, smoothstep(0.12, 0.65, f), softness); // canh tan thanh khoi
    fragColor = vec4(col, a);
}
''',
default_input=dict(
    texture_scale=Vec2(1, 1),
    texture_offset=Vec2(0, 0),
    time=0.0,
    heat=0.5,
    softness=0.0,
))


# ----------------------------------------------------------------------
# 1. WEBCAM + MEDIAPIPE
# ----------------------------------------------------------------------
CAM_W, CAM_H = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

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
    f = [1 if lm[4].x < lm[3].x else 0]
    for tip in (8, 12, 16, 20):
        f.append(1 if lm[tip].y < lm[tip - 2].y else 0)
    return f


# ----------------------------------------------------------------------
# 2. URSINA SCENE
# ----------------------------------------------------------------------
app = Ursina()
window.title = "Fireball Hand Magic"
window.color = color.black

camera.shader = bloom_shader
# Meo HDR: webcam bi giam sang xuong duoi threshold (BG_BRIGHTNESS < threshold)
# -> nen KHONG bao gio bi bloom/nhoe; chi lua (sang 1.0) moi glow.
camera.set_shader_input("threshold", 0.90)
camera.set_shader_input("intensity", 1.7)
camera.set_shader_input("blur_size", 0.013)

BG_BRIGHTNESS = 0.85     # do sang webcam (phai < threshold de nen khong glow)


def make_fire_texture(size=256):
    """Noise dam may TILEABLE (bang FFT low-pass) - dung cho khoi va tan lua."""
    rng = np.random.default_rng(7)
    white = rng.random((size, size))
    f = np.fft.fft2(white)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fx ** 2 + fy ** 2)
    filt = 1.0 / (radius * size * 0.15 + 1) ** 2
    img = np.fft.ifft2(f * filt).real
    img -= img.min()
    img /= (img.max() + 1e-9)
    img = img ** 1.4
    img = 0.5 + 0.5 * img
    arr = (img * 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Texture(Image.fromarray(rgb))


fire_tex = make_fire_texture()

# --- Background quad: dan webcam len ---
aspect = CAM_W / CAM_H
BG_Z = 50
fov_rad = math.radians(camera.fov)
bg_h = 2 * BG_Z * math.tan(fov_rad / 2)
bg_w = bg_h * aspect
background = Entity(parent=camera, model="quad", scale=(bg_w, bg_h),
                   z=BG_Z, double_sided=True)

# Mat phang spawn fireball
SPAWN_Z = 16
spawn_h = 2 * SPAWN_Z * math.tan(fov_rad / 2)
spawn_w = spawn_h * aspect


def hand_to_world(hx, hy):
    x = (hx - 0.5) * spawn_w
    y = (0.5 - hy) * spawn_h
    return Vec3(x, y, SPAWN_Z)


# ----------------------------------------------------------------------
# 3. PARTICLE: TAN LUA, KHOI, VET LUA
# ----------------------------------------------------------------------
class Ember(Entity):
    """Tan lua (spark) bay ra tu qua cau."""

    def __init__(self, position):
        super().__init__(
            model="sphere",
            color=random.choice([color.yellow, color.orange, color.white]),
            scale=random.uniform(0.25, 0.6),
            position=position,
            unlit=True,
        )
        self.velocity = Vec3(random.uniform(-3, 3),
                             random.uniform(-3, 3),
                             random.uniform(5, 11))
        self.life = random.uniform(0.35, 0.8)

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.scale *= 0.90
        self.alpha = max(0.0, self.life * 1.6)
        if self.life <= 0:
            destroy(self)


class Smoke(Entity):
    """Khoi: mang xam, no to dan, troi len va ra sau, mo dan."""

    def __init__(self, position, size=1.0):
        super().__init__(
            model="sphere",
            color=color.gray,
            texture=fire_tex,
            scale=size * random.uniform(0.8, 1.3),
            position=position,
            unlit=True,
        )
        self.velocity = Vec3(random.uniform(-1, 1),
                             random.uniform(0.5, 2.0),
                             random.uniform(2, 5))
        self.life = random.uniform(0.9, 1.6)
        self.max_life = self.life
        self.spin = random.uniform(-40, 40)

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.scale *= 1.02
        self.rotation_z += self.spin * utime.dt
        self.alpha = 0.30 * max(0.0, self.life / self.max_life)
        if self.life <= 0:
            destroy(self)


class TrailPuff(Entity):
    """Vet lua: dom sang o lai tai vi tri qua cau -> tao streak lien tuc."""

    def __init__(self, position, size=1.0):
        super().__init__(
            model="sphere",
            color=color.orange,
            texture=fire_tex,
            scale=size * 0.9,
            position=position,
            unlit=True,
        )
        self.life = 0.35

    def update(self):
        self.life -= utime.dt
        self.scale *= 0.90
        self.alpha = max(0.0, self.life * 2.2)
        if self.life <= 0:
            destroy(self)


# ----------------------------------------------------------------------
# 4. FIREBALL (dung fire_shader nhieu lop)
# ----------------------------------------------------------------------
class Fireball(Entity):
    # (scale, heat, softness, alpha) cho tung lop
    BASE = [
        (1.1, 1.00, 0.00, 1.00),   # loi trang nong
        (1.9, 0.70, 0.30, 0.85),   # vang
        (2.9, 0.42, 0.65, 0.55),   # cam
        (4.3, 0.18, 0.90, 0.30),   # quang do / khoi
    ]

    def __init__(self, position, velocity, power=1.0):
        super().__init__(position=position)
        self.velocity = velocity
        self.life = 3.6
        self.timer = 0.0
        self.size = 0.7 + power * 0.9          # to theo luong nap

        self.layers = []
        self.base_scales = []
        for (sc, heat, soft, a) in Fireball.BASE:
            e = Entity(parent=self, model="sphere", color=color.white,
                       shader=fire_shader)
            e.set_shader_input("heat", heat)
            e.set_shader_input("softness", soft)
            e.set_shader_input("texture_scale", Vec2(3, 3))
            e.set_shader_input("texture_offset",
                               Vec2(random.random(), random.random()))
            e.alpha = a
            self.layers.append(e)
            self.base_scales.append(sc * self.size)

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.timer += utime.dt

        flick = 1 + 0.18 * math.sin(self.timer * 45)
        flick2 = 1 + 0.10 * math.sin(self.timer * 27 + 1.5)
        for i, e in enumerate(self.layers):
            fl = flick if i % 2 == 0 else flick2
            e.scale = self.base_scales[i] * fl
            e.set_shader_input("time", self.timer * 1.6 + i * 3.3)

        # Vet lua lien tuc
        TrailPuff(self.world_position, self.size)
        # Khoi
        if random.random() < 0.4:
            Smoke(self.world_position, self.size)
        # Tan lua
        for _ in range(2):
            if random.random() < 0.8:
                Ember(self.world_position + Vec3(random.uniform(-0.6, 0.6),
                                                 random.uniform(-0.6, 0.6),
                                                 random.uniform(-0.3, 0.3)))

        if self.z < 0.8 or self.life <= 0:
            destroy(self)


# ----------------------------------------------------------------------
# 5. VONG LAP CHINH — BAN LIEN TUC
# ----------------------------------------------------------------------
_last_fire_time = 0.0
# Thoi gian hoi chieu giua 2 qua. Giam de ban nhanh hon, tang de cham lai.
FIRE_COOLDOWN = 0.15


def update():
    global _last_fire_time

    ok, frame = cap.read()
    if not ok:
        return
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Giam sang nen truoc khi hien thi de nen nam duoi nguong bloom
    # (MediaPipe van dung 'rgb' goc de nhan dien chinh xac)
    bg_rgb = (rgb.astype(np.uint16) * int(BG_BRIGHTNESS * 256) >> 8).astype(np.uint8)
    background.texture = None
    tex = _np_to_texture(bg_rgb)
    if tex:
        background.texture = tex

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms = int((time.time() - _t0) * 1000)
    res = detector.detect_for_video(mp_image, ts_ms)

    if not res.hand_landmarks:
        return

    lm = res.hand_landmarks[0]
    f = fingers_up(lm)
    palm_open = sum(f) == 5

    # Xoe tay + het cooldown -> ban
    if palm_open and (time.time() - _last_fire_time > FIRE_COOLDOWN):
        pos = hand_to_world(lm[9].x, lm[9].y)
        # jitter nhe de cac qua khong trung diem xuat phat
        pos += Vec3(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0)

        dx = lm[12].x - lm[0].x
        dy = lm[0].y - lm[12].y
        n = math.hypot(dx, dy) or 1
        aim = Vec3(dx / n * 2.5, dy / n * 2.5, -10)

        Fireball(pos, aim, power=random.uniform(0.5, 0.8))
        _last_fire_time = time.time()


def _np_to_texture(rgb):
    try:
        img = Image.fromarray(rgb)
        return Texture(img)
    except Exception as e:
        print("Texture error:", e)
        return None


def input(key):
    if key == "escape":
        application.quit()
    elif key == "b":
        camera.shader = None if camera.shader else bloom_shader


if __name__ == "__main__":
    print("Xoe tay de nap nang luong, nam tay lai de ban. ESC de thoat.")
    try:
        app.run()
    finally:
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