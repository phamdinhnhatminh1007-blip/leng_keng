"""
Fireball3D.py — MODULE HIEU UNG PHUN LUA (file phu)
===================================================
Chi chua nhung gi lien quan den hieu ung: shader, texture, particle,
luong phun lua. KHONG chua webcam / MediaPipe / ML / vong lap game
— nhung thu do nam o main.py.

Cach dung (xem main.py):
    from Fireball3D import bloom_shader, init_effects, cast

    app = Ursina()
    camera.shader = bloom_shader          # bat glow
    init_effects()                        # tao texture (goi SAU Ursina())
    ...
    cast(lm, hand_to_world, hand_id)      # phun lua khi con giu gesture
"""

import math
import os
import random
import time

import numpy as np
from PIL import Image
from panda3d.core import ColorBlendAttrib
from ursina import (
    Entity, camera, color, destroy, time as utime, Vec3, Vec2, Shader, Texture,
    Audio,
)

# --- Am thanh: dat file .wav/.ogg vao thu muc sounds/ ---
_SFX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")


def play_sound(name, volume=0.7):
    """Phat 1 file am thanh trong sounds/. Thieu file thi bo qua (khong crash).

    LUU Y: Ursina Audio tim file theo TEN trong application.asset_folder (quet
    de quy), KHONG nhan duong dan tuyet doi -> chi truyen ten (vd "fireball.wav").
    """
    if not os.path.exists(os.path.join(_SFX_DIR, name)):
        return
    try:
        Audio(name, autoplay=True, loop=False, volume=volume)
    except Exception as e:
        print("Sound error:", e)


def load_effect_texture(path):
    """Doc PNG (giu kenh alpha/trong suot) tu Canva thanh Ursina Texture."""
    return Texture(Image.open(path).convert("RGBA"))

# ----------------------------------------------------------------------
# BLOOM SHADER (screen-space post-processing, gan vao camera.shader)
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
    threshold=0.90,
    intensity=1.7,
    blur_size=0.013,
))


# ----------------------------------------------------------------------
# FIRE SHADER (procedural): sinh lua bang noise + time.
# heat: 0..1 (cao = nong/trang), softness: 0..1 (cao = canh tan nhu khoi)
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

uniform float time;
uniform float heat;
uniform float softness;

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
    vec3 c1 = vec3(0.6, 0.0, 0.0);
    vec3 c2 = vec3(1.0, 0.25, 0.0);
    vec3 c3 = vec3(1.0, 0.6, 0.05);
    vec3 c4 = vec3(1.0, 0.9, 0.35);
    vec3 c5 = vec3(1.0, 1.0, 0.92);
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
    float hot = smoothstep(0.48, 0.95, f + heat * 0.28);
    vec3 col = firePalette(t) * (1.20 + heat * 0.55);
    col += vec3(1.0, 0.34, 0.03) * hot * hot * 0.75;
    col *= p3d_ColorScale.rgb;

    float a = p3d_ColorScale.a;
    a *= mix(1.0, smoothstep(0.24, 0.70, f), softness);
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
# TEXTURE NOISE (dung cho khoi + vet lua)
# ----------------------------------------------------------------------
fire_tex = None       # noise cho khoi
glow_tex = None       # dia sang mem cho loi/trail
ring_tex = None       # vong xung kich khi no
flame_tex = None      # giot lua nhon dung cho luong phun


def make_fire_texture(size=256):
    """Noise dam may TILEABLE (FFT low-pass)."""
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


def make_radial_texture(size=256, ring=False):
    """Tao texture RGBA mem cho glow hoac vong xung kich."""
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    radius = np.sqrt(xx * xx + yy * yy)

    if ring:
        alpha = np.exp(-((radius - 0.58) / 0.09) ** 2)
        alpha *= np.clip((1.0 - radius) * 8.0, 0.0, 1.0)
    else:
        alpha = np.clip(1.0 - radius, 0.0, 1.0) ** 2.4

    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    return Texture(Image.fromarray(rgba, mode="RGBA"), filtering="bilinear")


def make_flame_texture(size=256):
    """Tao sprite ngon lua thuon nhon, meo nhe thay vi mot dom tron."""
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Than rong o goc, thu nhanh ve mui; bien uon song de giong lua bi gio xe.
    center = 0.12 * np.sin(yy * 11.0) * yy
    width = 0.54 * (1.0 - yy) ** 0.82 + 0.004
    edge = np.clip(1.0 - np.abs(xx - center) / width, 0.0, 1.0)
    vertical = np.clip(
        np.sin(np.pi * np.clip(yy, 0.0, 1.0)),
        0.0,
        1.0,
    ) ** 0.28
    alpha = (edge ** 1.65) * vertical

    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
    return Texture(Image.fromarray(rgba, mode="RGBA"), filtering="bilinear")


def enable_additive(entity):
    """Cong mau particle vao background de glow ma khong can bloom."""
    entity.setAttrib(ColorBlendAttrib.make(
        ColorBlendAttrib.M_add,
        ColorBlendAttrib.O_incoming_alpha,
        ColorBlendAttrib.O_one,
    ))
    entity.setDepthWrite(False)


def init_effects():
    """Goi 1 lan SAU app = Ursina() de tao texture dung chung."""
    global fire_tex, glow_tex, ring_tex, flame_tex
    if fire_tex is None:
        fire_tex = make_fire_texture()
    if glow_tex is None:
        glow_tex = make_radial_texture()
    if ring_tex is None:
        ring_tex = make_radial_texture(ring=True)
    if flame_tex is None:
        flame_tex = make_flame_texture()


# ----------------------------------------------------------------------
# PARTICLES: TAN LUA, KHOI, VET LUA
# ----------------------------------------------------------------------
class Ember(Entity):
    """Tan lua nho, sang va sac net."""

    def __init__(self, position, size=1.0, velocity=None):
        super().__init__(
            model="quad",
            texture=glow_tex,
            billboard=True,
            color=random.choice([
                color.rgba32(255, 245, 180, 255),
                color.rgba32(255, 155, 30, 255),
                color.rgba32(255, 70, 8, 255),
            ]),
            scale=size * random.uniform(0.12, 0.30),
            position=position,
            unlit=True,
            double_sided=True,
        )
        enable_additive(self)
        self.velocity = velocity or Vec3(
            random.uniform(-2.8, 2.8),
            random.uniform(-2.2, 3.5),
            random.uniform(2.0, 7.0),
        )
        self.life = random.uniform(0.28, 0.62)
        self.max_life = self.life

    def update(self):
        self.position += self.velocity * utime.dt
        self.velocity.y -= 1.2 * utime.dt
        self.life -= utime.dt
        self.scale *= max(0.0, 1.0 - 2.5 * utime.dt)
        self.alpha = max(0.0, self.life / self.max_life) ** 1.4
        if self.life <= 0:
            destroy(self)


class Smoke(Entity):
    """Khoi mem, no dan va troi len."""

    def __init__(self, position, size=1.0):
        super().__init__(
            model="quad",
            texture=glow_tex,
            billboard=True,
            color=color.rgba32(62, 52, 48, 72),
            scale=size * random.uniform(0.55, 0.95),
            position=position,
            unlit=True,
            double_sided=True,
        )
        self.velocity = Vec3(
            random.uniform(-0.7, 0.7),
            random.uniform(0.8, 1.8),
            random.uniform(1.0, 3.0),
        )
        self.life = random.uniform(0.65, 1.10)
        self.max_life = self.life
        self.spin = random.uniform(-40, 40)

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.scale *= 1.0 + 1.1 * utime.dt
        self.rotation_z += self.spin * utime.dt
        ratio = max(0.0, self.life / self.max_life)
        self.alpha = 0.22 * ratio * ratio
        if self.life <= 0:
            destroy(self)


class TrailPuff(Entity):
    """Vet lua additive nho gon, de lai duong bay lien tuc."""

    def __init__(self, position, size=1.0):
        super().__init__(
            model="quad",
            texture=glow_tex,
            billboard=True,
            color=random.choice([
                color.rgba32(255, 175, 35, 210),
                color.rgba32(255, 82, 8, 190),
            ]),
            scale=size * random.uniform(0.65, 0.95),
            position=position,
            unlit=True,
            double_sided=True,
        )
        enable_additive(self)
        self.life = random.uniform(0.22, 0.34)
        self.max_life = self.life
        self.velocity = Vec3(
            random.uniform(-0.30, 0.30),
            random.uniform(0.15, 0.65),
            random.uniform(0.25, 1.1),
        )

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.scale *= max(0.0, 1.0 - 2.0 * utime.dt)
        ratio = max(0.0, self.life / self.max_life)
        self.alpha = ratio * ratio
        if self.life <= 0:
            destroy(self)


class ImpactBurst(Entity):
    """Vu no bom lua: chop sang, hai vong xung kich, tan lua va khoi toa tron."""

    def __init__(self, position, size=1.0):
        super().__init__(position=position)
        self.life = 0.52
        self.max_life = self.life
        self.size = size

        self.flash = Entity(
            parent=self,
            model="quad",
            texture=glow_tex,
            billboard=True,
            color=color.rgba32(255, 205, 75, 240),
            scale=size * 2.0,
            unlit=True,
            double_sided=True,
        )
        self.ring = Entity(
            parent=self,
            model="quad",
            texture=ring_tex,
            billboard=True,
            color=color.rgba32(255, 92, 12, 230),
            scale=size * 1.2,
            unlit=True,
            double_sided=True,
        )
        self.inner_ring = Entity(
            parent=self,
            model="quad",
            texture=ring_tex,
            billboard=True,
            color=color.rgba32(255, 220, 92, 245),
            scale=size * 0.75,
            unlit=True,
            double_sided=True,
        )
        enable_additive(self.flash)
        enable_additive(self.ring)
        enable_additive(self.inner_ring)

        # Chia deu goc de vu no luon toe tron, them jitter de khong bi may moc.
        spark_count = max(28, int(34 * size))
        for i in range(spark_count):
            angle = 2 * math.pi * i / spark_count + random.uniform(-0.11, 0.11)
            speed = random.uniform(7.0, 16.0) * size
            direction = Vec3(
                math.cos(angle) * speed,
                math.sin(angle) * speed,
                random.uniform(-3.0, 6.0),
            )
            ember = Ember(position, size=size * random.uniform(0.65, 1.15),
                          velocity=direction)
            ember.life = random.uniform(0.38, 0.82)
            ember.max_life = ember.life

        # Khoi cung bung theo cac huong, sau do cham dan va boc len.
        for i in range(9):
            angle = 2 * math.pi * i / 9 + random.uniform(-0.18, 0.18)
            smoke = Smoke(
                position + Vec3(
                    random.uniform(-0.25, 0.25),
                    random.uniform(-0.25, 0.25),
                    random.uniform(-0.15, 0.15),
                ),
                size=size * random.uniform(0.75, 1.25),
            )
            smoke.velocity = Vec3(
                math.cos(angle) * random.uniform(1.8, 4.2) * size,
                math.sin(angle) * random.uniform(1.8, 4.2) * size + 1.0,
                random.uniform(0.5, 3.0),
            )

    def update(self):
        self.life -= utime.dt
        progress = 1.0 - max(0.0, self.life / self.max_life)
        self.flash.scale = self.size * (2.0 + progress * 4.5)
        self.ring.scale = self.size * (1.2 + progress * 7.5)
        self.inner_ring.scale = self.size * (0.75 + progress * 5.2)
        self.flash.alpha = max(0.0, 1.0 - progress * 1.8)
        self.ring.alpha = max(0.0, (1.0 - progress) ** 1.3)
        self.inner_ring.alpha = max(0.0, 1.0 - progress * 1.35) ** 1.5
        if self.life <= 0:
            destroy(self)


# ----------------------------------------------------------------------
# FIREBALL
# ----------------------------------------------------------------------
class Fireball(Entity):
    """Cau lua co loi nong, glow additive, trail va vu no ket thuc."""

    # (scale, heat, softness, alpha) cho tung lop
    BASE = [
        (0.72, 1.00, 0.00, 1.00),   # loi trang nong
        (1.10, 0.76, 0.18, 0.92),   # vang
        (1.58, 0.42, 0.52, 0.58),   # cam
        (2.05, 0.18, 0.82, 0.24),   # vien do mem
    ]

    def __init__(self, position, velocity, power=1.0):
        super().__init__(position=position)
        self.velocity = velocity
        self.life = 2.4
        self.timer = 0.0
        self.size = 0.65 + power * 0.75
        self.trail_timer = 0.0
        self.smoke_timer = 0.0
        self.ember_timer = 0.0

        self.layers = []
        self.base_scales = []
        for (sc, heat, soft, a) in Fireball.BASE:
            e = Entity(parent=self, model="sphere", color=color.white,
                       shader=fire_shader, unlit=True)
            e.set_shader_input("heat", heat)
            e.set_shader_input("softness", soft)
            e.set_shader_input("texture_scale", Vec2(4, 4))
            e.set_shader_input("texture_offset",
                               Vec2(random.random(), random.random()))
            e.alpha = a
            self.layers.append(e)
            self.base_scales.append(sc * self.size)

        self.glows = []
        for scale, tint in [
            (2.25, color.rgba32(255, 100, 12, 115)),
            (3.20, color.rgba32(255, 38, 4, 48)),
        ]:
            glow = Entity(
                parent=self,
                model="quad",
                texture=glow_tex,
                billboard=True,
                color=tint,
                scale=self.size * scale,
                unlit=True,
                double_sided=True,
            )
            enable_additive(glow)
            self.glows.append((glow, scale, tint.a))

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.timer += utime.dt

        flick = 1 + 0.08 * math.sin(self.timer * 38)
        flick2 = 1 + 0.05 * math.sin(self.timer * 23 + 1.5)
        for i, e in enumerate(self.layers):
            fl = flick if i % 2 == 0 else flick2
            e.scale = self.base_scales[i] * fl
            e.set_shader_input("time", self.timer * 1.6 + i * 3.3)
            e.rotation_y += (18 + i * 7) * utime.dt

        for i, (glow, scale, base_alpha) in enumerate(self.glows):
            pulse = 1.0 + 0.07 * math.sin(self.timer * (17 + i * 4) + i)
            glow.scale = self.size * scale * pulse
            glow.alpha = base_alpha * (0.88 + 0.12 * math.sin(self.timer * 21 + i))

        self.trail_timer -= utime.dt
        if self.trail_timer <= 0:
            TrailPuff(self.world_position, self.size)
            self.trail_timer = 0.035

        self.smoke_timer -= utime.dt
        if self.smoke_timer <= 0:
            Smoke(self.world_position, self.size)
            self.smoke_timer = 0.13

        self.ember_timer -= utime.dt
        if self.ember_timer <= 0:
            Ember(
                self.world_position + Vec3(
                    random.uniform(-0.35, 0.35),
                    random.uniform(-0.35, 0.35),
                    random.uniform(-0.20, 0.20),
                ),
                size=self.size,
            )
            self.ember_timer = 0.065

        if self.z < 0.8:            # dap vao man hinh -> no tia lua
            self._impact()
            destroy(self)
        elif self.life <= 0:        # chay het -> tat, khong no
            destroy(self)

    def _impact(self):
        """No nhu bom va toe tan lua theo moi huong khi dap vao man hinh."""
        p = self.world_position
        ImpactBurst(p, self.size * 1.15)
        camera.shake(
            duration=0.22,
            magnitude=min(1.0, 0.45 + self.size * 0.24),
            speed=0.018,
        )


# ----------------------------------------------------------------------
# FLAMETHROWER — cac luoi lua nhon ghep thanh luong phun lien tuc
# ----------------------------------------------------------------------
class FlameTongue(Entity):
    """Mot luoi lua nhon trong luong phun lien tuc tu long ban tay."""

    def __init__(self, position, velocity, size=1.0, angle=0.0, hot=False):
        tint = random.choice([
            color.rgba32(255, 54, 2, 250),
            color.rgba32(255, 102, 3, 255),
            color.rgba32(255, 176, 18, 255),
        ])
        super().__init__(
            model="quad",
            texture=flame_tex,
            billboard=True,
            color=tint,
            position=position,
            rotation_z=angle + random.uniform(-15, 15),
            unlit=True,
            double_sided=True,
        )
        enable_additive(self)
        self.velocity = Vec3(*velocity)
        self.life = random.uniform(0.68, 0.98)
        self.max_life = self.life
        self.age = 0.0
        self.phase = random.uniform(0, 2 * math.pi)
        self.spin = random.uniform(-75, 75)
        self.base_width = size * random.uniform(0.42, 0.70)
        self.base_length = size * random.uniform(4.8, 6.8)
        self.scale = Vec3(self.base_width, self.base_length, 1)

        # Loi vang-trang hep lam luong lua nong va sac, khong thanh dom tron.
        self.core = Entity(
            parent=self,
            model="quad",
            texture=flame_tex,
            color=(color.white if hot else color.rgba32(255, 224, 92, 245)),
            scale=Vec3(0.50, 0.84, 1),
            z=-0.01,
            unlit=True,
            double_sided=True,
        )
        enable_additive(self.core)

    def _die(self):
        p = self.world_position
        for _ in range(1):
            Ember(
                p,
                size=self.base_width * 1.4,
                velocity=Vec3(
                    random.uniform(-4.5, 4.5),
                    random.uniform(-2.0, 5.0),
                    random.uniform(0.5, 4.0),
                ),
            )
        if random.random() < 0.30:
            Smoke(p, size=self.base_width * 1.3)
        destroy(self)

    def update(self):
        dt = utime.dt
        self.age += dt
        self.life -= dt

        # Nhieu dong nho dao qua lai khac pha -> luong lua bi xe, du va bat on.
        self.velocity.x += math.sin(self.age * 19.0 + self.phase) * 3.8 * dt
        self.velocity.y += math.cos(self.age * 15.0 + self.phase) * 2.7 * dt
        self.position += self.velocity * dt
        self.rotation_z += self.spin * dt

        progress = min(1.0, self.age / self.max_life)
        flicker = 0.82 + 0.18 * math.sin(self.age * 43.0 + self.phase)
        self.scale = Vec3(
            self.base_width * (1.0 + progress * 1.25),
            self.base_length * (1.0 + progress * 0.42) * flicker,
            1,
        )
        fade = max(0.0, 1.0 - progress)
        self.alpha = min(1.0, (fade ** 0.48) * 1.12)
        self.core.alpha = fade ** 0.9
        self.core.scale_x = 0.46 + progress * 0.18

        if random.random() < 1.2 * dt:
            Ember(self.world_position, size=self.base_width * 0.9)
        if random.random() < 0.55 * dt and progress > 0.35:
            Smoke(self.world_position, size=self.base_width)

        if self.z < 0.9 or self.life <= 0:
            self._die()


class FlameStream(Entity):
    """Emitter giu than lua lien mach, khong phu thuoc FPS nhan dien tay."""

    EMIT_INTERVAL = 0.022
    REFRESH_TIMEOUT = 0.22

    def __init__(self, hand_id):
        super().__init__()
        self.hand_id = hand_id
        self.nozzle = Vec3(0, 0, 0)
        self.base_velocity = Vec3(0, 0, -19)
        self.ux = 0.0
        self.uy = 1.0
        self.angle = 0.0
        self.last_refresh = time.time()
        self.emit_timer = 0.0
        self.timer = 0.0

        # Hai lop gan tay luon ton tai de noi cac particle thanh mot than lua.
        self.outer_source = Entity(
            parent=self,
            model="quad",
            texture=flame_tex,
            billboard=True,
            color=color.rgba32(255, 64, 2, 255),
            unlit=True,
            double_sided=True,
        )
        self.hot_source = Entity(
            parent=self,
            model="quad",
            texture=flame_tex,
            billboard=True,
            color=color.rgba32(255, 238, 125, 255),
            z=-0.012,
            unlit=True,
            double_sided=True,
        )
        enable_additive(self.outer_source)
        enable_additive(self.hot_source)

        # Cac lop cau noi nam sau dan theo truc z. Chung lap hinh len nhau,
        # noi nguon o long ban tay voi particle dang lao ra phia camera.
        self.bridges = []
        bridge_specs = [
            (1.5, 0.82, 4.5, color.rgba32(255, 225, 90, 250)),
            (3.2, 1.02, 5.0, color.rgba32(255, 150, 18, 255)),
            (5.2, 1.28, 5.6, color.rgba32(255, 82, 3, 250)),
            (7.4, 1.58, 6.2, color.rgba32(238, 38, 1, 235)),
        ]
        for depth, width, length, tint in bridge_specs:
            bridge = Entity(
                parent=self,
                model="quad",
                texture=flame_tex,
                billboard=True,
                color=tint,
                unlit=True,
                double_sided=True,
            )
            enable_additive(bridge)
            self.bridges.append((bridge, depth, width, length))

    def refresh(self, nozzle, velocity, ux, uy, angle):
        self.nozzle = Vec3(*nozzle)
        self.base_velocity = Vec3(*velocity)
        self.ux = ux
        self.uy = uy
        self.angle = angle
        self.last_refresh = time.time()
        self.position = self.nozzle

    def _emit(self):
        # Ba dong lap day; dong giua la loi nong, hai dong ngoai xe mui lua.
        for i in range(3):
            spread = 0.42 if i == 0 else 0.9
            velocity = self.base_velocity + Vec3(
                random.uniform(-2.1, 2.1) * spread,
                random.uniform(-1.8, 2.5) * spread,
                random.uniform(-1.6, 1.0),
            )
            spawn = self.nozzle + Vec3(
                random.uniform(-0.16, 0.16),
                random.uniform(-0.14, 0.14),
                random.uniform(-0.10, 0.10),
            )
            FlameTongue(
                spawn,
                velocity,
                size=random.uniform(0.86, 1.18),
                angle=self.angle,
                hot=(i == 0),
            )

    def update(self):
        dt = utime.dt
        self.timer += dt
        idle = time.time() - self.last_refresh
        if idle > self.REFRESH_TIMEOUT:
            if _streams.get(self.hand_id) is self:
                _streams.pop(self.hand_id, None)
            destroy(self)
            return

        self.position = self.nozzle
        flicker = 0.90 + 0.10 * math.sin(self.timer * 47.0)
        self.outer_source.rotation_z = self.angle + math.sin(self.timer * 21) * 5
        self.hot_source.rotation_z = self.angle - math.sin(self.timer * 27) * 3
        self.outer_source.scale = Vec3(1.15 * flicker, 5.8, 1)
        self.hot_source.scale = Vec3(0.58 * flicker, 4.5, 1)
        self.outer_source.position = Vec3(0, 0, 0)
        self.hot_source.position = Vec3(0, 0, -0.02)

        for i, (bridge, depth, width, length) in enumerate(self.bridges):
            wave = math.sin(self.timer * (19 + i * 3) + i) * 0.12
            pulse = 0.90 + 0.10 * math.sin(self.timer * (37 + i * 2) + i)
            bridge.position = Vec3(
                self.ux * depth * 0.24 + wave,
                self.uy * depth * 0.24 - wave * 0.5,
                -depth,
            )
            bridge.rotation_z = self.angle + math.sin(self.timer * 23 + i) * 7
            bridge.scale = Vec3(width * pulse, length * pulse, 1)
            bridge.alpha = 0.82 + 0.18 * pulse

        self.emit_timer -= dt
        while self.emit_timer <= 0:
            self._emit()
            self.emit_timer += self.EMIT_INTERVAL


FIRE_SOUND_COOLDOWN = 0.55
_streams = {}
_last_sound = {}


def cast(lm, hand_to_world, hand_id=""):
    """Chieu "Xoe ban tay": phun mot luong lua lien tuc ve phia man hinh.

    - lm           : 21 landmarks ban tay (tu MediaPipe)
    - hand_to_world: ham doi toa do tay -> toa do 3D (main.py truyen vao)
    - hand_id      : khoa phan biet tay (vd "Left"/"Right") de cooldown doc lap
    """
    now = time.time()
    # Tam long ban tay = trung diem giua co tay va tam hang bon khop goc.
    # Cach nay dat nguon thap hon centroid cu, dung vao phan rong cua long tay.
    mcp_ids = (5, 9, 13, 17)
    mcp_x = sum(lm[i].x for i in mcp_ids) / len(mcp_ids)
    mcp_y = sum(lm[i].y for i in mcp_ids) / len(mcp_ids)
    palm_x = (lm[0].x + mcp_x) * 0.5
    palm_y = (lm[0].y + mcp_y) * 0.5
    pos = hand_to_world(palm_x, palm_y)
    dx = lm[12].x - lm[0].x
    dy = lm[0].y - lm[12].y
    n = math.hypot(dx, dy) or 1
    ux, uy = dx / n, dy / n
    nozzle = pos
    base_velocity = Vec3(ux * 5.0, uy * 5.0, -17.0)
    screen_angle = -math.degrees(math.atan2(ux, uy))

    stream = _streams.get(hand_id)
    if stream is None:
        stream = FlameStream(hand_id)
        _streams[hand_id] = stream
    stream.refresh(nozzle, base_velocity, ux, uy, screen_angle)

    if now - _last_sound.get(hand_id, 0.0) >= FIRE_SOUND_COOLDOWN:
        play_sound("fireball.wav", volume=0.55)
        _last_sound[hand_id] = now


# ----------------------------------------------------------------------
# SPRITE EFFECT (BILLBOARD) — dung anh tu Canva, van giu do sau 3D
# ----------------------------------------------------------------------
class SpriteEffect(Entity):
    """Tam anh (quad) luon quay mat ve camera, bay trong khong gian 3D.

    - texture : Ursina Texture (dung load_effect_texture("file.png"))
    - Anh tinh: de frames=1.
    - Sprite sheet (nhieu frame trong 1 anh): dat frames, columns, fps.
      Vi du sheet 4x4 = 16 frame -> frames=16, columns=4.
    """

    def __init__(self, position, velocity, texture, size=3.0, life=3.0,
                 frames=1, columns=1, fps=20, spin=0.0, fade=False,
                 grow=1.0):
        super().__init__(model="quad", texture=texture, position=position,
                         double_sided=True, scale=size)
        self.velocity = velocity
        self.life = life
        self.max_life = life
        self.spin = spin                       # xoay quanh truc nhin (do/giay)
        self.fade = fade                       # True -> mo dan theo life
        self.grow = grow                       # >1 to dan, <1 nho dan moi giay
        self.timer = 0.0

        # Sprite sheet
        self.frames = frames
        self.columns = max(1, columns)
        self.rows = math.ceil(frames / self.columns)
        self.fps = fps
        if frames > 1:
            self.texture_scale = Vec2(1 / self.columns, 1 / self.rows)

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.timer += utime.dt

        # Luon quay mat ve phia camera (billboard)
        self.look_at(camera.world_position)
        if self.spin:
            self.rotation_z += self.spin * utime.dt
        if self.grow != 1.0:
            self.scale *= (1 + (self.grow - 1) * utime.dt)
        if self.fade:
            self.alpha = max(0.0, self.life / self.max_life)

        # Chay animation neu la sprite sheet
        if self.frames > 1:
            idx = int(self.timer * self.fps) % self.frames
            col = idx % self.columns
            row = idx // self.columns
            self.texture_offset = Vec2(col / self.columns,
                                       1 - (row + 1) / self.rows)

        if self.z < 0.8 or self.life <= 0:
            ImpactBurst(self.world_position, self.size)
            destroy(self)
