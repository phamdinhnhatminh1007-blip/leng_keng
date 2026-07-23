"""
Fireball3D.py — MODULE HIEU UNG CAU LUA (file phu)
===================================================
Chi chua nhung gi lien quan den hieu ung: shader, texture, particle,
class Fireball. KHONG chua webcam / MediaPipe / ML / vong lap game
— nhung thu do nam o main.py.

Cach dung (xem main.py):
    from Fireball3D import bloom_shader, init_effects, Fireball

    app = Ursina()
    camera.shader = bloom_shader          # bat glow
    init_effects()                        # tao texture (goi SAU Ursina())
    ...
    Fireball(position, velocity, power)   # ban 1 qua cau lua
"""

import math
import random

import numpy as np
from PIL import Image
from ursina import (
    Entity, color, destroy, time as utime, Vec3, Vec2, Shader, Texture,
)

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
    vec3 col = firePalette(t) * (1.35 + heat * 0.7);

    float a = p3d_ColorScale.a;
    a *= mix(1.0, smoothstep(0.12, 0.65, f), softness);
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
fire_tex = None      # duoc tao boi init_effects() sau khi Ursina() khoi dong


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


def init_effects():
    """Goi 1 lan SAU app = Ursina() de tao texture dung chung."""
    global fire_tex
    if fire_tex is None:
        fire_tex = make_fire_texture()


# ----------------------------------------------------------------------
# PARTICLES: TAN LUA, KHOI, VET LUA
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
# FIREBALL
# ----------------------------------------------------------------------
class Fireball(Entity):
    """Qua cau lua nhieu lop dung fire_shader, tu nha khoi/tan lua/vet lua."""

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
        self.size = 0.7 + power * 0.9          # to theo power

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