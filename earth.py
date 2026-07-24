"""
earth.py — MODULE HIEU UNG THO / DAT (Earthbending) (file phu)
==============================================================
Nam dam -> dung len mot TAM KHIEN DA va GIU MAI khi con nam dam (khien bam
theo tay). Khi bo nam dam -> khien tu SUP DO thanh manh da + bui.

Cach dung (trong main.py):
    import earth
    SKILLS = { 1: earth.cast, ... }     # 1 = "Nam dam"
"""

import math
import os
import random

import numpy as np
from PIL import Image
from ursina import (
    Entity, camera, color, destroy, time as utime, Vec3, Vec2, Texture, Audio,
)

# --- Am thanh: dat file .wav/.ogg vao thu muc sounds/ (tuy chon) ---
_SFX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")


def play_sound(name, volume=0.8):
    if not os.path.exists(os.path.join(_SFX_DIR, name)):
        return
    try:
        Audio(name, autoplay=True, loop=False, volume=volume)
    except Exception as e:
        print("Sound error:", e)


def _lerp3(a, b, t):
    return Vec3(a.x + (b.x - a.x) * t,
               a.y + (b.y - a.y) * t,
               a.z + (b.z - a.z) * t)


# ----------------------------------------------------------------------
# TEXTURE DAT/DA — noise nau xu xi, co van nut (nhieu chi tiet)
# ----------------------------------------------------------------------
rock_tex = None


def _make_rock_texture(size=256):
    """Noise nau sam TILEABLE, tron 2 tan so + van nut cho ra chat da."""
    rng = np.random.default_rng(11)
    white = rng.random((size, size))
    f = np.fft.fft2(white)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)

    low = np.fft.ifft2(f * (1.0 / (r * size * 0.10 + 1) ** 2)).real
    det = np.fft.ifft2(f * (1.0 / (r * size * 0.45 + 1) ** 1.3)).real
    for a in (low, det):
        a -= a.min(); a /= (a.max() + 1e-9)

    v = 0.40 + 0.45 * low + 0.15 * det
    crack = np.clip(np.abs(det - 0.5) * 5.0, 0, 1)    # van nut toi
    v *= (0.55 + 0.45 * crack)
    v = np.clip(v ** 1.1, 0, 1)

    brown = np.clip(np.stack([v * 0.60, v * 0.42, v * 0.28], axis=-1) * 1.7, 0, 1)
    arr = (brown * 255).astype(np.uint8)
    return Texture(Image.fromarray(arr))


def _ensure_tex():
    global rock_tex
    if rock_tex is None:
        rock_tex = _make_rock_texture()
    return rock_tex


# ----------------------------------------------------------------------
# BUI DAT
# ----------------------------------------------------------------------
class Dust(Entity):
    """Dam bui nau-xam: no to dan, troi ra, mo dan."""

    def __init__(self, position, size=1.0):
        super().__init__(
            model="sphere", texture=rock_tex,
            color=color.rgb32(120, 95, 70),
            scale=size * random.uniform(0.8, 1.5),
            position=position, unlit=True,
        )
        self.velocity = Vec3(random.uniform(-3, 3),
                             random.uniform(-1, 3),
                             random.uniform(-2, 4))
        self.life = random.uniform(0.5, 1.1)
        self.max_life = self.life

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.scale *= 1.03
        self.rotation_y += 60 * utime.dt
        self.alpha = 0.5 * max(0.0, self.life / self.max_life)
        if self.life <= 0:
            destroy(self)


# ----------------------------------------------------------------------
# TAM KHIEN DA — dung len, GIU MAI (bam theo tay), roi sup do khi tha
# ----------------------------------------------------------------------
_shields = {}          # {hand_id: RockShield dang song}


class RockShield(Entity):
    def __init__(self, position, size=1.0):
        super().__init__(position=Vec3(*position))
        self.size = size
        self.hand_id = ""
        self.target_center = Vec3(*position)   # vi tri tay de bam theo

        self.assemble = 0.32       # thoi gian khien dung len
        self.age = 0.0
        self.since_refresh = 0.0   # lau chua duoc 'refresh' -> mat nam dam
        self.timeout = 0.22        # qua nguong nay -> bat dau sup do
        self.crumbling = False
        self.crumble_age = 0.0
        self.crumble_dur = 0.7
        self.dead = False

        self.blocks, self.starts, self.targets = [], [], []
        self.finals, self.spins, self.vels = [], [], []

        # Bo cuc: cac vong dong tam -> tam khien tron day khoi
        pts = [Vec3(0, 0, 0)]
        for radius, count in ((1.3, 6), (2.5, 12), (3.5, 16)):
            for k in range(count):
                a = k / count * 2 * math.pi + random.uniform(-0.15, 0.15)
                pts.append(Vec3(math.cos(a) * radius, math.sin(a) * radius, 0))

        for tp in pts:
            depth = random.uniform(-0.3, 0.6)             # loi lom -> san sui
            target = Vec3(tp.x, tp.y, -depth) * size      # -z = loi ve camera
            start = target + Vec3(random.uniform(-3, 3),
                                  random.uniform(-4, 4), 9) * size

            b = Entity(parent=self, model="cube", texture=rock_tex,
                       color=color.rgb32(random.randint(110, 165),
                                         random.randint(80, 120),
                                         random.randint(55, 88)),
                       unlit=True, position=start)
            fscale = size * random.uniform(0.9, 1.5)
            b.rotation = Vec3(random.uniform(-25, 25), random.uniform(-25, 25),
                              random.uniform(0, 360))
            b.texture_scale = Vec2(random.uniform(1, 2.2), random.uniform(1, 2.2))
            b.texture_offset = Vec2(random.random(), random.random())
            b.scale = fscale * 0.2

            self.blocks.append(b)
            self.starts.append(start)
            self.targets.append(target)
            self.finals.append(fscale)
            self.spins.append(Vec3(random.uniform(-200, 200),
                                   random.uniform(-200, 200),
                                   random.uniform(-200, 200)))
            self.vels.append(Vec3(0, 0, 0))

        # Chan dong + bui luc khien dung len -> cam giac luc
        camera.shake(duration=0.28, magnitude=0.6 * size, speed=0.02)
        for _ in range(int(14 * size)):
            Dust(self.world_position + Vec3(random.uniform(-3, 3),
                                            random.uniform(-3, 3), 0) * size, size)

    def refresh(self, pos):
        """Con nam dam -> goi moi frame: bam theo tay + gia han (khong sup)."""
        self.target_center = Vec3(*pos)
        self.since_refresh = 0.0

    def _start_crumble(self):
        self.crumbling = True
        for i in range(len(self.blocks)):
            self.vels[i] = Vec3(random.uniform(-4, 4),
                                random.uniform(-1, 3),
                                random.uniform(-5, -1))
        for _ in range(int(20 * self.size)):
            Dust(self.world_position, self.size)

    def update(self):
        dt = utime.dt
        # Bam theo tay (muot)
        self.position = _lerp3(self.position, self.target_center, min(1, 12 * dt))

        if self.crumbling:                               # DANG SUP DO
            self.crumble_age += dt
            for i, b in enumerate(self.blocks):
                self.vels[i] += Vec3(0, -20, 0) * dt     # trong luc
                b.position += self.vels[i] * dt
                b.rotation += self.spins[i] * dt
            if self.crumble_age >= self.crumble_dur:
                for b in self.blocks:
                    destroy(b)
                self.dead = True
                if _shields.get(self.hand_id) is self:
                    _shields.pop(self.hand_id, None)
                destroy(self)
            return

        self.since_refresh += dt

        if self.age < self.assemble:                     # DANG DUNG LEN
            self.age += dt
            t = min(1, self.age / self.assemble)
            e = 1 - (1 - t) ** 3                          # ease-out
            for i, b in enumerate(self.blocks):
                b.position = _lerp3(self.starts[i], self.targets[i], e)
                b.scale = self.finals[i] * (0.2 + 0.8 * e)
        else:                                            # GIU VUNG (mai)
            for i, b in enumerate(self.blocks):
                b.position = self.targets[i]
                b.scale = self.finals[i]

        # Khong con nam dam (lau chua refresh) -> sup do
        if self.since_refresh > self.timeout:
            self._start_crumble()


# ----------------------------------------------------------------------
# LOGIC TUNG CHIEU — main.py chi goi cast(lm, hand_to_world)
# ----------------------------------------------------------------------
def cast(lm, hand_to_world, hand_id=""):
    """Chieu "Nam dam": giu khien da khi con nam dam, bo tay thi sup do.

    Duoc goi MOI FRAME khi dang nam dam -> lan dau dung khien, cac lan sau
    chi 'refresh' de khien bam theo tay va khong sup.
    """
    _ensure_tex()
    pos = hand_to_world(lm[9].x, lm[9].y)
    pos.z -= 3                       # dat khien ra truoc long ban tay

    sh = _shields.get(hand_id)
    if sh is None or sh.dead or sh.crumbling:
        sh = RockShield(pos, size=1.3)
        sh.hand_id = hand_id
        _shields[hand_id] = sh
        play_sound("earth.wav", volume=0.8)
    else:
        sh.refresh(pos)
