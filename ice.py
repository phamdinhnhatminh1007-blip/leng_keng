"""
ice.py — MODULE HIEU UNG THUY / BANG (Icebending) (file phu)
============================================================
Do ngon tro + ngon ut -> mot DAY COT BANG dam tu long dat CHUI LEN quanh
tay: cac cot bang moc voi tu day len (cao dan o giua), rung man hinh cho
uy luc, toa suong lanh, giu mot lat roi tu tut xuong / vo tan.

Cach dung (trong main.py):
    import ice
    SKILLS = { 5: ice.cast, ... }     # 5 = "Do ngon tro va ngon ut"
"""

import math
import os
import random
import time

from ursina import (
    Entity, camera, color, destroy, time as utime, Vec3, Audio,
)

_SFX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")


def play_sound(name, volume=0.8):
    if not os.path.exists(os.path.join(_SFX_DIR, name)):
        return
    try:
        Audio(name, autoplay=True, loop=False, volume=volume)
    except Exception as e:
        print("Sound error:", e)


def _ice_color():
    return color.rgb32(random.randint(150, 185),
                       random.randint(205, 228),
                       random.randint(225, 240))    # xanh bang nhat


# ----------------------------------------------------------------------
# SUONG LANH + MANH BANG
# ----------------------------------------------------------------------
class Frost(Entity):
    """Man suong lanh: no nhe, troi len, mo dan."""

    def __init__(self, position):
        super().__init__(model="sphere", color=color.rgb32(200, 230, 245),
                         scale=random.uniform(0.5, 1.2), position=position,
                         unlit=True)
        self.velocity = Vec3(random.uniform(-1.5, 1.5),
                             random.uniform(0.5, 2.2),
                             random.uniform(-1.5, 1.5))
        self.life = random.uniform(0.5, 1.1)
        self.max_life = self.life

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.scale *= 1.03
        self.alpha = 0.4 * max(0.0, self.life / self.max_life)
        if self.life <= 0:
            destroy(self)


class IceShard(Entity):
    """Manh bang van ra khi cot bang vo."""

    def __init__(self, position):
        super().__init__(model="cube", color=_ice_color(),
                         scale=random.uniform(0.25, 0.6), position=position,
                         unlit=True, rotation_y=45)
        self.velocity = Vec3(random.uniform(-7, 7),
                             random.uniform(1, 8),
                             random.uniform(-5, 3))
        self.spin = Vec3(random.uniform(-360, 360), random.uniform(-360, 360),
                         random.uniform(-360, 360))
        self.life = random.uniform(0.4, 0.8)

    def update(self):
        self.velocity += Vec3(0, -16, 0) * utime.dt      # roi
        self.position += self.velocity * utime.dt
        self.rotation += self.spin * utime.dt
        self.life -= utime.dt
        if self.life < 0.3:
            self.scale *= 0.9
        if self.life <= 0:
            destroy(self)


# ----------------------------------------------------------------------
# COT BANG — moc tu day len, giu, roi tut xuong + vo
# ----------------------------------------------------------------------
class IceSpike(Entity):
    def __init__(self, base_pos, height, thin, delay=0.0, hold=0.9):
        super().__init__(model="cube", color=_ice_color(), unlit=True)
        self.base = Vec3(*base_pos)
        self.H = height
        self.thin = thin
        self.delay = delay
        self.hold = hold
        self.rise_dur = 0.20
        self.rotation = Vec3(0, random.uniform(0, 360), 0)   # tinh the goc canh
        self.state = "wait"
        self.t = 0.0
        self._set_h(0.001)

    def _set_h(self, h):
        """Giu DAY co dinh, moc len tren (chui tu dat len)."""
        h = max(0.001, h)
        self.scale = Vec3(self.thin, h, self.thin)
        self.position = Vec3(self.base.x, self.base.y + h / 2, self.base.z)

    def _shatter(self):
        for _ in range(random.randint(2, 4)):
            IceShard(Vec3(self.base.x, self.base.y + self.H, self.base.z))

    def update(self):
        dt = utime.dt
        self.t += dt
        if self.state == "wait":
            if self.t >= self.delay:
                self.state, self.t = "rise", 0.0
        elif self.state == "rise":
            k = min(1, self.t / self.rise_dur)
            self._set_h(self.H * (1 - (1 - k) ** 3))     # ease-out: bung nhanh
            if k >= 1:
                self.state, self.t = "hold", 0.0
        elif self.state == "hold":
            if self.t >= self.hold:
                self.state, self.t = "fall", 0.0
                self._shatter()
        elif self.state == "fall":
            k = min(1, self.t / 0.3)
            self._set_h(self.H * (1 - k))                # tut xuong
            if k >= 1:
                destroy(self)


# ----------------------------------------------------------------------
# TUNG CHIEU
# ----------------------------------------------------------------------
def cast_ice(center):
    """Dung mot day cot bang chui len tu duoi quanh 'center'."""
    center = Vec3(*center)
    ground_y = center.y - 5.0            # 'mat dat' o duoi tay

    # Chan dong manh cho uy luc
    camera.shake(duration=0.32, magnitude=0.8, speed=0.02)

    # Nhieu hang cot bang lech chieu sau, TRAI RONG ngang, cao dan o giua
    count = 17                            # nhieu cot -> tuong bang rong
    spacing = 1.35                        # khoang cach ngang -> trai rong hon
    for row, zoff in enumerate((0.0, 1.7, 3.4)):
        for i in range(count):
            fx = i - (count - 1) / 2
            dist = abs(fx)
            x = center.x + fx * spacing + random.uniform(-0.25, 0.25)
            z = center.z + zoff + random.uniform(-0.3, 0.3)
            H = max(2.0, 6.0 - dist * 0.35) * random.uniform(0.85, 1.15)
            thin = random.uniform(0.45, 0.85)
            delay = dist * 0.03 + row * 0.06     # bung lan ra 2 ben (song)
            IceSpike(Vec3(x, ground_y, z), H, thin, delay=delay,
                     hold=random.uniform(0.8, 1.2))

    # Suong lanh bung len (trai rong theo tuong bang)
    for _ in range(40):
        Frost(center + Vec3(random.uniform(-11, 11), random.uniform(-3, 1),
                            random.uniform(-1, 4)))


# ----------------------------------------------------------------------
# LOGIC TUNG CHIEU — main.py chi goi cast(lm, hand_to_world)
# ----------------------------------------------------------------------
ICE_COOLDOWN = 1.3           # giay giua 2 lan dung bang
_last_cast = {}


def cast(lm, hand_to_world, hand_id=""):
    """Chieu "Do ngon tro + ut": day cot bang chui len (cooldown rieng moi tay)."""
    if time.time() - _last_cast.get(hand_id, 0.0) < ICE_COOLDOWN:
        return
    center = hand_to_world(lm[9].x, lm[9].y)
    cast_ice(center)
    play_sound("ice.wav", volume=0.8)
    _last_cast[hand_id] = time.time()
