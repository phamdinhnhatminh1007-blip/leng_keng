"""
air.py — MODULE HIEU UNG KHI / GIO (Airbending) (file phu)
==========================================================
Do ngon cai -> mot CON LOC XOAY bao quanh ban tay: cot gio gom nhieu
luong xoay tron va boc len, phinh ra o dinh (hinh pheu). GIU khi con do
ngon cai (lo bam theo tay), bo tay thi loc TAN ra.

Cach dung (trong main.py):
    import air
    SKILLS = { 4: air.cast, ... }     # 4 = "Do ngon cai"
"""

import math
import os
import random

from ursina import (
    Entity, color, destroy, time as utime, Vec3, Audio,
)

# --- Am thanh (tuy chon): sounds/wind.wav ---
_SFX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")


def play_loop(name, volume=0.6):
    """Phat am thanh LAP, tra ve doi tuong Audio de sau con tat duoc.
    Thieu file -> tra None (khong crash)."""
    if not os.path.exists(os.path.join(_SFX_DIR, name)):
        return None
    try:
        return Audio(name, autoplay=True, loop=True, volume=volume)
    except Exception as e:
        print("Sound error:", e)
        return None


def _lerp3(a, b, t):
    return Vec3(a.x + (b.x - a.x) * t,
               a.y + (b.y - a.y) * t,
               a.z + (b.z - a.z) * t)


# ----------------------------------------------------------------------
# CON LOC — cot gio xoay tron quanh tay, giu mai, tan khi tha
# ----------------------------------------------------------------------
_tornados = {}         # {hand_id: Tornado dang song}

N_BLADES = 54          # so luong gio
HEIGHT = 5.5           # chieu cao cot loc
R_BOT = 0.7            # ban kinh day (hep)
R_TOP = 2.4            # ban kinh dinh (rong -> hinh pheu)


class Tornado(Entity):
    def __init__(self, position, size=1.0):
        super().__init__(position=Vec3(*position))
        self.size = size
        self.hand_id = ""
        self.target_center = Vec3(*position)

        self.since_refresh = 0.0
        self.timeout = 0.22
        self.dissipating = False
        self.dis_age = 0.0
        self.dis_dur = 0.5
        self.dead = False

        self.blades, self.vels = [], []
        for _ in range(N_BLADES):
            b = Entity(parent=self, model="cube", unlit=True,
                       color=color.rgb32(random.randint(185, 215),
                                         random.randint(205, 225),
                                         random.randint(215, 228)))
            b.a = random.uniform(0, 2 * math.pi)         # goc quanh truc
            b.h = random.uniform(0, HEIGHT)              # do cao hien tai
            b.aspeed = random.uniform(6, 9)              # toc do xoay (rad/s)
            b.rise = random.uniform(2.5, 4.5)            # toc do boc len
            b.alpha0 = random.uniform(0.35, 0.6)
            b.alpha = b.alpha0
            self.blades.append(b)
            self.vels.append(Vec3(0, 0, 0))
        self._layout()

        # Tieng gio LAP: keu suot khi con do ngon cai
        self.sound = play_loop("wind.wav", volume=0.5)

    def _layout(self):
        """Dat cac luong gio len duong xoan oc (goi moi frame khi con song)."""
        s = self.size
        for b in self.blades:
            r = (R_BOT + (R_TOP - R_BOT) * (b.h / HEIGHT))       # pheu: rong dan len
            x = math.cos(b.a) * r
            z = math.sin(b.a) * r
            y = b.h - HEIGHT / 2
            b.position = Vec3(x, y, z) * s
            b.rotation_y = math.degrees(-b.a)                    # nam theo phuong xoay
            # luong gio thuon dai theo phuong xoay
            b.scale = Vec3(0.9 * r * 0.5 + 0.4, 0.12, 0.12) * s

    def refresh(self, pos):
        self.target_center = Vec3(*pos)
        self.since_refresh = 0.0

    def _start_dissipate(self):
        self.dissipating = True
        # Tat tieng gio ngay khi bo ngon cai
        if self.sound:
            try:
                self.sound.stop(destroy=True)
            except Exception:
                pass
            self.sound = None
        for i, b in enumerate(self.blades):
            r = math.cos(b.a), math.sin(b.a)
            self.vels[i] = Vec3(r[0] * random.uniform(3, 7),
                                random.uniform(2, 6),
                                r[1] * random.uniform(3, 7))

    def update(self):
        dt = utime.dt
        self.position = _lerp3(self.position, self.target_center, min(1, 14 * dt))

        if self.dissipating:                          # DANG TAN
            self.dis_age += dt
            k = max(0.0, 1 - self.dis_age / self.dis_dur)
            for i, b in enumerate(self.blades):
                b.position += self.vels[i] * dt
                b.rotation_y += 200 * dt
                b.alpha = b.alpha0 * k
            if self.dis_age >= self.dis_dur:
                for b in self.blades:
                    destroy(b)
                self.dead = True
                if _tornados.get(self.hand_id) is self:
                    _tornados.pop(self.hand_id, None)
                destroy(self)
            return

        self.since_refresh += dt
        # Xoay + boc len
        for b in self.blades:
            b.a += b.aspeed * dt
            b.h += b.rise * dt
            if b.h > HEIGHT:                           # len den dinh -> ve day
                b.h -= HEIGHT
                b.a += random.uniform(-0.5, 0.5)
        self._layout()

        if self.since_refresh > self.timeout:          # bo ngon cai -> tan
            self._start_dissipate()


# ----------------------------------------------------------------------
# LOGIC TUNG CHIEU — main.py chi goi cast(lm, hand_to_world)
# ----------------------------------------------------------------------
def cast(lm, hand_to_world, hand_id=""):
    """Chieu "Do ngon cai": giu con loc quanh tay khi con do ngon cai."""
    pos = hand_to_world(lm[9].x, lm[9].y)     # tam long ban tay
    t = _tornados.get(hand_id)
    if t is None or t.dead or t.dissipating:
        t = Tornado(pos, size=1.0)
        t.hand_id = hand_id
        _tornados[hand_id] = t
    else:
        t.refresh(pos)
