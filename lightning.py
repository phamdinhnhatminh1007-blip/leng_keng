"""
lightning.py — MODULE HIEU UNG SET (Lightning) (file phu)
=========================================================
Hieu ung tia set 3D: tia zigzag phan nhanh, chop giat, tia lua dien, flash.
Tan dung bloom cua camera (bat o main.py) de tia set phat sang.

Cach dung (trong main.py):
    from lightning import cast_lightning

    # trong update(), khi nhan dien chieu "Nam dam":
    cast_lightning(pos)                     # pos = hand_to_world(...)
    # hoac chi dinh huong:
    cast_lightning(pos, direction=Vec3(0, 0, -1))

Yeu cau: camera dung he toa do nhu main.py (camera nhin +z, -z la ve phia
nguoi xem). Khong can init gi them.
"""

import math
import os
import random
import time

from ursina import (
    Entity, camera, color, destroy, time as utime, Vec3, Audio,
)

# --- Am thanh: dat file .wav/.ogg vao thu muc sounds/ ---
_SFX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")


def play_sound(name, volume=0.7):
    """Phat 1 file am thanh trong sounds/. Thieu file thi bo qua (khong crash).

    LUU Y: Ursina Audio tim file theo TEN trong application.asset_folder (quet
    de quy), KHONG nhan duong dan tuyet doi -> chi truyen ten (vd "thunder.wav").
    """
    if not os.path.exists(os.path.join(_SFX_DIR, name)):
        return
    try:
        Audio(name, autoplay=True, loop=False, volume=volume)
    except Exception as e:
        print("Sound error:", e)


# ----------------------------------------------------------------------
# TIEN ICH
# ----------------------------------------------------------------------
def _lerp3(a, b, t):
    return Vec3(a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t)


# ----------------------------------------------------------------------
# TIA LUA DIEN (spark) — dom sang nho bay ra tai diem set danh
# ----------------------------------------------------------------------
class Spark(Entity):
    def __init__(self, position):
        super().__init__(
            model="cube",
            color=random.choice([color.cyan, color.azure, color.white]),
            scale=random.uniform(0.08, 0.22),
            position=position,
            unlit=True,
        )
        self.velocity = Vec3(random.uniform(-6, 6),
                             random.uniform(-6, 6),
                             random.uniform(-4, 4))
        self.life = random.uniform(0.15, 0.4)

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.scale *= 0.85
        self.alpha = max(0.0, self.life * 3)
        if self.life <= 0:
            destroy(self)


# ----------------------------------------------------------------------
# FLASH — quang sang lon loe len roi tat, o diem xuat phat
# ----------------------------------------------------------------------
class Flash(Entity):
    def __init__(self, position, size=3.0):
        super().__init__(
            model="sphere",
            color=color.white,
            scale=size,
            position=position,
            unlit=True,
        )
        self.life = 0.18

    def update(self):
        self.life -= utime.dt
        self.scale *= 1.12
        self.alpha = max(0.0, self.life * 5)
        if self.life <= 0:
            destroy(self)


# ----------------------------------------------------------------------
# DU AM — vet sang mo dan o lai sau khi tia set tat
# ----------------------------------------------------------------------
class Afterglow(Entity):
    """Vet sang tan du: o yen tai vi tri doan set, mo dan cham."""

    def __init__(self, position, rotation, scale):
        super().__init__(model="cube", position=position, rotation=rotation,
                         scale=scale, color=color.azure, unlit=True)
        self.life = random.uniform(0.5, 0.9)
        self.max_life = self.life

    def update(self):
        self.life -= utime.dt
        self.alpha = 0.5 * max(0.0, self.life / self.max_life)
        if self.life <= 0:
            destroy(self)


# ----------------------------------------------------------------------
# TIA SET — duong zigzag gap khuc, chop giat, co phan nhanh
# ----------------------------------------------------------------------
class LightningBolt(Entity):
    """Mot tia set tu 'start' toi 'end', tu chop giat va phan nhanh."""

    def __init__(self, start, end, segments=9, offset=1.6, thickness=0.10,
                 life=0.32, branch=True, depth=0):
        super().__init__(position=Vec3(0, 0, 0))
        self.start = Vec3(*start)
        self.end = Vec3(*end)
        self.segments = segments
        self.offset = offset
        self.thickness = thickness
        self.life = life
        self.branch = branch
        self.depth = depth              # do sau phan nhanh (chan de quy)
        self.parts = []                 # cac doan + nhanh con
        self.flick_timer = 0.0
        self._build()

    # ---- tao duong zigzag bang midpoint displacement ----
    def _jagged_path(self):
        pts = [self.start]
        for i in range(1, self.segments):
            t = i / self.segments
            base = _lerp3(self.start, self.end, t)
            taper = 1 - abs(2 * t - 1)          # lech nhieu nhat o giua
            disp = Vec3(random.uniform(-1, 1),
                        random.uniform(-1, 1),
                        random.uniform(-0.5, 0.5)) * (self.offset * taper)
            pts.append(base + disp)
        pts.append(self.end)
        return pts

    # ---- dung lai toan bo doan set ----
    def _build(self):
        for p in self.parts:
            destroy(p)
        self.parts = []

        pts = self._jagged_path()
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            mid = _lerp3(a, b, 0.5)
            length = (b - a).length()

            # loi trang sang (bloom se lam glow)
            core = Entity(parent=self, model="cube", position=mid,
                          color=color.white, unlit=True)
            core.look_at(b)
            core.scale = Vec3(self.thickness, self.thickness, length)
            self.parts.append(core)

            # vo xanh dien mo hon
            glow = Entity(parent=self, model="cube", position=mid,
                          color=color.azure, unlit=True)
            glow.look_at(b)
            glow.scale = Vec3(self.thickness * 2.6, self.thickness * 2.6, length)
            glow.alpha = 0.4
            self.parts.append(glow)

            # thi thoang phan nhanh (khong de quy sau qua)
            if self.branch and self.depth < 2 and random.random() < 0.25:
                be = b + Vec3(random.uniform(-2.5, 2.5),
                              random.uniform(-2.5, 2.5),
                              random.uniform(-1, 1))
                child = LightningBolt(b, be, segments=4, offset=0.9,
                                      thickness=self.thickness * 0.7,
                                      life=self.life, branch=False,
                                      depth=self.depth + 1)
                child.parent = self
                self.parts.append(child)

    def update(self):
        self.life -= utime.dt
        self.flick_timer += utime.dt

        # chop giat: cu ~0.045s doi hinh dang + nhap nhay do sang
        if self.flick_timer > 0.045:
            self.flick_timer = 0.0
            self._build()
        for p in self.parts:
            if isinstance(p, LightningBolt):
                continue
            base = 1.0 if p.color == color.white else 0.4
            p.alpha = base * random.uniform(0.5, 1.0)

        if self.life <= 0:
            # De lai DU AM: vet sang mo dan doc theo loi tia
            for p in self.parts:
                if isinstance(p, LightningBolt):
                    continue
                if p.color == color.white:      # chi loi trang -> vet sang chinh
                    Afterglow(p.world_position, p.world_rotation, p.scale)
            for p in self.parts:
                destroy(p)
            destroy(self)


# ----------------------------------------------------------------------
# HAM TUNG CHIEU — goi tu main.py
# ----------------------------------------------------------------------
def cast_lightning(target, height=22.0, bolts=2):
    """Set giang tu TREN TROI xuong, danh xuong diem 'target' (mat dat).

    - target : Vec3 (diem set danh - lay tu vi tri ngon tay)
    - height : do cao xuat phat tia set phia tren target
    - bolts  : so tia set chinh giang xuong
    """
    target = Vec3(*target)
    # Diem danh o "mat dat": ngay duoi ngon tay mot chut
    ground = target + Vec3(0, -3.0, 0)

    # 1) Cac tia set chinh: tu tren cao giang thang xuong mat dat
    for i in range(bolts):
        top = ground + Vec3(random.uniform(-1.5, 1.5), height,
                            random.uniform(-1.5, 1.5))
        LightningBolt(top, ground, segments=random.randint(10, 14),
                      offset=random.uniform(1.4, 2.2),
                      thickness=random.uniform(0.10, 0.15),
                      life=random.uniform(0.28, 0.45))

    # 2) Flash sang bung len tai diem danh (mat dat)
    Flash(ground, size=3.5)

    # 3) Vai tia set ngan toe ngang tren mat dat (nhu set lan)
    for _ in range(4):
        out = ground + Vec3(random.uniform(-4, 4),
                            random.uniform(0, 1.5),
                            random.uniform(-2, 2))
        LightningBolt(ground, out, segments=5, offset=0.8,
                      thickness=0.08, life=random.uniform(0.18, 0.3),
                      branch=False)

    # 4) Tia lua dien bat ra tu diem danh
    for _ in range(22):
        Spark(ground)


# ----------------------------------------------------------------------
# LOGIC TUNG CHIEU — main.py chi goi cast(lm, hand_to_world)
# ----------------------------------------------------------------------
LIGHTNING_COOLDOWN = 0.7     # giay giua 2 don set
_last_cast = {}              # cooldown rieng cho tung tay: {hand_id: thoi_diem}


def cast(lm, hand_to_world, hand_id=""):
    """Chieu "Do ngon tro": set giang tu tren troi xuong (cooldown rieng moi tay).

    - lm           : 21 landmarks ban tay (tu MediaPipe)
    - hand_to_world: ham doi toa do tay -> toa do 3D (main.py truyen vao)
    - hand_id      : khoa phan biet tay (vd "Left"/"Right") de cooldown doc lap
    """
    if time.time() - _last_cast.get(hand_id, 0.0) < LIGHTNING_COOLDOWN:
        return
    # Diem set danh = dau ngon tro (landmark 8) cho khop voi ngon tay gio len
    pos = hand_to_world(lm[8].x, lm[8].y)
    cast_lightning(pos)
    play_sound("thunder.wav", volume=0.8)
    _last_cast[hand_id] = time.time()