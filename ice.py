"""
ice.py — MODULE HIEU UNG THUY / BANG (Icebending) (file phu)
============================================================
The Spider-Man (cai + tro + ut) -> mot DAY COT BANG dam tu long dat CHUI LEN quanh
tay: cac khoi bang sac moc hon loan theo ca chieu ngang va chieu sau, rung
man hinh cho uy luc, giu mot lat roi bi keo chim xuong.

Cach dung (trong main.py):
    import ice
    SKILLS = { 5: ice.cast, ... }     # 5 = "The Spider-Man"
"""

import math
import os
import random
import time

from ursina import (
    Entity, camera, color, destroy, time as utime, Vec3, Audio, Mesh,
)

_SFX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")


def _make_ice_crystal():
    """Khoi bang than day 5 mat, chi thu nhon o 28% tren cung."""
    sides = 5
    bottom = []
    shoulder = []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        wobble = 1.0 if i % 2 == 0 else 0.88
        bottom.append(Vec3(math.cos(angle) * 0.58 * wobble, 0,
                           math.sin(angle) * 0.58 * wobble))
        shoulder.append(Vec3(math.cos(angle) * 0.47 * wobble, 0.72,
                             math.sin(angle) * 0.47 * wobble))
    tip = Vec3(0.07, 1.0, -0.04)

    face_colors = [
        color.rgba32(115, 195, 235, 255),
        color.rgba32(170, 228, 250, 255),
        color.rgba32(92, 176, 224, 255),
        color.rgba32(202, 244, 255, 255),
        color.rgba32(128, 207, 242, 255),
    ]
    vertices = []
    colors = []
    for i in range(sides):
        j = (i + 1) % sides
        tint = face_colors[i]
        body = [
            bottom[i], bottom[j], shoulder[j],
            bottom[i], shoulder[j], shoulder[i],
        ]
        vertices.extend(body)
        colors.extend([tint] * len(body))

        point = [shoulder[i], shoulder[j], tip]
        vertices.extend(point)
        colors.extend([color.rgba32(205, 248, 255, 255)] * len(point))

        base = [bottom[j], bottom[i], Vec3(0, 0, 0)]
        vertices.extend(base)
        colors.extend([face_colors[2]] * len(base))

    return Mesh(vertices=vertices, colors=colors, mode="triangle")


_CRYSTAL_MODEL = _make_ice_crystal()


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
    def __init__(self, base_pos, height, thin, delay=0.0, hold=0.9,
                 tilt_x=None, tilt_z=None):
        super().__init__(model=_CRYSTAL_MODEL, color=color.white, unlit=True)
        self.base = Vec3(*base_pos)
        self.H = height
        self.thin = thin
        self.delay = delay
        self.hold = hold
        self.rise_dur = 0.20
        self.sink_dur = random.uniform(0.42, 0.58)
        self.rotation = Vec3(
            tilt_x if tilt_x is not None else random.uniform(-12, 12),
            random.uniform(0, 360),
            tilt_z if tilt_z is not None else random.uniform(-12, 12),
        )
        self.state = "wait"
        self.t = 0.0
        self._set_h(0.001)

    def _set_h(self, h):
        """Khoi crystal co goc tai day, than day moc len va giu chop sac."""
        h = max(0.001, h)
        self.scale = Vec3(self.thin, h, self.thin)
        self.position = Vec3(self.base.x, self.base.y, self.base.z)

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
        elif self.state == "fall":
            # Giu nguyen hinh nhon va keo ca tinh the chim xuong, nhanh dan.
            k = min(1, self.t / self.sink_dur)
            eased = k * k
            self.position = Vec3(
                self.base.x,
                self.base.y - self.H * 1.08 * eased,
                self.base.z,
            )
            self.alpha = max(0.0, 1.0 - max(0.0, k - 0.68) / 0.32)
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

    # Moi diem la mot CHUM chung goc: loi dung + cac mui nghieng toa 360 do.
    # Cac chum lai trai theo x/z quanh tay de vua day dac vua co chieu sau.
    cluster_count = 16
    for _ in range(cluster_count):
        radius = math.sqrt(random.random()) * 8.8
        angle = random.uniform(0, 2 * math.pi)
        depth = math.sin(angle) * radius * 0.62 + random.uniform(-1.4, 2.2)
        cluster_x = center.x + math.cos(angle) * radius * 1.18
        cluster_z = center.z + depth
        local_ground = ground_y + random.uniform(-0.45, 0.38)

        closeness = 1.0 - min(1.0, radius / 8.8)
        core_h = random.uniform(3.6, 5.1) + closeness * random.uniform(2.6, 4.8)
        cluster_delay = random.uniform(0.0, 0.14) + radius * random.uniform(0.006, 0.018)
        hold = random.uniform(0.9, 1.3)

        # Tinh the loi cao, gan thang dung.
        IceSpike(
            Vec3(cluster_x, local_ground, cluster_z),
            core_h,
            random.uniform(0.82, 1.22),
            delay=cluster_delay,
            hold=hold,
            tilt_x=random.uniform(-8, 8),
            tilt_z=random.uniform(-8, 8),
        )

        ray_count = random.randint(4, 7)
        for ray in range(ray_count):
            ray_angle = (
                2 * math.pi * ray / ray_count
                + random.uniform(-0.24, 0.24)
            )
            tilt = random.uniform(22, 43)
            base_offset = random.uniform(0.10, 0.42)
            IceSpike(
                Vec3(
                    cluster_x + math.cos(ray_angle) * base_offset,
                    local_ground + random.uniform(-0.10, 0.12),
                    cluster_z + math.sin(ray_angle) * base_offset,
                ),
                core_h * random.uniform(0.55, 0.88),
                random.uniform(0.48, 0.92),
                delay=cluster_delay + random.uniform(0.0, 0.035),
                hold=hold + random.uniform(-0.08, 0.10),
                tilt_x=math.cos(ray_angle) * tilt,
                tilt_z=math.sin(ray_angle) * tilt,
            )


# ----------------------------------------------------------------------
# LOGIC TUNG CHIEU — main.py chi goi cast(lm, hand_to_world)
# ----------------------------------------------------------------------
ICE_COOLDOWN = 1.3           # giay giua 2 lan dung bang
_last_cast = {}


def cast(lm, hand_to_world, hand_id=""):
    """The Spider-Man: day cot bang nhon chui len (cooldown rieng moi tay)."""
    if time.time() - _last_cast.get(hand_id, 0.0) < ICE_COOLDOWN:
        return
    center = hand_to_world(lm[9].x, lm[9].y)
    cast_ice(center)
    play_sound("ice.wav", volume=0.8)
    _last_cast[hand_id] = time.time()
