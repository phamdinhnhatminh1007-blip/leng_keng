"""
IceWater3D.py - hieu ung Frost Bolt cho gesture ngón tro.

Module nay chi chua do hoa Ursina. Webcam, MediaPipe va ML nam trong main.py.
"""

import math
import random

from ursina import Entity, Shader, Vec2, Vec3, color, destroy, time as utime

import Fireball3D as common_fx


ice_shader = Shader(
    name="ice_water_shader",
    language=Shader.GLSL,
    vertex="""
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
""",
    fragment="""
#version 140
uniform vec4 p3d_ColorScale;
in vec2 texcoords;
out vec4 fragColor;

uniform float time;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main() {
    vec2 uv = texcoords;
    float flow = noise(uv * 7.0 + vec2(time * 0.55, -time * 1.4));
    float veins = abs(sin((uv.x + uv.y) * 25.0 + flow * 7.0 + time * 2.2));
    veins = smoothstep(0.78, 0.98, veins);

    vec3 deep = vec3(0.015, 0.18, 0.48);
    vec3 aqua = vec3(0.02, 0.78, 1.10);
    vec3 frost = vec3(0.78, 0.98, 1.25);
    vec3 col = mix(deep, aqua, flow);
    col = mix(col, frost, veins * 0.82);
    col *= p3d_ColorScale.rgb * 1.35;

    fragColor = vec4(col, p3d_ColorScale.a);
}
""",
    default_input=dict(
        texture_scale=Vec2(2.8, 2.8),
        texture_offset=Vec2(0, 0),
        time=0.0,
    ),
)


def init_ice_effects():
    """Khoi tao texture dung chung sau khi Ursina da khoi dong."""
    common_fx.init_effects()


class FrostParticle(Entity):
    """Hat suong lanh nho bay quanh duong dan."""

    def __init__(self, position, size=1.0, velocity=None):
        super().__init__(
            model="quad",
            texture=common_fx.glow_tex,
            billboard=True,
            color=random.choice([
                color.rgba32(205, 250, 255, 235),
                color.rgba32(70, 205, 255, 220),
                color.rgba32(235, 255, 255, 245),
            ]),
            position=position,
            scale=size * random.uniform(0.10, 0.25),
            unlit=True,
            double_sided=True,
        )
        common_fx.enable_additive(self)
        self.velocity = velocity or Vec3(
            random.uniform(-1.0, 1.0),
            random.uniform(-0.4, 1.5),
            random.uniform(0.8, 3.0),
        )
        self.life = random.uniform(0.32, 0.68)
        self.max_life = self.life

    def update(self):
        self.position += self.velocity * utime.dt
        self.velocity *= max(0.0, 1.0 - 1.3 * utime.dt)
        self.life -= utime.dt
        ratio = max(0.0, self.life / self.max_life)
        self.scale *= 1.0 + 0.45 * utime.dt
        self.alpha = ratio * ratio
        if self.life <= 0:
            destroy(self)


class FrostTrail(Entity):
    """Vet nuoc lanh phia sau Frost Bolt."""

    def __init__(self, position, size=1.0):
        super().__init__(
            model="quad",
            texture=common_fx.glow_tex,
            billboard=True,
            color=color.rgba32(35, 175, 255, 145),
            position=position,
            scale=size * random.uniform(0.65, 0.92),
            unlit=True,
            double_sided=True,
        )
        common_fx.enable_additive(self)
        self.life = random.uniform(0.28, 0.44)
        self.max_life = self.life
        self.velocity = Vec3(
            random.uniform(-0.22, 0.22),
            random.uniform(0.12, 0.55),
            random.uniform(0.35, 1.2),
        )

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        ratio = max(0.0, self.life / self.max_life)
        self.scale *= 1.0 + 0.8 * utime.dt
        self.alpha = 0.68 * ratio * ratio
        if self.life <= 0:
            destroy(self)


class IceShard(Entity):
    """Manh bang vang ra khi Frost Bolt no."""

    def __init__(self, position, size=1.0, velocity=None):
        super().__init__(
            model="cube",
            color=random.choice([
                color.rgba32(115, 225, 255, 220),
                color.rgba32(215, 250, 255, 235),
                color.rgba32(35, 145, 255, 210),
            ]),
            position=position,
            scale=(
                size * random.uniform(0.06, 0.13),
                size * random.uniform(0.06, 0.13),
                size * random.uniform(0.35, 0.75),
            ),
            rotation=(
                random.uniform(0, 360),
                random.uniform(0, 360),
                random.uniform(0, 360),
            ),
            unlit=True,
        )
        self.velocity = velocity or Vec3(
            random.uniform(-5.0, 5.0),
            random.uniform(-4.0, 6.0),
            random.uniform(-1.0, 7.0),
        )
        self.spin = Vec3(
            random.uniform(-260, 260),
            random.uniform(-260, 260),
            random.uniform(-260, 260),
        )
        self.life = random.uniform(0.52, 0.95)
        self.max_life = self.life

    def update(self):
        self.position += self.velocity * utime.dt
        self.velocity.y -= 2.2 * utime.dt
        self.rotation += self.spin * utime.dt
        self.life -= utime.dt
        ratio = max(0.0, self.life / self.max_life)
        self.alpha = ratio
        if self.life <= 0:
            destroy(self)


class IceBurst(Entity):
    """Vong nuoc-bang va manh tinh the khi cham muc tieu."""

    def __init__(self, position, size=1.0):
        super().__init__(position=position)
        self.life = 0.58
        self.max_life = self.life
        self.size = size

        self.flash = Entity(
            parent=self,
            model="quad",
            texture=common_fx.glow_tex,
            billboard=True,
            color=color.rgba32(175, 245, 255, 225),
            scale=size * 1.7,
            unlit=True,
            double_sided=True,
        )
        self.ring = Entity(
            parent=self,
            model="quad",
            texture=common_fx.ring_tex,
            billboard=True,
            color=color.rgba32(35, 190, 255, 235),
            scale=size * 1.0,
            unlit=True,
            double_sided=True,
        )
        common_fx.enable_additive(self.flash)
        common_fx.enable_additive(self.ring)

        for _ in range(14):
            FrostParticle(
                position,
                size=size,
                velocity=Vec3(
                    random.uniform(-6.0, 6.0),
                    random.uniform(-5.0, 6.0),
                    random.uniform(-2.0, 8.0),
                ),
            )
        for _ in range(7):
            IceShard(position, size=size)

    def update(self):
        self.life -= utime.dt
        progress = 1.0 - max(0.0, self.life / self.max_life)
        self.flash.scale = self.size * (1.7 + progress * 4.2)
        self.ring.scale = self.size * (1.0 + progress * 8.0)
        self.flash.alpha = max(0.0, 1.0 - progress * 1.9)
        self.ring.alpha = max(0.0, (1.0 - progress) ** 1.2)
        self.rotation_z += 80 * utime.dt
        if self.life <= 0:
            destroy(self)


class IceBolt(Entity):
    """Dan bang/nuoc bay tu dau ngon tro."""

    def __init__(self, position, velocity, power=1.0):
        if common_fx.glow_tex is None:
            init_ice_effects()

        super().__init__(position=position)
        self.velocity = velocity
        self.life = 2.5
        self.timer = 0.0
        self.size = 0.62 + power * 0.62
        self.trail_timer = 0.0
        self.frost_timer = 0.0

        self.core = Entity(
            parent=self,
            model="sphere",
            shader=ice_shader,
            color=color.white,
            scale=self.size,
            unlit=True,
        )
        self.core.set_shader_input(
            "texture_offset",
            Vec2(random.random(), random.random()),
        )

        self.glows = []
        for scale, tint in [
            (1.85, color.rgba32(45, 195, 255, 120)),
            (2.65, color.rgba32(15, 95, 255, 52)),
        ]:
            glow = Entity(
                parent=self,
                model="quad",
                texture=common_fx.glow_tex,
                billboard=True,
                color=tint,
                scale=self.size * scale,
                unlit=True,
                double_sided=True,
            )
            common_fx.enable_additive(glow)
            self.glows.append((glow, scale, tint.a))

        self.crystal_root = Entity(parent=self)
        self.orbit_shards = []
        for index in range(4):
            angle = index * math.tau / 4.0
            shard = Entity(
                parent=self.crystal_root,
                model="cube",
                position=Vec3(
                    math.cos(angle) * self.size * 0.72,
                    math.sin(angle) * self.size * 0.72,
                    0,
                ),
                scale=(self.size * 0.08, self.size * 0.08, self.size * 0.52),
                rotation=(35 + index * 23, index * 90, angle * 57.2958),
                color=color.rgba32(185, 245, 255, 215),
                unlit=True,
            )
            self.orbit_shards.append(shard)

    def update(self):
        self.position += self.velocity * utime.dt
        self.life -= utime.dt
        self.timer += utime.dt

        pulse = 1.0 + 0.08 * math.sin(self.timer * 24.0)
        self.core.scale = self.size * pulse
        self.core.set_shader_input("time", self.timer)
        self.core.rotation_y += 55 * utime.dt
        self.crystal_root.rotation_z += 210 * utime.dt
        self.crystal_root.rotation_y += 95 * utime.dt

        for index, (glow, scale, base_alpha) in enumerate(self.glows):
            glow.scale = self.size * scale * (
                1.0 + 0.06 * math.sin(self.timer * 18 + index)
            )
            glow.alpha = base_alpha * (
                0.88 + 0.12 * math.sin(self.timer * 20 + index)
            )

        self.trail_timer -= utime.dt
        if self.trail_timer <= 0:
            FrostTrail(self.world_position, self.size)
            self.trail_timer = 0.045

        self.frost_timer -= utime.dt
        if self.frost_timer <= 0:
            FrostParticle(
                self.world_position + Vec3(
                    random.uniform(-0.25, 0.25),
                    random.uniform(-0.25, 0.25),
                    random.uniform(-0.15, 0.15),
                ),
                size=self.size,
            )
            self.frost_timer = 0.075

        if self.z < 0.8 or self.life <= 0:
            IceBurst(self.world_position, self.size)
            destroy(self)
