import argparse
import bpy
import colorsys
import json
import math
import os
import shutil
import sys
import time
import zlib
from functools import cache
from mathutils import Matrix, Vector
from pathlib import Path

import numpy as np  # bundled with Blender
import OpenImageIO as oiio  # bundled with Blender; splits and reads the EXRs


def script_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="//dataset")
    parser.add_argument("--frames", type=int, default=150)  # frames to save per scene
    parser.add_argument("--cycles-device", default="OPTIX")  # Cycles reads it too
    args, _unknown = parser.parse_known_args(argv)  # skips other args after --
    return args


ARGS = script_args()
OUT_DIR = ARGS.out_dir.rstrip("/\\")


scene_name = Path(bpy.data.filepath).stem
ops = bpy.ops
scene = bpy.context.scene
cam: bpy.types.Object = scene.camera
context = bpy.context

T0 = time.perf_counter()


def log(*args):
    # flush=True so lines show up immediately, even if Blender is busy later
    print(f"[{time.perf_counter() - T0:8.2f}s]", *args, flush=True)


log(f"script start: scene={scene_name!r} camera={cam.name if cam else None}")

MASK64 = (1 << 64) - 1

# Rendering (roadmap 1.3-1.6)
RESOLUTION = 512  # square; training uses 128x128 crops
SAMPLES_NOISY = (16, 8, 4, 1)  # 16 first: it gives the aux passes and the black check
SAMPLES_GT = 2048
MIN_MEAN_LUMINANCE = 0.005  # frames darker than this (scene-linear) are rejected
MAX_ATTEMPTS_PER_FRAME = 3  # a scene gives up after frames * this many attempts
AUX_PASSES = {  # pass in the 16 spp render -> (file, bit depth)
    "Denoising Albedo": ("albedo.exr", oiio.HALF),
    "Denoising Normal": ("normal.exr", oiio.HALF),
    "Depth": ("depth.exr", oiio.FLOAT),  # 32-bit: background depth overflows half
}

# Camera scoring thresholds (meters where noted)
MIN_COVERAGE = 0.95  # fraction of grid rays that must hit something (no holes to outside)
NEAR_DIST = 1.0  # m, a grid hit closer than this counts as "near"
MAX_NEAR_FRACTION = 0.5  # reject if more of the view than this is near (a wall close-up)
MIN_VARIETY = 4  # distinct objects hit
MIN_CENTER_DIST = 1.5  # center ray closer than this = up against something
INSIDE_DIST = 0.05  # a hit this close counts as "tiny"
MAX_AIM_DIST = 50.0  # loose limit: center hit to scene center
INSIDE_AXIS_DIST = 0.2  # roadmap: all 6 axis rays hit within this = inside an object

# Randomization ranges (roadmap 1.7). Pairs are (low, high).
BOUNDS_NAME = "CamBounds"  # Empty (Cube display) covering the inside of the room
FALLBACK_HALF_SIZE = (
    3.0  # m, room box around the original camera if there's no CamBounds
)
WALL_MARGIN = 0.3  # m, camera and extra lights stay this far inside the room bounds
CAM_HEIGHT = (0.8, 2.0)  # m above the floor
FOCAL_LENGTH = (18.0, 50.0)  # mm
LOOK_AT_OBJECT_PROB = 0.5  # else look at a random point in the room
MAX_TARGET_SIZE = 3.0  # m, bigger meshes (walls, floor, ceiling, mouldings) aren't aimed at
TARGET_MARGIN = 1.0  # m, "random point in the room" aims stay this far from its sides
DOF_PROB = 0.2
DOF_FSTOP = (1.4, 8.0)  # log-uniform

SUN_ELEVATION = (5.0, 80.0)  # degrees
SUN_AZIMUTH = (0.0, 360.0)  # degrees
SUN_STRENGTH = (1.0, 10.0)  # log-uniform
SUN_ANGLE = (0.5, 5.0)  # degrees
SUN_TEMPERATURE = (3000.0, 7000.0)  # K
WORLD_STRENGTH = (0.3, 3.0)  # log-uniform
WORLD_ROTATION = (0.0, 360.0)  # degrees
EXTRA_LIGHT_COUNT = (0, 3)
EXTRA_LIGHT_POWER = (50.0, 1000.0)  # W, log-uniform
EXTRA_LIGHT_AREA_SIZE = (0.25, 1.0)  # m (not in the roadmap)

MAT_TWEAK_PROB = 0.3  # these get a hue shift and a new roughness
MAT_HUE_SHIFT = 0.1  # +- on the 0-1 hue wheel
MAT_ROUGHNESS = (0.05, 0.9)
MAT_METAL_PROB = 0.1
MAT_GLASS_PROB = 0.05

RAND_PREFIX = "Rand_"  # name prefix for objects and nodes this script creates


class PCG32:
    """PCG-XSH-RR 32-bit generator: fast, small state, passes BigCrush/PractRand,
    and fully deterministic from (seed, stream) so dataset scenes are reproducible."""

    MULT = 6364136223846793005

    def __init__(self, seed, stream=0):
        self.state = 0
        self.inc = ((stream << 1) | 1) & MASK64
        self.next_u32()
        self.state = (self.state + seed) & MASK64
        self.next_u32()

    def next_u32(self):
        old = self.state
        self.state = (old * self.MULT + self.inc) & MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = old >> 59
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def random(self):
        """Float in [0, 1) with 53 bits of precision."""
        a = self.next_u32() >> 5
        b = self.next_u32() >> 6
        return (a * 67108864 + b) / 9007199254740992

    def uniform(self, low, high):
        return low + (high - low) * self.random()

    def randint(self, low, high):
        """Unbiased integer in [low, high], inclusive."""
        span = high - low + 1
        threshold = (1 << 32) % span
        while True:
            r = self.next_u32()
            if r >= threshold:
                return low + r % span

    def choice(self, seq):
        return seq[self.randint(0, len(seq) - 1)]

    def log_uniform(self, low, high):
        """Spread evenly across orders of magnitude (roadmap: use for strengths/powers)."""
        return math.exp(self.uniform(math.log(low), math.log(high)))

    def chance(self, p):
        """True with probability p."""
        return self.random() < p


def lerp(a, b, t):
    """Point a fraction t of the way from a to b (t=0 gives a, t=1 gives b)."""
    return tuple(ai + t * (bi - ai) for ai, bi in zip(a, b))


@cache
def scene_center():
    """Average of every mesh's world-space bounding-box center."""
    log("scene_center: computing (runs once)")
    centers = []
    for obj in scene.objects:
        if obj.type == "MESH":
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            centers.append(sum(corners, Vector()) / 8)
    if not centers:
        return Vector()
    result = sum(centers, Vector()) / len(centers)
    log(f"scene_center: {len(centers)} meshes, center={tuple(result)}")
    return result


def is_current_cam_pos_good(cam: bpy.types.Object):
    log("  check: getting depsgraph")
    deps = context.evaluated_depsgraph_get()
    frame_cam = cam.data.view_frame(scene=scene)

    frame_world = [(cam.matrix_world @ pos.to_4d()).to_3d() for pos in frame_cam]
    p0, p1, p2, p3 = frame_world[0], frame_world[1], frame_world[2], frame_world[3]

    grid = []
    SPREAD = 8
    GRID_SIZE = SPREAD * SPREAD
    for i in range(SPREAD):
        vt = i * 1 / (SPREAD - 1)
        right = lerp(p0, p1, vt)  # Right border
        left = lerp(p3, p2, vt)  # Left border

        for j in range(SPREAD):
            ht = j * 1 / (SPREAD - 1)
            p = lerp(right, left, ht)
            grid.append(p)
            # Visual Debug
            # scene.cursor.location = p
            # bpy.ops.mesh.primitive_vert_add()

    cam_origin = (
        cam.matrix_world.translation
    )  # cam.location can differ when mods, parents, etc.

    log(f"  check: casting {len(grid)} rays")
    hits = []  # (distance, object name) for each grid ray that hit something
    for pos in grid:
        direction = (Vector(pos) - cam_origin).normalized()
        hit, location, _normal, _index, obj, _matrix = scene.ray_cast(
            deps, cam_origin, direction
        )
        if hit:
            hits.append(((location - cam_origin).length, obj.name))

    # Coverage: too much empty sky/background below this
    coverage = len(hits) / GRID_SIZE
    if coverage < MIN_COVERAGE:
        log(f"  reject: coverage {coverage:.2f} < {MIN_COVERAGE}")
        return False

    # Variety: one wall or one desk filling the frame
    variety = len({name for _, name in hits})
    if variety < MIN_VARIETY:
        log(f"  reject: variety {variety} < {MIN_VARIETY}")
        return False

    # Close-up: most of the view is a wall or object right in front of the camera
    near = sum(1 for dist, _ in hits if dist < NEAR_DIST) / GRID_SIZE
    if near > MAX_NEAR_FRACTION:
        log(f"  reject: {near:.0%} of the view is closer than {NEAR_DIST} m")
        return False

    # Inside an object: most distances are tiny
    tiny = sum(1 for dist, _ in hits if dist < INSIDE_DIST)
    if tiny > len(hits) / 2:
        log(f"  reject: inside object ({tiny}/{len(hits)} tiny hits)")
        return False

    # Inside an object (roadmap 1.7): rays along all 6 axes hit within INSIDE_AXIS_DIST
    axes = [
        Vector(v)
        for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    ]
    close = 0
    for axis in axes:
        hit, location, *_ = scene.ray_cast(deps, cam_origin, axis)
        if hit and (location - cam_origin).length < INSIDE_AXIS_DIST:
            close += 1
    if close == len(axes):
        log(
            f"  reject: inside an object (all 6 axis rays hit within {INSIDE_AXIS_DIST} m)"
        )
        return False

    # Center ray (middle of the camera frame)
    center = (p0 + p2) / 2
    hit, location, *_ = scene.ray_cast(
        deps, cam_origin, (center - cam_origin).normalized()
    )
    if not hit:
        log("  reject: center ray hit nothing")
        return False

    # Too close: up against something
    center_dist = (location - cam_origin).length
    if center_dist < MIN_CENTER_DIST:
        log(f"  reject: center ray too close ({center_dist:.2f} m)")
        return False

    # Aim: loose limit, off-center views are still useful
    aim = (location - scene_center()).length
    if aim > MAX_AIM_DIST:
        log(f"  reject: aim {aim:.1f} m > {MAX_AIM_DIST}")
        return False

    log(
        f"  accept: coverage {coverage:.2f}, variety {variety}, center {center_dist:.2f} m"
    )
    return True


def clear_animation(*ids):
    """Remove keyframes and drivers. Rendering re-evaluates animation, which would undo
    the random values (Classroom's renderCam has keyframes on its location)."""
    for id_data in ids:
        if id_data is not None and id_data.animation_data is not None:
            id_data.animation_data_clear()
            log(f"cleared animation on {id_data.name}")


clear_animation(cam, cam.data)
BASE_MATRIX = (
    cam.matrix_world.copy()
)  # original camera pose, restored if no good camera


def cast_ray(origin, direction):
    """Ray-cast the evaluated scene. Returns (hit location, object) or (None, None)."""
    deps = context.evaluated_depsgraph_get()
    hit, location, _normal, _index, obj, _matrix = scene.ray_cast(
        deps, origin, direction
    )
    return (location, obj) if hit else (None, None)


def bbox_of(points):
    """Axis-aligned (min, max) corners around a list of points."""
    lo = Vector([min(p[k] for p in points) for k in range(3)])
    hi = Vector([max(p[k] for p in points) for k in range(3)])
    return lo, hi


def inset(a, b, margin):
    """Range (a, b) shrunk by margin on both sides, or its middle if it's too narrow."""
    if b - a > 2 * margin:
        return a + margin, b - margin
    return (a + b) / 2, (a + b) / 2


def random_point(rng: PCG32, lo, hi):
    return Vector([rng.uniform(lo[k], hi[k]) for k in range(3)])


@cache
def room_bounds():
    """(min, max) corners of the room the camera and extra lights stay in.
    Uses an Empty named CamBounds if there is one, else a box around the original camera.
    """
    bounds = scene.objects.get(BOUNDS_NAME)
    if bounds is not None and bounds.type == "EMPTY":
        # An Empty's bound_box is all zeros, so build the cube from its display size
        s = bounds.empty_display_size
        corners = [
            bounds.matrix_world @ Vector((x * s, y * s, z * s))
            for x in (-1, 1)
            for y in (-1, 1)
            for z in (-1, 1)
        ]
        lo, hi = bbox_of(corners)
        log(f"room_bounds: from {BOUNDS_NAME}, {tuple(lo)} to {tuple(hi)}")
        return lo, hi

    if bounds is not None:
        log(
            f"room_bounds: {BOUNDS_NAME} must be an Empty (a mesh blocks the ray casts), ignoring it"
        )

    origin = BASE_MATRIX.translation
    floor_hit, _ = cast_ray(origin, Vector((0, 0, -1)))
    floor = floor_hit.z if floor_hit is not None else origin.z - 1.5
    h = FALLBACK_HALF_SIZE
    lo = Vector((origin.x - h, origin.y - h, floor))
    hi = Vector((origin.x + h, origin.y + h, floor + 3.0))
    log(
        f"room_bounds: no {BOUNDS_NAME} Empty, using +-{h} m around the original camera"
    )
    return lo, hi


@cache
def target_boxes():
    """World bounding boxes of visible, renderable meshes centered inside the room.
    Room-sized meshes (walls, floor, ceiling, long mouldings) are left out."""
    lo, hi = room_bounds()
    boxes = []
    too_big = 0
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render or not obj.visible_get():
            continue
        box = bbox_of([obj.matrix_world @ Vector(c) for c in obj.bound_box])
        center = (box[0] + box[1]) / 2
        if not all(lo[k] <= center[k] <= hi[k] for k in range(3)):
            continue
        if max(box[1][k] - box[0][k] for k in range(3)) > MAX_TARGET_SIZE:
            too_big += 1
            continue
        boxes.append(box)
    log(f"target_boxes: {len(boxes)} objects to aim at, {too_big} room-sized ones left out")
    return boxes


# Randomize camera position, look-at, focal length and DOF (roadmap 1.7)
def randomize_camera(cam: bpy.types.Object, rng: PCG32, default=True):
    """One random camera. Returns the chosen values for meta.json."""
    if default is True:
        return {}

    lo, hi = room_bounds()

    # Position: inside the room, WALL_MARGIN from the walls, CAM_HEIGHT above the floor
    position = Vector(
        (
            rng.uniform(*inset(lo.x, hi.x, WALL_MARGIN)),
            rng.uniform(*inset(lo.y, hi.y, WALL_MARGIN)),
            min(lo.z + rng.uniform(*CAM_HEIGHT), hi.z - WALL_MARGIN),
        )
    )

    # Look-at: a random point on a visible object, or anywhere in the room
    boxes = target_boxes()
    if boxes and rng.chance(LOOK_AT_OBJECT_PROB):
        target = random_point(rng, *rng.choice(boxes))
        target_kind = "object"
    else:  # away from the walls, floor and ceiling, so this doesn't aim at them either
        inner = [inset(lo[k], hi[k], TARGET_MARGIN) for k in range(3)]
        target = random_point(rng, [a for a, _ in inner], [b for _, b in inner])
        target_kind = "room"

    direction = target - position
    if direction.length < 1e-6:
        direction = Vector((0, 1, 0))
    rotation = direction.to_track_quat("-Z", "Y")  # cameras look down -Z with Y up
    # Setting matrix_world works for any rotation mode and for parented cameras
    cam.matrix_world = Matrix.LocRotScale(position, rotation, BASE_MATRIX.to_scale())

    cam.data.lens = rng.uniform(*FOCAL_LENGTH)

    # after you move or rotate the camera, call this to update transforms
    context.view_layer.update()

    # Depth of field on some frames, focused on what the center ray hits
    dof = cam.data.dof
    dof.use_dof = rng.chance(DOF_PROB)
    fstop = focus = None
    if dof.use_dof:
        hit, _ = cast_ray(position, rotation @ Vector((0, 0, -1)))
        focus = (hit - position).length if hit is not None else 10.0
        fstop = rng.log_uniform(*DOF_FSTOP)
        dof.focus_object = None
        dof.focus_distance = focus
        dof.aperture_fstop = fstop

    return {
        "location": list(position),
        "target": list(target),
        "target_kind": target_kind,
        "lens_mm": cam.data.lens,
        "dof": dof.use_dof,
        "fstop": fstop,
        "focus_distance": focus,
    }


def find_good_camera(cam: bpy.types.Object, rng: PCG32, max_tries=50):
    """Randomize until the camera passes the checks. Returns (found, tries, camera values).
    If nothing passes, restores the starting pose and found is False."""
    for tries in range(1, max_tries + 1):
        log(f"camera try {tries}/{max_tries}")
        values = randomize_camera(cam, rng, False)
        if is_current_cam_pos_good(cam):
            return True, tries, values

    cam.matrix_world = BASE_MATRIX
    context.view_layer.update()
    return False, max_tries, {}


def get_sun():
    """First editable sun light in the scene, or a new one."""
    for obj in scene.objects:
        if (
            obj.type == "LIGHT"
            and obj.data.type == "SUN"
            and obj.library is None
            and obj.data.library is None
        ):
            clear_animation(obj, obj.data)
            return obj
    data = bpy.data.lights.new(RAND_PREFIX + "Sun", "SUN")
    obj = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(obj)
    log("sun: none found, created one")
    return obj


def randomize_sun(rng: PCG32):
    sun = get_sun()
    elevation = rng.uniform(*SUN_ELEVATION)
    azimuth = rng.uniform(*SUN_AZIMUTH)
    # A sun shines down -Z; tilting it by (90 - elevation) on X raises it to that elevation
    sun.rotation_mode = "XYZ"
    sun.rotation_euler = (math.radians(90 - elevation), 0.0, math.radians(azimuth))
    sun.data.energy = rng.log_uniform(*SUN_STRENGTH)
    sun.data.angle = math.radians(rng.uniform(*SUN_ANGLE))
    sun.data.color = (1.0, 1.0, 1.0)
    sun.data.use_temperature = True
    sun.data.temperature = rng.uniform(*SUN_TEMPERATURE)
    return {
        "name": sun.name,
        "elevation_deg": elevation,
        "azimuth_deg": azimuth,
        "strength": sun.data.energy,
        "angle_deg": math.degrees(sun.data.angle),
        "temperature_k": sun.data.temperature,
    }


def world_mapping(world, env):
    """Mapping node feeding the environment texture, inserted on first use."""
    tree = world.node_tree
    vector = env.inputs["Vector"]
    if vector.is_linked and vector.links[0].from_node.type == "MAPPING":
        return vector.links[0].from_node
    if vector.is_linked:
        source = vector.links[0].from_socket
    else:
        source = tree.nodes.new("ShaderNodeTexCoord").outputs["Generated"]
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.name = RAND_PREFIX + "Mapping"
    tree.links.new(source, mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], vector)
    return mapping


def randomize_world(rng: PCG32):
    world = scene.world
    if world is None or world.library is not None or world.node_tree is None:
        log("world: missing, linked or without nodes, skipped")
        return {}
    clear_animation(world, world.node_tree)
    nodes = world.node_tree.nodes
    background = next((n for n in nodes if n.type == "BACKGROUND"), None)
    env = next((n for n in nodes if n.type == "TEX_ENVIRONMENT"), None)
    values = {}
    if background is not None:
        values["strength"] = rng.log_uniform(*WORLD_STRENGTH)
        background.inputs["Strength"].default_value = values["strength"]
    if env is not None:  # rotation only means something for an HDRI
        values["rotation_deg"] = rng.uniform(*WORLD_ROTATION)
        mapping = world_mapping(world, env)
        mapping.inputs["Rotation"].default_value[2] = math.radians(
            values["rotation_deg"]
        )
    return values


def remove_extra_lights():
    """Delete the extra lights made for the previous frame."""
    for obj in list(scene.objects):
        if obj.type == "LIGHT" and obj.name.startswith(RAND_PREFIX + "Light"):
            data = obj.data
            bpy.data.objects.remove(obj)
            if data.users == 0:
                bpy.data.lights.remove(data)


def randomize_extra_lights(rng: PCG32):
    remove_extra_lights()
    lo, hi = room_bounds()
    lights = []
    for n in range(rng.randint(*EXTRA_LIGHT_COUNT)):
        kind = rng.choice(("POINT", "AREA"))
        data = bpy.data.lights.new(f"{RAND_PREFIX}Light{n}", kind)
        data.energy = rng.log_uniform(*EXTRA_LIGHT_POWER)
        if kind == "AREA":
            data.size = rng.uniform(*EXTRA_LIGHT_AREA_SIZE)
        obj = bpy.data.objects.new(data.name, data)
        obj.location = Vector(
            (
                rng.uniform(*inset(lo.x, hi.x, WALL_MARGIN)),
                rng.uniform(*inset(lo.y, hi.y, WALL_MARGIN)),
                rng.uniform(lo.z + 0.5, max(lo.z + 0.5, hi.z - WALL_MARGIN)),
            )
        )
        scene.collection.objects.link(obj)
        lights.append(
            {"type": kind, "power_w": data.energy, "location": list(obj.location)}
        )
    return lights


MAT_SOCKETS = {
    "base": "Base Color",
    "rough": "Roughness",
    "metal": "Metallic",
    "trans": "Transmission Weight",
    "ior": "IOR",
}


def principled_nodes(mat):
    return [n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"]


def set_input(node, name, value):
    """Set an unconnected input. Connected ones are driven by textures, so leave them."""
    socket = node.inputs.get(name)
    if socket is not None and not socket.is_linked:
        socket.default_value = value


def hue_node(mat, node):
    """Hue/Saturation node in front of a textured Base Color, inserted on first use.
    Returns None if the Base Color is a plain color."""
    base = node.inputs["Base Color"]
    if not base.is_linked:
        return None
    source = base.links[0].from_node
    if source.type == "HUE_SAT" and source.name.startswith(RAND_PREFIX):
        return source
    tree = mat.node_tree
    source_socket = base.links[0].from_socket
    hsv = tree.nodes.new("ShaderNodeHueSaturation")
    hsv.name = RAND_PREFIX + "HSV"
    tree.links.new(source_socket, hsv.inputs["Color"])
    tree.links.new(hsv.outputs["Color"], base)
    return hsv


@cache
def prepare_materials():
    """Editable materials that use a Principled BSDF. Saves their original values once,
    in a custom property, so re-running the script in the same session never drifts."""
    mats = []
    for mat in bpy.data.materials:
        if not mat.users or mat.library is not None or mat.node_tree is None:
            continue
        nodes = principled_nodes(mat)
        if not nodes:
            continue
        clear_animation(mat, mat.node_tree)
        if "rand_orig" not in mat:
            orig = {}
            for node in nodes:
                orig[node.name] = {
                    key: (
                        list(node.inputs[name].default_value)
                        if key == "base"
                        else node.inputs[name].default_value
                    )
                    for key, name in MAT_SOCKETS.items()
                    if name in node.inputs
                }
            mat["rand_orig"] = orig
        for node in nodes:
            hue_node(mat, node)
        mats.append(mat)
    log(f"materials: {len(mats)} editable")
    return mats


def randomize_materials(rng: PCG32):
    """Hue shift + roughness on ~30%, metal on ~10%, glass on ~5% of materials."""
    changed = {}
    for mat in prepare_materials():
        tweak = rng.chance(MAT_TWEAK_PROB)
        hue_shift = rng.uniform(-MAT_HUE_SHIFT, MAT_HUE_SHIFT) if tweak else 0.0
        roughness = rng.uniform(*MAT_ROUGHNESS) if tweak else None
        kind = rng.random()
        glass = kind < MAT_GLASS_PROB
        metal = not glass and kind < MAT_GLASS_PROB + MAT_METAL_PROB

        orig = mat["rand_orig"]
        for node in principled_nodes(mat):
            if node.name not in orig:
                continue
            o = orig[node.name]
            # Start from the original values every frame so changes never pile up
            for key, name in MAT_SOCKETS.items():
                if key in o:
                    set_input(node, name, tuple(o[key]) if key == "base" else o[key])

            hsv = hue_node(mat, node)
            if hsv is not None:
                hsv.inputs["Hue"].default_value = 0.5 + hue_shift  # 0.5 = unchanged
            elif tweak and "base" in o:
                r, g, b, a = o["base"]
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                rgb = colorsys.hsv_to_rgb((h + hue_shift) % 1.0, s, v)
                set_input(node, "Base Color", (*rgb, a))

            if tweak:
                set_input(node, "Roughness", roughness)
            if metal:
                set_input(node, "Metallic", 1.0)
            if glass:
                set_input(node, "Metallic", 0.0)
                set_input(node, "Transmission Weight", 1.0)
                set_input(node, "Roughness", 0.05)
                set_input(node, "IOR", 1.45)

        if tweak or metal or glass:
            changed[mat.name] = {
                "hue_shift": hue_shift,
                "roughness": roughness,
                "metal": metal,
                "glass": glass,
            }
    return changed


# Randomizes lights, world and materials for one frame (roadmap 1.7)
def randomize_frames(rng: PCG32):
    """Returns every randomized value, for meta.json."""
    return {
        "sun": randomize_sun(rng),
        "world": randomize_world(rng),
        "extra_lights": randomize_extra_lights(rng),
        "materials": randomize_materials(rng),
    }


def use_device(device):
    """Render on `device` (OPTIX, CUDA, HIP, ONEAPI, METAL or CPU; add +CPU to also use
    the CPU). Raises instead of silently falling back to the CPU."""
    gpu, _, extra = device.upper().partition("+")
    if gpu == "CPU":
        scene.cycles.device = "CPU"
        log("render device: CPU")
        return
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = gpu
    except TypeError:
        raise RuntimeError(f"this Blender has no {gpu} support, use --cycles-device CUDA or CPU")
    prefs.refresh_devices()
    used = []
    for d in prefs.devices:
        d.use = d.type == gpu or (extra == "CPU" and d.type == "CPU")
        if d.use:
            used.append(f"{d.name} ({d.type})")
    if not any(d.use and d.type == gpu for d in prefs.devices):
        raise RuntimeError(f"no {gpu} device found, use --cycles-device CUDA or CPU")
    scene.cycles.device = "GPU"
    log(f"render device: {', '.join(used)}")


def setup_render():
    """Settings shared by all 5 renders of every frame (roadmap 1.3/1.4)."""
    render = scene.render
    render.engine = "CYCLES"
    render.resolution_x = RESOLUTION
    render.resolution_y = RESOLUTION
    render.resolution_percentage = 100
    render.film_transparent = False
    render.use_persistent_data = True  # keeps the BVH between the 5 renders of a frame
    render.threads_mode = "AUTO"

    cycles = scene.cycles
    cycles.use_denoising = False
    cycles.use_adaptive_sampling = False  # noise must be the same everywhere
    cycles.use_animated_seed = False

    for view_layer in scene.view_layers:
        view_layer.use_pass_combined = True
        view_layer.use_pass_z = True  # depth.exr
        view_layer.cycles.denoising_store_passes = True  # albedo.exr, normal.exr

    use_device(ARGS.cycles_device)


def render_samples(name, samples, seed):
    """Render the current frame. Returns the render time in seconds."""
    scene.cycles.samples = samples
    scene.cycles.seed = seed
    log(f"render START {name} ({samples} spp, seed {seed})")
    t = time.perf_counter()
    ops.render.render()
    seconds = time.perf_counter() - t
    log(f"render DONE  {name} in {seconds:.1f}s")
    return seconds


def save_result(path, multilayer=False, depth="16"):
    """Save the last render as a lossless EXR. depth '16' = half, '32' = full float."""
    image = scene.render.image_settings
    if multilayer:  # every pass, each as its own part of the file
        image.media_type = "MULTI_LAYER_IMAGE"
        image.file_format = "OPEN_EXR_MULTILAYER"
    else:  # just the RGB image
        image.media_type = "IMAGE"
        image.file_format = "OPEN_EXR"
        image.color_mode = "RGB"
    image.color_depth = depth
    image.exr_codec = "ZIP"  # lossless; never DWAA/DWAB
    bpy.data.images["Render Result"].save_render(filepath=path, scene=scene)


def write_exr(path, pixels, channel_names, dtype):
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    # Picking channels with a list (pixels[..., [i, j, k]]) can give an array whose
    # memory isn't row by row; OpenImageIO refuses those, so make a normal copy
    pixels = np.ascontiguousarray(pixels)
    height, width, channels = pixels.shape
    spec = oiio.ImageSpec(width, height, channels, dtype)
    spec.channelnames = tuple(channel_names)
    spec.attribute("compression", "zip")
    out = oiio.ImageOutput.create(path)
    if out is None or not out.open(path, spec):
        raise RuntimeError(f"can't write {path}: {oiio.geterror()}")
    wrote = out.write_image(pixels)
    error = out.geterror()
    out.close()
    if not wrote:  # otherwise a failed write leaves a file with only a header
        raise RuntimeError(f"can't write {path}: {error}")


def split_aux(aux_path, folder):
    """Copy albedo, normal and depth out of the multilayer 16 spp file into their own
    EXRs (roadmap 1.5), then delete it."""
    inp = oiio.ImageInput.open(aux_path)
    if inp is None:
        raise RuntimeError(f"can't open {aux_path}: {oiio.geterror()}")
    found = {}
    sub = 0
    # Blender writes either one part per pass or one part holding every pass, so check
    # every channel. Names are "<view layer>.<pass>.<channel>", e.g. "ViewLayer.Depth.Z".
    while inp.seek_subimage(sub, 0):
        spec = inp.spec()
        wanted = {}  # pass -> (view layer, [(channel, index)])
        for index, full_name in enumerate(spec.channelnames):
            layer_pass, _, channel = full_name.rpartition(".")
            layer, _, pass_name = layer_pass.rpartition(".")
            if pass_name not in AUX_PASSES or pass_name in found:
                continue
            first_layer, channels = wanted.setdefault(pass_name, (layer, []))
            if layer == first_layer:  # one view layer only, if there are several
                channels.append((channel, index))
        if wanted:
            pixels = inp.read_image(sub, 0, 0, spec.nchannels, oiio.FLOAT)
            if pixels.ndim == 2:
                pixels = pixels[..., None]
            for pass_name, (_, channels) in wanted.items():
                channels.sort(key=lambda c: "RGBAXYZW".find(c[0]))
                found[pass_name] = (
                    pixels[..., [index for _, index in channels]],
                    [channel for channel, _ in channels],
                )
        sub += 1
    inp.close()
    for pass_name, (file_name, dtype) in AUX_PASSES.items():
        if pass_name not in found:
            raise RuntimeError(f"pass {pass_name!r} is missing from {aux_path}")
        pixels, channel_names = found[pass_name]
        write_exr(os.path.join(folder, file_name), pixels, channel_names, dtype)
    os.remove(aux_path)


def mean_luminance(path):
    inp = oiio.ImageInput.open(path)
    px = inp.read_image(0, 0, 0, 3, oiio.FLOAT)
    inp.close()
    return float((0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]).mean())


def render_frame(folder, seed_base):
    """Render one frame into `folder` (roadmap 1.5/1.6).
    Returns (kept, info); kept is False if the frame is almost black."""
    info = {"resolution": RESOLUTION, "seeds": {}, "render_seconds": {}}
    for k, spp in enumerate(SAMPLES_NOISY):
        name = f"noisy_{spp}spp"
        seed = seed_base + 1 + k  # noisy seeds differ from the gt seed, so it can't leak
        info["seeds"][name] = seed
        info["render_seconds"][name] = render_samples(name, spp, seed)
        path = os.path.join(folder, name + ".exr")
        save_result(path)
        if k == 0:  # the 16 spp render: check it, then save the aux passes from it
            info["mean_luminance"] = mean_luminance(path)
            if info["mean_luminance"] < MIN_MEAN_LUMINANCE:
                return False, info
            aux_path = os.path.join(folder, "aux_tmp.exr")
            save_result(aux_path, multilayer=True, depth="32")
            split_aux(aux_path, folder)
    info["seeds"]["gt"] = seed_base
    info["render_seconds"]["gt"] = render_samples("gt", SAMPLES_GT, seed_base)
    save_result(os.path.join(folder, "gt.exr"), depth="32")
    return True, info


def read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    """Write through a temp file, so a crash never leaves a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


if not bpy.data.filepath:
    raise RuntimeError("save the .blend first: the output folder is named after it")

SCENE_DIR = os.path.normpath(os.path.join(bpy.path.abspath(OUT_DIR), scene_name))
REJECTED_PATH = os.path.join(SCENE_DIR, "rejected.json")  # skipped frames, kept across runs
FRAMES = ARGS.frames
os.makedirs(SCENE_DIR, exist_ok=True)
log(f"output: {SCENE_DIR}, {FRAMES} frames wanted")
setup_render()

rejected = read_json(REJECTED_PATH, {})
saved = 0
total_tries = 0
attempts = 0
for attempt in range(FRAMES * MAX_ATTEMPTS_PER_FRAME):
    if saved >= FRAMES:
        break
    attempts = attempt + 1
    folder = os.path.join(SCENE_DIR, f"frame_{attempt:04d}")
    # Resume: meta.json is written last, so a frame that has it is complete
    if os.path.exists(os.path.join(folder, "meta.json")):
        saved += 1
        continue
    if str(attempt) in rejected:
        continue
    log(f"--- frame {attempt} ({saved}/{FRAMES} saved) ---")
    # Same scene + frame index always gives the same random choices (roadmap 1.6)
    frame_seed = zlib.crc32(f"{scene_name}/{attempt}".encode())
    rng = PCG32(seed=frame_seed)
    found, tries, camera_values = find_good_camera(cam, rng)
    total_tries += tries
    if not found:
        log(f"frame {attempt}: no good camera after {tries} tries, skipping")
        rejected[str(attempt)] = "no good camera"
        write_json(REJECTED_PATH, rejected)
        continue
    frame_meta = {
        "scene": scene_name,
        "frame": attempt,
        "frame_seed": frame_seed,
        "camera_tries": tries,
        "camera": camera_values,
        **randomize_frames(rng),
    }
    log(
        f"frame {attempt}: {len(frame_meta['extra_lights'])} extra lights, "
        f"{len(frame_meta['materials'])} materials changed"
    )

    os.makedirs(folder, exist_ok=True)
    # Cycles seeds must stay below 2**31; leave room for the +1..+4 noisy seeds
    seed_base = frame_seed % (2**31 - 1 - len(SAMPLES_NOISY))
    kept, frame_meta["render"] = render_frame(folder, seed_base)
    if not kept:
        lum = frame_meta["render"]["mean_luminance"]
        log(f"frame {attempt}: almost black (mean luminance {lum:.4f}), skipping")
        shutil.rmtree(folder, ignore_errors=True)
        rejected[str(attempt)] = f"almost black ({lum:.4f})"
        write_json(REJECTED_PATH, rejected)
        continue
    write_json(os.path.join(folder, "meta.json"), frame_meta)
    saved += 1
    log(f"frame {attempt}: saved ({saved}/{FRAMES})")

log(
    f"done: {saved}/{FRAMES} frames saved after {attempts} attempts, "
    f"{len(rejected)} rejected in total, {total_tries} camera tries this run"
)
if saved < FRAMES:
    raise RuntimeError(
        f"only {saved}/{FRAMES} frames after {attempts} attempts; "
        f"see {REJECTED_PATH} for why frames were skipped"
    )
