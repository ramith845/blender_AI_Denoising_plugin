"""Build procedural indoor scenes ("layouts") from Poly Haven assets (roadmap 1.2).

Run inside Blender, headless:
    blender -b --factory-startup -P data/layout.py -- --count 100

Each layout is a random room: floor, walls and ceiling with Poly Haven textures,
windows looking out onto an HDRI sky, an optional ceiling light, and Poly Haven
furniture and props spread over the floor. It's saved as its own .blend with a
camera and a CamBounds Empty, so generate.py / custom_scene.py render it like any
other scene.

Layout N is always built the same way (its random generator is seeded with N), so
a layout can be rebuilt exactly. Existing layouts are skipped unless --overwrite.
"""

import argparse
import json
import math
import os
import random
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Ranges for the random choices (metres unless noted) ----
ROOM_WIDTH = (4.0, 9.0)  # inside size along X
ROOM_DEPTH = (4.0, 9.0)  # inside size along Y
ROOM_HEIGHT = (2.6, 3.6)
WALL_THICKNESS = 0.2

WINDOW_WALLS = (1, 2)  # how many walls get windows
WINDOWS_PER_WALL = (1, 3)
WINDOW_WIDTH = (0.8, 2.0)
WINDOW_SILL = (0.6, 1.0)  # height of the bottom of the window above the floor
WINDOW_TOP_GAP = (0.3, 0.6)  # gap between the top of the window and the ceiling
WINDOW_MIN_HEIGHT = 0.8
WINDOW_EDGE_GAP = 0.4  # minimum wall left between windows and next to corners
GLASS_PROB = 0.3  # chance the windows have glass panes (refraction = a harder case)

CEILING_LIGHT_PROB = 0.7
CEILING_LIGHT_POWER = (100.0, 600.0)  # W
CEILING_LIGHT_SIZE = (0.4, 1.5)

PROP_COUNT = (4, 14)
PROP_WALL_GAP = 0.15  # props stay this far from the walls
PROP_GAP = 0.1  # and this far from each other
PROP_TRIES = 40  # random spots tried per prop before giving it up
AXIS_ALIGNED_PROB = 0.5  # chance a prop is turned to a multiple of 90 degrees


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", default=os.path.join(DATA_DIR, "assets", "polyhaven"))
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "blendfiles", "procedural"))
    parser.add_argument("--start", type=int, default=0)  # first layout number
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def log(*args):
    print("[layout]", *args, flush=True)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def new_material(name):
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:  # older Blender versions start without nodes
        mat.use_nodes = True
    return mat


def texture_material(name, tex, assets_dir):
    """Principled material from a Poly Haven texture (diffuse, roughness, normal).

    Uses Object coordinates with box projection, so the texture is tiled at its
    real-world size and lines up across separate wall pieces (the room meshes keep
    their origin at the world origin)."""
    mat = new_material(name)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")

    coords = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    size_x, size_y = tex["size_m"]
    mapping.inputs["Scale"].default_value = (1 / size_x, 1 / size_y, 1 / size_x)
    links.new(coords.outputs["Object"], mapping.inputs["Vector"])

    def image_node(key, colorspace):
        node = nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(os.path.join(assets_dir, tex[key]), check_existing=True)
        node.image.colorspace_settings.name = colorspace
        node.projection = "BOX"  # no UVs needed; blends between the 3 axis projections
        node.projection_blend = 0.2
        links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    links.new(image_node("diffuse", "sRGB").outputs["Color"], bsdf.inputs["Base Color"])
    links.new(image_node("rough", "Non-Color").outputs["Color"], bsdf.inputs["Roughness"])
    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(image_node("normal", "Non-Color").outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def glass_material():
    mat = new_material("window_glass")
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.0
    bsdf.inputs["IOR"].default_value = 1.45
    return mat


# ---------------------------------------------------------------------------
# Room geometry
# ---------------------------------------------------------------------------


def add_boxes(name, boxes, material, collection):
    """One mesh object made of axis-aligned boxes, each given as (min corner, max corner)
    in world space. The object stays at the world origin (see texture_material)."""
    bm = bmesh.new()
    for lo, hi in boxes:
        if any(hi[k] - lo[k] < 1e-4 for k in range(3)):
            continue  # zero-size piece, e.g. a window touching the floor
        verts = bmesh.ops.create_cube(bm, size=1.0)["verts"]
        for v in verts:
            # create_cube spans -0.5..0.5; stretch it to lo..hi
            v.co = Vector([lo[k] + (v.co[k] + 0.5) * (hi[k] - lo[k]) for k in range(3)])
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def wall_pieces(u_lo, u_hi, height, windows):
    """Split a wall into solid pieces around its window holes.

    u runs along the wall; windows are (u_start, u_end, z_bottom, z_top), sorted.
    Returns (u_start, u_end, z_bottom, z_top) for every solid piece."""
    pieces = []
    u = u_lo
    for wa, wb, za, zb in windows:
        pieces.append((u, wa, 0.0, height))  # full-height wall before the window
        pieces.append((wa, wb, 0.0, za))  # below the window
        pieces.append((wa, wb, zb, height))  # above the window
        u = wb
    pieces.append((u, u_hi, 0.0, height))
    return pieces


def random_windows(rng, length, height):
    """Random window holes along a wall of this inside length."""
    count = rng.randint(*WINDOWS_PER_WALL)
    widths = [rng.uniform(*WINDOW_WIDTH) for _ in range(count)]
    # Drop windows until they fit with the required gaps between them and the corners
    while widths and sum(widths) + WINDOW_EDGE_GAP * (len(widths) + 1) > length:
        widths.pop()
    if not widths:
        return []
    free = length - sum(widths) - WINDOW_EDGE_GAP * (len(widths) + 1)
    # Share the spare length out randomly between the gaps
    weights = [rng.random() for _ in range(len(widths) + 1)]
    gaps = [free * w / sum(weights) for w in weights]

    windows = []
    u = -length / 2 + WINDOW_EDGE_GAP + gaps[0]
    for i, width in enumerate(widths):
        sill = rng.uniform(*WINDOW_SILL)
        top = max(height - rng.uniform(*WINDOW_TOP_GAP), sill + WINDOW_MIN_HEIGHT)
        windows.append((u, u + width, sill, min(top, height - 0.1)))
        u += width + WINDOW_EDGE_GAP + gaps[i + 1]
    return windows


def build_room(rng, width, depth, height, materials, collection):
    """Floor, ceiling and 4 walls around the inside box x, y in +-size/2, z in 0..height.

    Returns a description of the windows for the layout's JSON file."""
    t = WALL_THICKNESS
    x0, x1, y0, y1 = -width / 2, width / 2, -depth / 2, depth / 2
    add_boxes("floor", [((x0 - t, y0 - t, -t), (x1 + t, y1 + t, 0))], materials["floor"], collection)
    add_boxes("ceiling", [((x0 - t, y0 - t, height), (x1 + t, y1 + t, height + t))], materials["ceiling"], collection)

    # name: (axis the wall runs along, inside length, function from (u, z) range to a box)
    walls = {
        # North/south walls span the full outside width, so they also close the corners
        "wall_south": ("x", width, lambda ua, ub, za, zb: ((ua, y0 - t, za), (ub, y0, zb))),
        "wall_north": ("x", width, lambda ua, ub, za, zb: ((ua, y1, za), (ub, y1 + t, zb))),
        "wall_west": ("y", depth, lambda ua, ub, za, zb: ((x0 - t, ua, za), (x0, ub, zb))),
        "wall_east": ("y", depth, lambda ua, ub, za, zb: ((x1, ua, za), (x1 + t, ub, zb))),
    }
    window_walls = rng.sample(sorted(walls), rng.randint(*WINDOW_WALLS))
    glass = rng.random() < GLASS_PROB
    glass_panes = []
    windows_info = {}

    for name, (axis, length, to_box) in walls.items():
        windows = random_windows(rng, length, height) if name in window_walls else []
        # The x-walls run over the corners: extend their ends by the wall thickness
        extra = t if axis == "x" else 0.0
        pieces = wall_pieces(-length / 2 - extra, length / 2 + extra, height, windows)
        add_boxes(name, [to_box(*p) for p in pieces], materials["wall"], collection)
        if glass:
            for wa, wb, za, zb in windows:
                lo, hi = to_box(wa, wb, za, zb)
                # A 1 cm pane in the middle of the wall's thickness
                mid = [(lo[k] + hi[k]) / 2 for k in range(3)]
                cross = 1 if axis == "x" else 0  # the axis through the wall
                lo, hi = list(lo), list(hi)
                lo[cross], hi[cross] = mid[cross] - 0.005, mid[cross] + 0.005
                glass_panes.append((tuple(lo), tuple(hi)))
        if windows:
            windows_info[name] = [[round(v, 3) for v in w] for w in windows]

    if glass_panes:
        add_boxes("window_glass", glass_panes, glass_material(), collection)
    return {"windows": windows_info, "glass": bool(glass_panes)}


# ---------------------------------------------------------------------------
# Props (Poly Haven models)
# ---------------------------------------------------------------------------


def load_template(model_id, info, assets_dir):
    """Append a model's objects once and prepare them as a template to copy from.

    The template objects are never linked to the scene (they're dropped when the
    file is saved); each placed prop is a copy that shares their mesh data. The
    model is moved so the middle of its footprint, at floor level, is the origin."""
    path = os.path.join(assets_dir, info["blend"])
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.objects = src.objects
    objects = []
    for obj in dst.objects:
        if obj is None:
            continue
        if obj.type in {"MESH", "EMPTY"}:
            objects.append(obj)
        else:  # cameras or lights saved in the asset file
            bpy.data.objects.remove(obj)
    meshes = [o for o in objects if o.type == "MESH"]
    if not meshes:
        return None

    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector([min(c[k] for c in corners) for k in range(3)])
    hi = Vector([max(c[k] for c in corners) for k in range(3)])
    center = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    roots = [o for o in objects if o.parent not in objects]
    for root in roots:
        root.matrix_world = Matrix.Translation(-center) @ root.matrix_world
    return {
        "id": model_id,
        "objects": objects,
        "half_x": (hi.x - lo.x) / 2,
        "half_y": (hi.y - lo.y) / 2,
    }


def place_copy(template, location, angle, collection, number):
    """Link a copy of the template into the scene at location, turned by angle (radians)."""
    copies = {o: o.copy() for o in template["objects"]}  # shares mesh data
    root = bpy.data.objects.new(f"prop_{number:02d}_{template['id']}", None)
    collection.objects.link(root)
    for original, copy in copies.items():
        if original.parent in copies:
            copy.parent = copies[original.parent]  # keep the model's own hierarchy
        else:
            copy.parent = root  # root is at the origin, so the copy doesn't move
        collection.objects.link(copy)
    root.location = location
    root.rotation_euler = (0.0, 0.0, angle)


def footprint(template, angle):
    """Half-size of the prop's footprint along X and Y after turning it by angle."""
    c, s = abs(math.cos(angle)), abs(math.sin(angle))
    hx, hy = template["half_x"], template["half_y"]
    return c * hx + s * hy, s * hx + c * hy


def place_props(rng, templates, width, depth, collection):
    """Scatter random props over the floor without overlaps. Returns what was placed."""
    placed = []  # (x, y, half_x, half_y) of every prop so far
    info = []
    for number in range(rng.randint(*PROP_COUNT)):
        template = rng.choice(templates)  # repeats are fine (sets of chairs)
        for _ in range(PROP_TRIES):
            if rng.random() < AXIS_ALIGNED_PROB:
                angle = math.radians(rng.choice((0, 90, 180, 270)) + rng.uniform(-5, 5))
            else:
                angle = rng.uniform(0, 2 * math.pi)
            hx, hy = footprint(template, angle)
            # The room's floor, shrunk by the gap to the walls and the prop's own size
            x_range = width / 2 - PROP_WALL_GAP - hx
            y_range = depth / 2 - PROP_WALL_GAP - hy
            if x_range < 0 or y_range < 0:
                break  # too big for this room in any spot
            x, y = rng.uniform(-x_range, x_range), rng.uniform(-y_range, y_range)
            overlaps = any(
                abs(x - px) < hx + phx + PROP_GAP and abs(y - py) < hy + phy + PROP_GAP
                for px, py, phx, phy in placed
            )
            if not overlaps:
                place_copy(template, (x, y, 0.0), angle, collection, number)
                placed.append((x, y, hx, hy))
                info.append({"model": template["id"], "location": [round(x, 3), round(y, 3)],
                             "angle_deg": round(math.degrees(angle), 1)})
                break
    return info


# ---------------------------------------------------------------------------
# Lights, world, camera
# ---------------------------------------------------------------------------


def setup_world(hdri_path):
    """HDRI sky through a Mapping node, so custom_scene.py can rotate it."""
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    background = next(n for n in nodes if n.type == "BACKGROUND")
    env = nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri_path, check_existing=True)
    mapping = nodes.new("ShaderNodeMapping")
    coords = nodes.new("ShaderNodeTexCoord")
    links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    links.new(env.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = 1.0


def add_ceiling_light(rng, width, depth, height, collection):
    light = bpy.data.lights.new("ceiling_light", "AREA")
    light.energy = rng.uniform(*CEILING_LIGHT_POWER)
    light.size = rng.uniform(*CEILING_LIGHT_SIZE)
    obj = bpy.data.objects.new("ceiling_light", light)
    # Somewhere over the middle half of the room, just under the ceiling, facing down
    obj.location = (rng.uniform(-width / 4, width / 4), rng.uniform(-depth / 4, depth / 4), height - 0.05)
    collection.objects.link(obj)
    return {"power_w": round(light.energy, 1), "size_m": round(light.size, 2)}


def add_camera_and_bounds(width, depth, height, collection):
    """A starting camera (custom_scene.py moves it) and the CamBounds Empty it uses."""
    cam = bpy.data.objects.new("LayoutCam", bpy.data.cameras.new("LayoutCam"))
    cam.location = (0.0, -depth / 2 + 0.5, 1.6)
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)  # look along +Y, into the room
    cam.data.lens = 28
    collection.objects.link(cam)
    bpy.context.scene.camera = cam

    bounds = bpy.data.objects.new("CamBounds", None)
    bounds.empty_display_type = "CUBE"  # a 2 m cube at scale 1, so scale = half size
    bounds.location = (0.0, 0.0, height / 2)
    bounds.scale = (width / 2, depth / 2, height / 2)
    collection.objects.link(bounds)


# ---------------------------------------------------------------------------
# One layout
# ---------------------------------------------------------------------------


def build_layout(number, index, assets_dir, out_dir):
    rng = random.Random(f"layout/{number}")  # string seeds are stable across runs
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    collection = scene.collection

    width, depth, height = rng.uniform(*ROOM_WIDTH), rng.uniform(*ROOM_DEPTH), rng.uniform(*ROOM_HEIGHT)
    textures = index["textures"]
    floors = sorted(t for t, v in textures.items() if v["use"] == "floor")
    walls = sorted(t for t, v in textures.items() if v["use"] == "wall")
    floor_id, wall_id, ceiling_id = rng.choice(floors), rng.choice(walls), rng.choice(walls)
    materials = {
        "floor": texture_material("floor", textures[floor_id], assets_dir),
        "wall": texture_material("wall", textures[wall_id], assets_dir),
        "ceiling": texture_material("ceiling", textures[ceiling_id], assets_dir),
    }
    room = build_room(rng, width, depth, height, materials, collection)

    hdri_id = rng.choice(sorted(index["hdris"]))
    setup_world(os.path.join(assets_dir, index["hdris"][hdri_id]["hdr"]))
    ceiling_light = None
    if rng.random() < CEILING_LIGHT_PROB:
        ceiling_light = add_ceiling_light(rng, width, depth, height, collection)

    # Load a random handful of models for this layout, then scatter copies of them
    model_ids = rng.sample(sorted(index["models"]), min(len(index["models"]), rng.randint(3, 8)))
    templates = [load_template(m, index["models"][m], assets_dir) for m in model_ids]
    props = place_props(rng, [t for t in templates if t], width, depth, collection)

    add_camera_and_bounds(width, depth, height, collection)

    name = f"layout_{number:04d}"
    blend_path = os.path.join(out_dir, name + ".blend")
    # relative_remap: texture paths are saved relative to the .blend
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, relative_remap=True)

    missing = [img.name for img in bpy.data.images if img.source == "FILE" and not os.path.exists(bpy.path.abspath(img.filepath))]
    if missing:
        log(f"{name}: WARNING missing image files: {missing}")

    description = {
        "layout": name,
        "room_m": [round(width, 3), round(depth, 3), round(height, 3)],
        "floor": floor_id,
        "wall": wall_id,
        "ceiling": ceiling_id,
        "hdri": hdri_id,
        "ceiling_light": ceiling_light,
        **room,
        "props": props,
    }
    with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(description, f, indent=2)
    log(f"{name}: {width:.1f} x {depth:.1f} x {height:.1f} m, {len(props)} props, "
        f"{sum(len(w) for w in room['windows'].values())} windows, saved")


def main():
    args = parse_args()
    index_path = os.path.join(args.assets, "index.json")
    if not os.path.exists(index_path):
        raise RuntimeError(f"{index_path} not found; run polyhaven_download.py first")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    os.makedirs(args.out, exist_ok=True)

    for number in range(args.start, args.start + args.count):
        path = os.path.join(args.out, f"layout_{number:04d}.blend")
        if os.path.exists(path) and not args.overwrite:
            log(f"layout_{number:04d}: exists, skipping")
            continue
        build_layout(number, index, os.path.abspath(args.assets), args.out)


main()
