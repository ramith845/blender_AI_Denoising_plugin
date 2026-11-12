"""Download a fixed set of Poly Haven assets for the procedural layouts (roadmap 1.2).

Run with normal Python (no Blender needed):
    python data/polyhaven_download.py

It picks the most-downloaded assets that suit indoor rooms:
- models:  furniture and props that fit in a room (no plants, rocks, buildings)
- floors:  floor textures that aren't terrain, roads or aerial shots
- walls:   wall textures (plaster, brick, concrete, tiles, ...)
- HDRIs:   outdoor daytime skies, seen through the windows

Files go to data/assets/polyhaven/, plus an index.json that layout.py reads.
Re-running skips files that are already downloaded, so it can resume.
All Poly Haven assets are CC0 (public domain): https://polyhaven.com/license
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

API = "https://api.polyhaven.com"
# Poly Haven asks API users to send a User-Agent that identifies the project
USER_AGENT = "AI-Denoising-dataset/1.0 (personal research project)"

# Categories (from the API) that decide which assets are suitable
MODEL_KEEP = {"furniture", "seating", "table", "shelves", "decorative", "lighting",
              "electronics", "appliances", "containers", "props"}
MODEL_SKIP = {"nature", "plants", "trees", "rocks", "ground cover", "structures", "food"}
FLOOR_SKIP = {"aerial", "terrain", "road", "sand", "rock", "asphalt", "cobblestone",
              "roofing", "bark"}
WALL_SKIP = {"aerial", "terrain", "roofing", "road", "bark"}
HDRI_SKIP = {"night", "indoor", "studio", "artificial light"}

# Size limits for models, in millimetres (the API's unit): small enough to fit in a
# 4 m room with space around them, big enough to be visible from across the room
MODEL_MAX_FOOTPRINT_MM = 2500
MODEL_MAX_HEIGHT_MM = 2400
MODEL_MIN_SIZE_MM = 200


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download(url, path, size=None):
    """Download url to path unless it's already there (same size). Writes to a .part
    file first, so an interrupted download is never mistaken for a finished one."""
    path = Path(path)
    if path.exists() and (size is None or path.stat().st_size == size):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)
    os.replace(part, path)
    time.sleep(0.2)  # be gentle with Poly Haven's servers
    return True


def most_downloaded(assets, keep):
    """Asset ids passing the `keep(info)` filter, most downloaded first."""
    ids = [asset_id for asset_id, info in assets.items() if keep(info)]
    return sorted(ids, key=lambda a: assets[a].get("download_count", 0), reverse=True)


def pick_models(count):
    assets = get_json(f"{API}/assets?t=models")

    def keep(info):
        cats = set(info.get("categories", []))
        dims = info.get("dimensions") or [0, 0, 0]  # x, y, z in mm
        return (
            cats & MODEL_KEEP
            and not cats & MODEL_SKIP
            and max(dims[0], dims[1]) <= MODEL_MAX_FOOTPRINT_MM
            and dims[2] <= MODEL_MAX_HEIGHT_MM
            and max(dims) >= MODEL_MIN_SIZE_MM
        )

    return assets, most_downloaded(assets, keep)[:count]


def pick_textures(floors, walls):
    assets = get_json(f"{API}/assets?t=textures")

    def is_floor(info):
        cats = set(info.get("categories", []))
        return "floor" in cats and not cats & FLOOR_SKIP

    def is_wall(info):
        cats = set(info.get("categories", []))
        return "wall" in cats and not cats & WALL_SKIP

    floor_ids = most_downloaded(assets, is_floor)[:floors]
    wall_ids = [a for a in most_downloaded(assets, is_wall) if a not in floor_ids][:walls]
    return assets, floor_ids, wall_ids


def pick_hdris(count):
    assets = get_json(f"{API}/assets?t=hdris")

    def keep(info):
        cats = set(info.get("categories", []))
        return "outdoor" in cats and not cats & HDRI_SKIP

    return assets, most_downloaded(assets, keep)[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "assets" / "polyhaven"))
    parser.add_argument("--models", type=int, default=40)
    parser.add_argument("--floors", type=int, default=20)
    parser.add_argument("--walls", type=int, default=20)
    parser.add_argument("--hdris", type=int, default=25)
    parser.add_argument("--model-res", default="1k")
    parser.add_argument("--texture-res", default="1k")
    parser.add_argument("--hdri-res", default="2k")
    args = parser.parse_args()
    out = Path(args.out)
    index = {"models": {}, "textures": {}, "hdris": {}}

    # Models: the .blend file plus the textures it references ("include"), saved
    # with the same relative paths so the .blend finds them
    model_assets, model_ids = pick_models(args.models)
    for n, asset_id in enumerate(model_ids, 1):
        blend = get_json(f"{API}/files/{asset_id}")["blend"][args.model_res]["blend"]
        folder = out / "models" / asset_id
        blend_path = folder / Path(blend["url"]).name
        download(blend["url"], blend_path, blend.get("size"))
        for rel_path, file in blend.get("include", {}).items():
            download(file["url"], folder / rel_path, file.get("size"))
        info = model_assets[asset_id]
        index["models"][asset_id] = {
            "blend": blend_path.relative_to(out).as_posix(),
            "categories": info.get("categories", []),
            "dimensions_mm": info.get("dimensions"),
        }
        print(f"model {n}/{len(model_ids)}: {asset_id}", flush=True)

    # Textures: just the three maps the room materials use
    texture_assets, floor_ids, wall_ids = pick_textures(args.floors, args.walls)
    maps = {"diffuse": "Diffuse", "rough": "Rough", "normal": "nor_gl"}  # ours -> API name
    for n, asset_id in enumerate(floor_ids + wall_ids, 1):
        files = get_json(f"{API}/files/{asset_id}")
        entry = {}
        for key, api_name in maps.items():
            file = files[api_name][args.texture_res]["jpg"]
            path = out / "textures" / asset_id / Path(file["url"]).name
            download(file["url"], path, file.get("size"))
            entry[key] = path.relative_to(out).as_posix()
        info = texture_assets[asset_id]
        # Real-world size of one texture tile, in metres (API gives millimetres)
        dims = info.get("dimensions") or [2000, 2000]
        entry["size_m"] = [dims[0] / 1000, dims[1] / 1000]
        entry["use"] = "floor" if asset_id in floor_ids else "wall"
        entry["categories"] = info.get("categories", [])
        index["textures"][asset_id] = entry
        print(f"texture {n}/{len(floor_ids) + len(wall_ids)}: {asset_id} ({entry['use']})", flush=True)

    # HDRIs
    hdri_assets, hdri_ids = pick_hdris(args.hdris)
    for n, asset_id in enumerate(hdri_ids, 1):
        file = get_json(f"{API}/files/{asset_id}")["hdri"][args.hdri_res]["hdr"]
        path = out / "hdris" / Path(file["url"]).name
        download(file["url"], path, file.get("size"))
        index["hdris"][asset_id] = {
            "hdr": path.relative_to(out).as_posix(),
            "categories": hdri_assets[asset_id].get("categories", []),
        }
        print(f"hdri {n}/{len(hdri_ids)}: {asset_id}", flush=True)

    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(
        f"done: {len(index['models'])} models, {len(index['textures'])} textures, "
        f"{len(index['hdris'])} HDRIs -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
