"""Split the dataset into train / val / test by scene and write data/dataset/splits.json.

Splitting by scene (not by frame) means the val and test scores measure how well the
model works on scenes it has never seen. A frame only counts if it's complete
(has meta.json). Re-running gives the same split.

    python models/make_splits.py
"""

import json
import random
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset"

TEST_SCENES = ["classroom"]  # whole demo scenes kept out of training
VAL_SCENES = ["blender-4.3-splash"]
LAYOUTS_PER_SPLIT = 10  # procedural layouts each for val and for test
SEED = "splits-v1"  # change it to draw different layouts


def complete_frames(scene_dir):
    return sorted(p.parent.name for p in scene_dir.glob("frame_*/meta.json"))


def main():
    scenes = {d.name: complete_frames(d) for d in sorted(DATASET.iterdir()) if d.is_dir()}
    empty = [s for s, frames in scenes.items() if not frames]
    scenes = {s: f for s, f in scenes.items() if f}

    missing = [s for s in TEST_SCENES + VAL_SCENES if s not in scenes]
    if missing:
        raise SystemExit(f"scenes not found (or empty) in {DATASET}: {missing}")

    layouts = sorted(s for s in scenes if s.startswith("layout_"))
    picked = random.Random(SEED).sample(layouts, 2 * LAYOUTS_PER_SPLIT)
    split_of = {s: "train" for s in scenes}
    for s in TEST_SCENES + picked[:LAYOUTS_PER_SPLIT]:
        split_of[s] = "test"
    for s in VAL_SCENES + picked[LAYOUTS_PER_SPLIT:]:
        split_of[s] = "val"

    splits = {name: sorted(s for s in scenes if split_of[s] == name) for name in ("train", "val", "test")}
    out = {
        "seed": SEED,
        "excluded_empty_scenes": empty,
        "splits": splits,
        # Frame folder names per scene, so every script uses exactly the same frames
        "frames": scenes,
    }
    (DATASET / "splits.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for name, members in splits.items():
        frames = sum(len(scenes[s]) for s in members)
        demo = [s for s in members if not s.startswith("layout_")]
        print(f"{name:5s}: {len(members):3d} scenes, {frames:5d} frames ({4 * frames} pairs); demo scenes: {demo}")
    if empty:
        print(f"excluded (no complete frames): {empty}")
    print(f"wrote {DATASET / 'splits.json'}")


if __name__ == "__main__":
    main()
