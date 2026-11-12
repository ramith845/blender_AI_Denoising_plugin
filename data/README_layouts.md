# Procedural layouts (Poly Haven rooms)

The roadmap's dataset plan (section 1.2) calls for about **100 procedural scenes × 15 frames**, alongside the demo files. These scenes are built from Poly Haven models, textures and HDRIs. Two new scripts make them:

- **`polyhaven_download.py`** (normal Python): downloads a fixed set of suitable Poly Haven assets.
- **`layout.py`** (runs inside Blender): builds random rooms from those assets and saves each one as a `.blend`.

The saved layouts are ordinary scenes. `generate.py` renders them with `custom_scene.py`, exactly like Classroom, with all the camera, light and material randomization.

## Steps

1. **Download the assets** (once, about 1 GB at the defaults):
   ```powershell
   & ".venv\Scripts\python.exe" "data\polyhaven_download.py"
   ```
   They go to `data/assets/polyhaven/`, along with an `index.json` listing them. Running it again skips files that are already downloaded.
2. **Build the layouts:**
   ```powershell
   & "E:\blender\stable\Blender 5.2\blender.exe" -b --factory-startup -P "data\layout.py" -- --count 100
   ```
   They go to `data/blendfiles/procedural/layout_0000.blend` and up, each with a `layout_0000.json` describing what's in it. Building takes seconds per layout; rendering is what takes time.
3. **Render them:** run `generate.py` as usual. It now finds every `layout_*.blend` automatically and renders `LAYOUT_FRAMES` (15) frames for each one, after the demo scenes in `SCENES`.

Before rendering all 100, open a few layouts in Blender or render a handful of frames, and check they look the way you want.

## What the downloader picks

It takes the **most-downloaded** assets in each group, using Poly Haven's categories and sizes:

| Group | Default count | Rule |
|---|---|---|
| Models | 40 | Furniture and props (`furniture`, `seating`, `table`, `shelves`, `decorative`, `lighting`, `electronics`, `appliances`, `containers`, `props`); no plants, rocks, food or buildings; footprint up to 2.5 m, height up to 2.4 m, at least 20 cm in size |
| Floor textures | 20 | Category `floor`, but not terrain, roads, sand, rock or aerial shots |
| Wall textures | 20 | Category `wall`, but not terrain or roofing |
| HDRIs | 25 | Outdoor, not night, indoor or studio |

- **Resolution:** models and textures are downloaded at 1k and HDRIs at 2k. Change the counts and resolutions with `--models`, `--floors`, `--walls`, `--hdris`, `--model-res`, `--texture-res` and `--hdri-res`.
- **Model files:** each model comes as its `.blend` plus the texture files it uses, saved at the same relative paths, so the `.blend` finds them.
- **Texture files:** each texture has 3 maps: diffuse, roughness and normal (the OpenGL-style normal map, which Blender uses).
- **License:** all Poly Haven assets are CC0 (public domain), so no attribution is required.

## What a layout contains

Each layout is one room:

| Part | How it's chosen |
|---|---|
| Room | Inside size 4–9 m × 4–9 m, 2.6–3.6 m high; 20 cm thick walls, floor and ceiling |
| Floor / walls / ceiling | One random floor texture, one wall texture for all 4 walls, one wall texture for the ceiling |
| Windows | 1–2 walls get 1–3 window openings each: 0.8–2.0 m wide, sill 0.6–1.0 m above the floor, top 0.3–0.6 m below the ceiling |
| Glass | In 30% of layouts the windows get 1 cm glass panes (refraction, a harder case for the denoiser); otherwise they're open |
| Outside | A random HDRI sky, seen and lit through the windows |
| Ceiling light | In 70% of layouts: one area light, 100–600 W, 0.4–1.5 m across, over the middle of the room |
| Props | 3–8 different models per layout, and 4–14 copies of them spread over the floor (repeats are fine, like a set of chairs) |
| Camera and `CamBounds` | A starting camera, plus a `CamBounds` Empty that exactly fills the inside of the room |

**How the props are placed:**
- Each prop gets a random spot on the floor, at least 15 cm from the walls and 10 cm from other props.
- Half the props are turned to a multiple of 90° (plus or minus 5°); the rest get any angle.
- If no free spot is found after 40 tries, that prop is left out.
- Overlaps are checked with each prop's rectangular footprint after it's turned.

**How the textures are mapped:** they use box projection with each object's own coordinates, scaled to the texture's real-world size from Poly Haven. A wood floor's planks or a brick wall's bricks come out at a believable size. The pattern also continues across the separate pieces of a wall with windows.

**How the models are loaded:** each model is loaded into the file once, and every copy shares its mesh. So 10 chairs cost about as much memory as 1. The model is first moved so the middle of its footprint, at floor level, is at its origin, which places props exactly on the floor.

## How it fits with the rest of the pipeline

- **Camera bounds:** `CamBounds` matches the room, so `custom_scene.py` keeps the camera inside it, with the usual 0.3 m margin from the walls and a height of 0.8–2.0 m.
- **Aim targets:** the walls, floor and ceiling sit outside `CamBounds`, so they're never aimed at; the props are.
- **Lighting:** the layout has no sun, so `custom_scene.py` adds its random `Rand_Sun`, which shines through the windows. The world uses a Mapping node, so `custom_scene.py`'s random HDRI rotation and strength take effect.
- **Materials:** the room and prop materials are all local Principled BSDFs, so the hue, roughness, metal and glass randomization can change all of them. Linked materials, as in Classroom, can't be changed.

## Re-running and changing things

- **Reproducible:** layout N always comes out the same, because its random generator is seeded with `"layout/N"`, as long as the downloaded assets in `index.json` stay the same.
- **Skipping:** existing layouts are skipped. Use `--overwrite` to rebuild them, and `--start N --count M` to build a range, for example to add 50 more later.
- **Changing ranges:** all the ranges are constants at the top of `layout.py`, such as `ROOM_WIDTH`, `GLASS_PROB` and `PROP_COUNT`. Rebuild with `--overwrite` after changing them, and delete those layouts' folders in `data/dataset/`, since their old frames show the old rooms.
- **Changing assets:** if you download a different asset set, the same layout number picks different assets. Rebuild all layouts then.

## Output

```
data/blendfiles/procedural/
  layout_0000.blend     the room, ready for generate.py
  layout_0000.json      room size, textures, HDRI, windows, glass, ceiling light, props
  ...
```

- **Texture paths:** the paths inside the `.blend` files point to `data/assets/polyhaven/` relative to the layout, so keep both folders where they are.
- **Rendered frames:** these go to `data/dataset/layout_0000/` and up, like every other scene.

## Tested

These tests used Blender 5.2.2, with a small asset set (4 models, 2 floors, 2 walls, 2 HDRIs, 41 MB) in a scratch folder:
- **Building:** 3 layouts were built and saved, with no missing texture files. Preview renders, from above and from the start camera, showed closed rooms with window openings, the HDRI outside, textures at a sensible scale, and props on the floor without overlaps.
- **Rendering:** `custom_scene.py` on the 3 layouts (fast settings, 8 frames each) found cameras within 1–3 tries and saved 24/24 frames with all files.
- **Brightness:** 4 of 28 attempts were rejected as almost black, all in the 2 layouts without a ceiling light. The saved frames ranged from quite dark (mean luminance 0.006) to bright (1.9).

Not yet tested: the full 40-model download, and rendering at the real settings. Rough time: 100 layouts × 15 frames × ~40 s per frame ≈ 17 GPU-hours (the per-frame time is from Classroom). Simple rooms may render faster.

## Choices to revisit

You didn't specify these, so I picked defaults:
- **One generic room type.** It's a box with windows and furniture on the floor. Nothing is stacked on tables, there are no doors, and there are no L-shaped rooms.
- **40 models, 20 floors, 20 walls, 25 HDRIs,** taken by download count.
- **15 frames per layout,** as in the roadmap (`LAYOUT_FRAMES` in `generate.py`).
- **Darker rooms are kept.** The 30% of layouts without a ceiling light are often dark. Dark frames are realistic and noisy, which is useful, but more get rejected. Raise `CEILING_LIGHT_PROB` if too many are skipped.
- **Rooms can feel sparse.** The largest rooms (9 × 9 m) can look empty. Narrow `ROOM_WIDTH`/`ROOM_DEPTH` or raise `PROP_COUNT` if needed.
