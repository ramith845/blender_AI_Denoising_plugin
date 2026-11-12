# Phase 1: dataset generation

> **Procedural Poly Haven rooms:** see [README_layouts.md](README_layouts.md). `generate.py` now also renders every `blendfiles/procedural/layout_*.blend`, at `LAYOUT_FRAMES` (15) frames each.

Two scripts in this folder make the training data for roadmap Phase 1:

- **`generate.py`** runs with normal Python. It starts Blender headless once for each scene in `SCENES` and runs `custom_scene.py` inside it.
- **`custom_scene.py`** runs inside Blender. Every frame, it randomizes the camera, sun, world, extra lights and materials. It then renders 4 noisy images and 1 ground truth (gt) and saves them with the albedo, normal and depth passes and a `meta.json`.

## Quick start

1. **Prepare each scene:** add a `CamBounds` Empty to the scene's .blend (see below) and save the file.
2. **Choose scenes:** list the scenes in `SCENES` in `generate.py`. Only Classroom is turned on now.
3. **Choose frame count:** set `FRAMES` in `generate.py`. It's 150; use about 20 the first time to check the output (roadmap 1.2).
4. **Run it** from any folder (paths are based on where `generate.py` lives):
   ```powershell
   & ".venv\Scripts\python.exe" "data\generate.py"
   ```
5. **Watch progress:** each scene's progress goes to `<scene>_render.log` next to its .blend, and errors go to `<scene>_render.err`. Both are overwritten on each run.

If a run stops for any reason (Ctrl+C, a crash, a reboot), run it again. Finished frames are kept, and it continues where it stopped.

## Setting up a scene

1. Add an **Empty** with display type **Cube** and name it `CamBounds`.
2. Move and scale it so it fills the **inside** of the room, with its bottom face on the floor.
   - The camera and the extra lights stay inside this box.
   - It must be an Empty. A mesh cube would block the ray casts.
3. If there's no `CamBounds`, the script uses a box of ±3 m around the scene's own camera. This works, but the views stay close to that camera.

## Output

```
data/dataset/<scene>/
  rejected.json          frames that were skipped, and why
  frame_0000/
    noisy_16spp.exr      16 spp, RGB, half float
    noisy_8spp.exr       8 spp
    noisy_4spp.exr       4 spp
    noisy_1spp.exr       1 spp
    gt.exr               2048 spp, RGB, 32-bit float
    albedo.exr           Denoising Albedo from the 16 spp render, RGB, half
    normal.exr           Denoising Normal from the 16 spp render, XYZ, half
    depth.exr            Z pass from the 16 spp render, 32-bit float
    meta.json            every random value, seeds, render times, brightness
```

- **Format:** every EXR is 512 × 512, scene-linear, and lossless (ZIP).
- **Numbering:** frame numbers can have gaps. A rejected frame's number is recorded in `rejected.json` and never reused.

## How one frame is made (roadmap 1.3–1.6)

1. **Camera:** up to 50 random tries. If none passes the checks, the frame is rejected as "no good camera".
2. **Randomize:** the sun, world, extra lights and materials.
3. **16 spp render:** saved as `noisy_16spp.exr`. If its mean brightness (luminance) is below 0.005, the folder is deleted and the frame is rejected as "almost black".
4. **Aux passes:** the same 16 spp render is saved as a temporary multilayer EXR. Albedo, normal and depth are copied into their own files, and the temporary file is deleted.
5. **Remaining renders:** 8, 4 and 1 spp, then the 2048 spp gt.
6. **`meta.json`:** written last, through a temp file. A frame with a `meta.json` is complete.

**Seeds:**
- The gt uses the frame seed. The noisy renders use the frame seed +1 to +4, so the gt never contains the noisy images' samples.
- The frame seed is `crc32("<scene>/<frame number>")`, so the same scene and frame always get the same random choices.
- Seeds are kept below Cycles' limit of 2³¹ − 1.

**Fixed settings:** Cycles, 512 × 512 at 100%. Adaptive sampling, denoising, animated seed and transparent film are off. Persistent data is on, which reuses the scene's BVH acceleration structure between the 5 renders.

**GPU:** the script selects OptiX in the Cycles preferences itself. If no OptiX GPU is found, it stops with an error instead of quietly rendering on the CPU. To use something else, change `script["--cycles-device"]` in `generate.py` to `CUDA`, `CPU` or `OPTIX+CPU`.

**Giving up:** a scene stops after `FRAMES × 3` attempts. If it didn't reach `FRAMES`, it exits with an error. `generate.py` then reports that scene as failed and moves on to the next one.

## Resuming and re-running

- **Re-running:** frames that are complete (have `meta.json`) or are listed in `rejected.json` are skipped. A frame that was interrupted halfway is rendered again from the start.
- **After changing the camera checks or thresholds:** delete that scene's `rejected.json` so the rejected frames get another try.
- **After changing the randomization ranges:** use a fresh output folder. The old frames were made with the old ranges.
- **Reproducibility:** re-rendering a frame gives the same scene setup (identical `meta.json` values), but the images may not be bit-identical. In one test the 16 spp brightness differed by about 1% between two runs that had rendered different frames before it.

## Running it inside Blender (for testing)

You can also paste `custom_scene.py` into Blender's Text Editor. Before running:
- Save the .blend first.
- Open **Window → Toggle System Console** to see the log.

Without arguments it renders 150 frames to `//dataset` next to the .blend, and the UI is frozen while it runs. It also adds objects and nodes to the open file (see below), so don't save afterwards; use **File → Revert**.

## Measured so far

From the first 20-frame Classroom run on the RTX 3060 laptop GPU with OptiX:
- **Time:** 874 s for 20 frames, about **44 s per frame**. The gt render takes 27–59 s depending on the view and the lights (42 s on average), and the four noisy renders together take about 1 s. 150 frames take about **1 h 50 min per scene**.
- **Disk:** about **8.3 MB per frame**, or about **1.25 GB per 150-frame scene**.

## Randomization (roadmap 1.7)

**Camera** (`randomize_camera`, checked by `is_current_cam_pos_good`):
- **Position:** inside `CamBounds`, 0.3 m from the walls and 0.8–2.0 m above the floor.
- **Aim:** 50/50 between a random point on a visible object and a random point in the room. Room-sized meshes (bigger than 3 m: walls, floor, ceiling, mouldings) aren't aimed at, and points in the room stay 1 m away from its sides.
- **Lens:** focal length 18–50 mm.
- **Depth of field:** on 20% of frames, with an f-stop of 1.4–8 (log-uniform) and focus on what the center ray hits.
- **Rejected when:**
  - fewer than 95% of the 64 grid rays hit something;
  - fewer than 4 distinct objects are visible;
  - more than half the grid rays hit something within 1 m;
  - the center ray hits within 1.5 m;
  - or the camera is inside an object (most hits tiny, or all 6 axis rays hit within 0.2 m).

**Lights, world and materials** (`randomize_frames`):

| Part | Range |
|---|---|
| Sun | Elevation 5–80°, azimuth 0–360°, strength 1–10 (log-uniform), angle 0.5–5°, 3000–7000 K |
| World HDRI | Strength 0.3–3 (log-uniform); rotation 0–360° if the world has an Environment Texture |
| Extra lights | 0–3 point or area lights, 50–1000 W (log-uniform), inside `CamBounds` |
| Materials | ~30% get a hue shift (±0.1) and roughness 0.05–0.9; ~10% become metal; ~5% become glass |

How the materials part works:
- **Which materials:** only those with a **Principled BSDF**. Materials linked from another .blend file are read-only, so they're skipped.
- **Originals:** each material's original values are saved once in a custom property, `rand_orig`, and restored at the start of every frame.

All ranges are constants at the top of `custom_scene.py`.

## Changes

### Camera fixes after the 20-frame review

- **The real cause of the wall close-ups and black patches:** Classroom's `renderCam` has keyframes on its location. Rendering re-applies animation, so every render moved the camera back to its keyed spot near a wall but kept the random rotation. The camera checks tested the random position, so they couldn't catch this. The script now removes keyframes and drivers from everything it randomizes (camera, sun, world, edited materials) before setting values.
- **Aim targets:** meshes bigger than 3 m (`MAX_TARGET_SIZE`) are no longer aimed at. "Random point in the room" aims stay 1 m (`TARGET_MARGIN`) away from the room's sides.
- **Close-ups:** a camera is rejected if more than half its grid rays hit something within 1 m (`NEAR_DIST`, `MAX_NEAR_FRACTION`).
- **Holes:** `MIN_COVERAGE` is now 0.95 (it was 0.8).
- **Checked** on 20 Classroom frames at fast settings. The views are varied (desks, chairs, blackboard, windows, doors), with no wall close-ups or black patches. It took 21 camera tries for 20 frames.

### Fix after the first 20-frame run

- **`albedo.exr` and `normal.exr` were unreadable in every frame.** Each file had a header but no pixels.
  - **Cause:** picking a pass's channels out of the multilayer file (`pixels[..., [i, j, k]]`) gave an array whose memory isn't stored row by row. OpenImageIO refuses to write those, and the code didn't check whether the write worked.
  - **Fix:** `write_exr` now makes a normal copy first, and raises an error if a write fails.
  - `depth.exr` was fine, because a single-channel array is stored normally.
- **Frames rendered before this fix must be rendered again.**

### Round 2: everything needed for a full run

| # | Problem | Fix |
|---|---|---|
| 1 | Albedo, normal and depth were turned on but never saved (single-layer EXR keeps only the image) | The 16 spp render is also saved as a multilayer EXR, and the three passes are copied into `albedo.exr`, `normal.exr` and `depth.exr` with OpenImageIO, which ships with Blender |
| 2 | No `meta.json`; every run started again at frame 0 | `meta.json` is written last and marks a frame complete; re-runs skip complete frames; `rejected.json` remembers skipped frames |
| 3 | Same render seeds (0–4) in every frame | Per-frame seeds: gt = frame seed, noisy = frame seed +1 to +4, kept below Cycles' limit |
| 4 | `generate.py` walked folders and would have rendered 13 files, including `test.blend` | Scenes are listed by hand in `SCENES` |
| 5 | Paths depended on the folder you launched from | Paths are based on where `generate.py` lives |
| 6 | Python errors inside Blender still exited with code 0 | `--python-exit-code 1`; `generate.py` reports failed scenes, continues with the next one, and exits with an error at the end |
| 7 | 1024 × 1024 (4× the render time) | 512 × 512 |
| 8 | Render order 1, 4, 8, 16, gt with no black check | 16 spp first, black check, then 8, 4, 1 and gt; black frames are deleted and recorded |
| 9 | Skipped frames meant fewer than 150 | Keeps going until `FRAMES` are saved, up to `FRAMES × 3` attempts |
| 10 | GPU not guaranteed (silent CPU fallback) | The script selects the device itself and stops if it isn't found |
| 11 | Everything saved as 32-bit | Noisy, albedo and normal are half floats; gt and depth are 32-bit |

Also in this round:
- **Two multilayer layouts:** Blender writes multilayer EXRs either with one part per pass or with everything in one part (Classroom does the latter). The split handles both, and uses one view layer if a file has several.
- **New `custom_scene.py` arguments:** `--frames` and `--cycles-device`, next to `--out-dir`.
- **Removed:** the old `render()`, `RENDER_SETTINGS`, `valid_engines` and `valid_render_devices` are replaced by `setup_render()`, `render_frame()` and helpers.

### Round 1: randomization

- **`randomize_camera`:** rewritten as described above. It replaces the ±10 m offsets that kept putting the camera outside Classroom.
- **`randomize_frames`:** written (sun, world, extra lights, materials).
- **`PCG32`:** new `log_uniform()` and `chance()` methods.
- **Camera check:** the 6-axis "inside an object" check was added.
- **Seeding:** the generator is seeded per frame.

## What the script adds to the open file

This only matters when running inside the Blender UI:
- **Objects:** `Rand_Sun` (only if the scene had no sun) and `Rand_Light*`.
- **Nodes:** `Rand_HSV*` in materials with textured base colors, and `Rand_Mapping` in the world.
- **Custom property:** `rand_orig` on each edited material.
- **Removed animation:** keyframes and drivers on the camera, sun, world and edited materials are deleted.

## Notes and limitations

- **Classroom materials barely change.** Only 2 of its materials are editable. The furniture is linked from `classroom/assets/*.blend`, and linked data is read-only. Camera and lighting still vary. A later option is to make the linked data local in memory before randomizing.
- **Classroom's world is black, at strength 0.** Its light comes from its own sun and a 1963 W area light outside the windows. So the world randomization has no visible effect in Classroom, the windows look black, and any part of a view that sees past the room's geometry is pure black.
- **`MIN_CENTER_DIST` is 1.5 m.** The roadmap says 0.5 m; 1.5 m rejects close-up views.
- **Choices not in the roadmap:**
  - Look-at is 50/50 between objects and the room.
  - Roughness changes only on the ~30% of materials that get a hue shift.
  - Glass uses roughness 0.05 and IOR 1.45.
  - Area lights are 0.25–1 m.
  - The fallback box is ±3 m.

## Still to do

- **Re-render Classroom:** delete `data/dataset/classroom` (its albedo and normal files are broken), run 20 frames again, check them, then run the full 150.
- Add `CamBounds` to any other scene you put in `SCENES`. Barbershop only if it fits in 6 GB of VRAM.
- Write `splits.json` by hand: 80/10/10 by scene, with one whole demo file kept out for testing.

## Testing so far

These tests ran in Blender 5.2.2.

- **Test room, real renders** (at fast settings: 128², gt 64 spp):
  - Every frame had all 8 EXRs with the right channels, bit depth and ZIP compression, plus `meta.json`, and all 15 seeds across 3 frames were distinct.
  - **Resume:** rendered only the missing frame and left the others untouched.
  - **Forced black frames:** folders were deleted, `rejected.json` was written, the scene stopped at the attempt cap with exit code 1, and a re-run skipped those frames.
- **`generate.py`,** launched from another folder: rendered the scene, reported a missing scene, and exited with code 1.
- **One Classroom frame at the real settings:** all files correct. This run found the single-part multilayer layout, which is now handled.
- **First 20-frame Classroom run:** this found the unreadable albedo and normal files described above. My earlier checks had only read the file headers. Now every pixel of every file is read back, in both multilayer layouts.
