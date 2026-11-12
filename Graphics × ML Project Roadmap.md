# Graphics × ML Project Roadmap

Sep 30, 2026 · @Ramith Sudheer

## Overview

Build two projects in sequence: a Blender neural denoising add-on first, then a differentiable rendering project as the flagship. The add-on gets you a working, demoable ML tool fast inside a codebase you already know. Differentiable rendering then shows deeper graphics-and-ML skill that few candidates have.

Together with your merged Blender PR, the story on your resume becomes: contributor to a major C++ graphics codebase, who ships ML tools for artists, and understands rendering down to the gradients.

## Project 1: Blender neural denoising add-on

The goal is an add-on that takes a noisy, low-sample Cycles render and outputs a clean image using a small neural network you trained yourself. Blender already ships OpenImageDenoise and OptiX, so your angle is not beating them. It is showing you can build the whole pipeline end to end, and measure your model against theirs honestly.

**Phase 1: Generate a dataset**

1. Write a Python script using `bpy` that renders scenes twice: once at low samples (e.g. 4 to 16) and once at high samples (e.g. 1024+) as ground truth.
2. Also export auxiliary passes per frame: albedo, normals, and depth. These make denoising far easier and are what production denoisers use.
3. Randomize camera, lighting, and materials across free scenes (Blender demo files, Poly Haven assets) to get a few thousand image pairs.

*Detailed spec below, sized for Blender 5.2 on an RTX 3060 (6GB VRAM). Render times are estimates; time one Classroom frame first and adjust the counts.*

**1.1 Terms**

- **Scene**: one source of geometry, either a .blend demo file or one procedurally generated Poly Haven layout. The train/test split is done by scene.
- **Frame**: one randomized setup of a scene (camera + lights + materials). Each frame produces 1 ground truth render plus 4 noisy renders.
- **Pair**: one noisy render matched with its ground truth. Each frame gives 4 pairs.

**1.2 Dataset size and renders per .blend file**

| Source | Scenes | Frames per scene | Frames | Pairs |
|---|---|---|---|---|
| Blender demo files (Classroom, Junk Shop, Lone Monk, Barcelona Pavilion, Barbershop if it fits in 6GB) | 4–5 | 100–150 | ~500 | ~2,000 |
| Procedural Poly Haven layouts (models + HDRI + textures) | ~100 | 15 | ~1,500 | ~6,000 |
| Hard cases (glass/caustics, volumes, depth of field, dark interiors, small bright lights) | ~25 | 20 | ~500 | ~2,000 |
| **Total** | **~130** | | **~2,500** | **~10,000** |

- Cap each demo file at about 150 frames. After that, extra views mostly repeat the same materials and the model overfits to them.
- Start with Classroom only: 20 frames to check the pipeline, then the full 150.
- Keep at least one whole demo file out of training for the test set (e.g. Lone Monk), so test scores measure generalization.

**1.3 Resolution and output**

| Setting | Value | Why |
|---|---|---|
| Resolution | 512 × 512, 100% | Training uses 128×128 crops, so one frame gives 16 crops. Square avoids aspect-ratio handling. |
| File format | OpenEXR, ZIP compression (lossless) | Never DWAA/DWAB, which are lossy. |
| Bit depth | gt and depth 32-bit float; noisy, albedo, normal 16-bit half | Half is enough for inputs. Depth needs 32-bit because background depth overflows half. |
| Channels | RGB (no alpha); depth 1 channel | `film_transparent` off. |
| Color | Scene-linear (EXR is always saved linear); exposure 0, look None | Tonemapping happens in training, not in the files. |
| Disk | ~10–15 MB per frame → ~25–35 GB total | Keep it outside the git repo. |

**1.4 Render settings (Cycles)**

These must be **identical** for the noisy and gt renders except samples, seed and adaptive sampling. Otherwise the gt is a different image, not the converged version of the noisy one.

| Property (`scene.cycles.*` unless noted) | Noisy | Ground truth |
|---|---|---|
| `scene.render.engine` | `'CYCLES'` | same |
| `device` / compute type | `'GPU'`, OptiX in Preferences → System | same |
| `samples` | 1, 4, 8, 16 (one render each) | 2048 (1024 for simple procedural scenes) |
| `use_adaptive_sampling` | `False` | `False`, or `True` with `adaptive_threshold` 0.002 and `adaptive_min_samples` 512 if time is short |
| `use_denoising` | `False` | `False` |
| `seed` | frame_seed + 1 … + 4 (different per spp) | frame_seed |
| `use_animated_seed` | `False` | same |
| `max_bounces` / diffuse / glossy / transmission / volume | 12 / 4 / 4 / 12 / 0 (Blender defaults) | same |
| `sample_clamp_direct` / `sample_clamp_indirect` | 0 / 10 (Blender defaults) | same |
| `blur_glossy` (Filter Glossy) | 1.0 (default) | same |
| `film_transparent` | `False` | same |
| `pixel_filter_type` / `filter_width` | Blackman-Harris / 1.5 px | same |
| `scene.render.use_persistent_data` | `True` (keeps the BVH between the 5 renders of a frame) | same |
| `scene.render.use_motion_blur` | `False` at first; add later as a hard case | same |
| `texture_limit_render` | `'OFF'`; set to `'2048'` only if a scene runs out of VRAM | same |

The noisy seeds must differ from the gt seed. With the same seed, the first 16 samples of the gt are exactly the noisy image, which leaks the answer into the target.

Use Blender's defaults for bounces and clamping on purpose. The add-on will denoise renders made with default settings, so the training data should look like them.

**1.5 Passes to save**

| File | Source | Enable with |
|---|---|---|
| `noisy_spp1/4/8/16.exr` | Combined pass of each noisy render | always on |
| `gt.exr` | Combined pass of the 2048 spp render | always on |
| `albedo.exr` | Denoising Albedo, from the 16 spp render | `view_layer.cycles.denoising_store_passes = True` |
| `normal.exr` | Denoising Normal, from the 16 spp render | same as above |
| `depth.exr` | Z pass, from the 16 spp render | `view_layer.use_pass_z = True` |
| `meta.json` | Every randomized value, seeds, spp, render time | written by the script |

The aux passes come from a noisy render because at inference the add-on will only have noisy aux. Saving them once per frame (from 16 spp) keeps the file layout the Phase 2 thread expects. At 1–4 spp the real aux would be a bit noisier at edges; that's an accepted shortcut.

Layout: `dataset/<scene>/<frame:04d>/<file>.exr`, plus `dataset/splits.json` listing scenes for train/val/test (80/10/10 by scene).

**1.6 Order of work per frame**

1. Seed: `frame_seed = hash(scene_name, frame_idx)`; `rng = random.Random(frame_seed)`.
2. Randomize camera, lights, materials (ranges in 1.7).
3. Check the camera with `scene.ray_cast`; resample if it fails.
4. Render 16 spp first and save noisy + aux. Reject the frame if mean luminance < 0.005 (almost black).
5. Render 8, 4, 1 spp. Then render gt.
6. Write `meta.json` last, so a frame without it is known to be incomplete and can be re-rendered.

**1.7 Randomization ranges**

Full details and `bpy` snippets are in `phase1/scene-randomization-guide.md`. Use log-uniform sampling for strengths and powers.

| Group | Parameter | Range |
|---|---|---|
| Camera | Position | Inside the room bounds, 0.3 m from walls, height 0.8–2.0 m |
| | Look-at target | Random point in the room or on a visible object |
| | Focal length | 18–50 mm |
| | Depth of field | On for ~20% of frames, f-stop 1.4–8, focus on the center ray hit |
| Sun | Elevation / azimuth | 5–80° / 0–360° |
| | Strength / angle | 1–10 / 0.5–5° |
| | Color temperature | 3000–7000 K |
| World | HDRI strength / rotation | 0.3–3 / 0–360° |
| Extra lights | Count / power | 0–3 area or point lights / 50–1000 W |
| Materials | Hue shift | ±0.1 on ~30% of materials |
| | Roughness / metallic | 0.05–0.9 / metallic 1 on ~10% |
| | Glass | ~5% of materials (hard case) |

Reject a camera if the center ray hits within 0.5 m, or if rays along all 6 axes hit within 0.2 m (inside an object).

**1.8 Time budget (estimate, measure on your GPU)**

- Classroom gt at 512², 2048 spp on a 3060 with OptiX: roughly 40–90 s. The four noisy renders add about 5–10 s total.
- ~2,500 frames × ~1 min ≈ 40 GPU-hours. Run it overnight in chunks of a few hundred frames; the script should skip frames that already have `meta.json`.
- If it's too slow: drop gt to 1024 spp for procedural scenes, or reduce frames, before lowering resolution.
- Convergence check: render one frame at 4096 spp. If PSNR between the 2048 and 4096 spp renders is above ~40 dB, 2048 is enough.

**1.9 Phase 1 is done when**

- Classroom renders 150 frames with no manual steps, and the run can resume after being stopped.
- Every frame folder has all 8 EXRs plus `meta.json`.
- Histograms of the randomized values look spread out, and a contact sheet of 50 random gt frames looks varied with no black or broken frames.
- `splits.json` exists and no scene appears in two splits.

**Phase 2: Train a model**

1. Start with a small U-Net in PyTorch, taking noisy RGB + albedo + normals as input channels.
2. Train on random crops (e.g. 128×128) with an L1 loss, then try adding a perceptual loss.
3. Track PSNR and SSIM on a held-out test set.

**Phase 3: Integrate into Blender**

1. Export the model to ONNX and run it with ONNX Runtime inside the add-on, so users don't need PyTorch.
2. Add a panel or compositor-adjacent operator: pick a render result, click Denoise, get the output as a new image.
3. Handle the practical bits: tiling large images, GPU vs CPU fallback, and bundling dependencies.

**Phase 4: Benchmark and publish**

1. Compare against OpenImageDenoise on the same inputs: quality (PSNR/SSIM) and speed.
2. Publish the code on GitHub with a clear README, before/after images, and a short demo video.
3. Write a blog post on what you learned, including where your model loses and why.

**Stretch goals:** temporal denoising for animations (using the previous frame), or optimizing inference with TensorRT or half precision. Given your compositor work, a real-time viewport version would be a standout.

## Project 2: Differentiable rendering

The goal is inverse rendering: given a target photo or render, recover scene parameters (lighting, material, or geometry) by running gradient descent through the renderer. Start with an existing differentiable renderer, then build one small piece yourself so the project shows real understanding, not just library calls.

**Phase 1: Learn the foundations**

1. Work through the Mitsuba 3 tutorials on inverse rendering, and skim the PyTorch3D rendering examples.
2. Understand why rendering is hard to differentiate: visibility edges, sampling noise, and the discontinuities at object silhouettes.

**Phase 2: First inverse problem**

1. Build a simple scene in Blender, render a target image, and export the scene to Mitsuba.
2. Perturb one parameter (e.g. a light's position or a material's color) and recover it by optimizing against the target.
3. Plot loss over iterations and show the parameter converging. This is your first demo.

**Phase 3: Build something of your own**

Pick one:

- Write a minimal differentiable rasterizer or ray tracer in PyTorch or JAX for simple shapes. This is the most impressive option.
- Recover a full material (roughness, albedo texture) from a few photos of a real object.
- Optimize a mesh's shape to match silhouettes from multiple views.

**Phase 4: Close the loop with Blender**

1. Write an add-on or script that exports a Blender scene, runs the optimization, and imports the recovered parameters back into Blender.
2. This ties the project to your existing expertise and makes a strong demo: photo in, editable Blender scene out.

**Phase 5: Publish**

1. GitHub repo with a README, convergence videos, and a technical write-up.
2. Optionally, a short blog post explaining differentiable rendering for developers. Clear explanations of hard topics get noticed.

**Tools:** Mitsuba 3 and Dr.Jit, PyTorch3D, nvdiffrast, PyTorch or JAX.

## Suggested timeline

At roughly 8 to 10 hours a week, both projects fit in about six months, with a public release at the end of each.

| Month | Project | Phase |
|---|---|---|
| 1 | Denoising add-on | Phase 1: Generate a dataset |
| 2 | Denoising add-on | Phase 2: Train a model |
| 3 | Denoising add-on | Phase 3: Integrate into Blender, Phase 4: Benchmark and publish |
| 4 | Differentiable rendering | Phase 1: Learn the foundations, Phase 2: First inverse problem |
| 5 | Differentiable rendering | Phase 3: Build something of your own |
| 6 | Differentiable rendering | Phase 4: Close the loop with Blender, Phase 5: Publish |

Publishing the add-on at month 3 gives you something on your resume early, while the flagship is still in progress. Stretch goals come after month 6.

## Turning it into resume and portfolio material

Each project earns one or two resume bullets that name the problem, what you built, and a measured result. Example shapes to fill in once you have numbers:

- Built a neural denoising add-on for Blender: trained a U-Net on X rendered image pairs, reaching Y dB PSNR at Z ms per frame via ONNX Runtime.
- Implemented inverse rendering with Mitsuba 3, recovering lighting and materials from target images; built a Blender round-trip pipeline.
- Contributed a merged fix to Blender's animation rigging module (add the release version it shipped in).

Beyond the resume:

- Pin both repos on GitHub, each with a GIF or video at the top of the README.
- Share demos on the Blender forums (devtalk.blender.org), Blender Artists, and X or LinkedIn. Graphics demos spread well visually.
- Keep contributing to Blender; the compositor work is a natural bridge to GPU and ML topics.
