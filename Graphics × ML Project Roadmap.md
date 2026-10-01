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

&#91;embedded content: suggested timeline · 2 projects, 9 phases\]

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
