# Phase 2: training the denoiser

Goal (roadmap Phase 2): train a small U-Net in PyTorch that turns a noisy, low-sample Cycles render plus its albedo and normal passes into a clean image, and measure it with PSNR and SSIM on scenes it has never seen. Phase 3 then exports it to ONNX for the Blender add-on.

The work is split:

| Part | Who | Status |
|---|---|---|
| EXR reading, train/val/test splits, datasets, metrics, data checks, evaluation | Claude | **done**, tested (see "Checks done") |
| The U-Net, `models/unet.py` | **you** | to do: spec in "Task 1" |
| The training loop, `models/train.py` | **you** | to do: spec in "Task 2" |
| Review of your code and first training curves | Claude | after you've written them |

## Quick start

All commands run from the project folder with the project's Python, `.venv\Scripts\python.exe`, shortened to `python` below.

```powershell
python models/make_splits.py      # already done; re-run only if the dataset changes
python models/check_dataset.py    # sanity checks + loader speed + preview PNG
python models/train.py            # yours (Task 2)
tensorboard --logdir runs         # watch training at http://localhost:6006
python models/evaluate.py --checkpoint models/checkpoints/unet_v1/best.pth --split test
```

## Files

| File | What it does |
|---|---|
| `exr_io.py` | `read_exr(path)` → float32 `(H, W, C)` numpy; `write_exr`, `write_png` |
| `make_splits.py` | Splits the dataset by scene and writes `data/dataset/splits.json` |
| `dataset.py` | `TrainCrops`, `FullFrames`, `collate_crops`, and the transforms `to_input`, `to_target`, `from_output` |
| `metrics.py` | `tonemap`, `psnr`, `ssim`; run it directly for a self-check |
| `check_dataset.py` | Checks shapes, alignment and flips, measures loader speed, saves `check_dataset.png` |
| `evaluate.py` | Scores a checkpoint on val or test per spp level; writes CSV, summary and images |
| `unet.py` | **yours**: the model |
| `train.py` | **yours**: the training loop |

Outputs (all git-ignored):
- `runs/<run>/`: TensorBoard logs.
- `models/checkpoints/<run>/`: `last.pth` and `best.pth`.
- `models/eval/<name>/`: evaluation results.
- `models/trained/`: put the final model here.

## Data

### Splits (`data/dataset/splits.json`)

Split **by scene**: every frame of a scene is in the same split, so val and test scores measure how well the model does on scenes it has never seen.

| Split | Scenes | Frames | Pairs |
|---|---|---|---|
| train | 6 demo scenes (barbershop_interior, benchmark, blender-2.81-splash, blender-283, blender-4.1-splash, blender3.1-splash) + 80 layouts | 2,100 | 8,400 |
| val | blender-4.3-splash + 10 layouts | 300 | 1,200 |
| test | **classroom** + 10 layouts | 300 | 1,200 |

- **Layout choice:** the val and test layouts are drawn with a fixed seed (`SEED` in `make_splits.py`).
- **Excluded:** blender-3.3-splash and blender-5.2-splash have no frames.
- **Use of each split:**
  - train is for learning;
  - val is for choosing between models and settings;
  - test is touched **only for the final numbers**, so they stay honest.

### What a sample looks like

Each frame folder has `noisy_{1,4,8,16}spp.exr`, `gt.exr` (2048 spp), `albedo.exr`, `normal.exr` and `depth.exr`. Each frame gives 4 training pairs, one per spp level, all sharing one gt.

- **Input:** 9 channels, made by `to_input`:

  | Channels | Content | Transform | Range |
  |---|---|---|---|
  | 0–2 | noisy RGB | `log(1 + x)` | 0 to about 8 |
  | 3–5 | albedo RGB | clamp | 0–1 |
  | 6–8 | normal XYZ | clamp | −1 to 1 |

- **Target:** `log(1 + gt)`, made by `to_target`.
- **Model output:** in the same log space. `from_output(y) = exp(y) − 1` (after clamping to ≥ 0) turns it back into radiance.
- **Why log:** radiance goes from 0 (shadows) to thousands (a lamp). In linear space the loss would be dominated by the few brightest pixels; log space treats dark and bright regions more evenly.

Things to know about the passes:
- **Shared aux passes:** albedo and normal come from the 16 spp render and are reused for all 4 spp levels.
- **Camera-space normals:** X points right and Y up, so a mirrored image needs its normal X negated. `augment` does that.
- **Depth isn't used yet;** adding it is experiment 2.

### Training samples (`TrainCrops`)

- **One item = one frame:** a random spp level of that frame is loaded, then 8 random 128×128 crops are cut from it. Item shapes are `(8, 9, 128, 128)` input and `(8, 3, 128, 128)` target.
- **Batching:** use `collate_fn=collate_crops`, so a batch of 2 frames gives `(16, 9, 128, 128)`.
- **Augmentation**, applied to each crop:

  | Change | Applied to | Why |
  |---|---|---|
  | Random exposure ×2^U(−1.5, 1.5) | noisy and gt | Handle brighter and darker renders |
  | Random RGB channel order | noisy, gt and albedo | More colour variety |
  | Horizontal flip (normal X negated) | everything | More layouts of the same content |

- **Epochs:** an epoch is the 2,100 training frames once each. Since every visit uses a new spp level, new crops and new augmentation, there's no real "end of the data", so train by **steps**, not epochs.

### Validation / test samples (`FullFrames`)

One item per (frame, spp level), whole 512×512 frames. Each item is a dict:
- `input` (9, 512, 512) and `target` (3, 512, 512);
- `noisy` and `gt`: linear radiance (3, 512, 512), for the metrics;
- `scene`, `frame`, `spp`.

`FullFrames("val", num_frames=12)` gives a fixed 12-frame × 4-spp subset (48 images), the same every time. That's quick enough to run during training.

## Metrics (`metrics.py`)

All metrics compare **what a user sees**:
1. Both images go through `tonemap(x) = clamp(x, 0, 1) ^ (1/2.2)`.
2. **PSNR** (dB, higher is better; +3 dB means half the squared error) is computed on the result.
3. So is **SSIM** (0–1, higher is better; it compares local structure, using 11×11 Gaussian windows).

Every function takes **linear** images `(N, 3, H, W)` and returns one score per image `(N,)`, so use `from_output` on the model output first. The same functions score the noisy input, our output and, in Phase 4, OIDN, so the numbers are directly comparable. Both match scikit-image exactly.

## Speed (measured on this laptop)

| What | Speed |
|---|---|
| Loader, 8 workers, 8 crops per frame | about 580 crops/s (36 batches of 16/s) |
| Loader, 8 workers, 4 crops per frame | about 375 crops/s (too slow: the GPU would wait) |
| Training a 1.0M-parameter U-Net (the Task 1 spec), batch 16, fp16 | about 610 crops/s (38 steps/s), 0.4 GB VRAM |
| Inference, one 512×512 frame, fp16 | about 9 ms |

So with `num_workers=8`, 2 frames per batch and the default 8 crops per frame, the loader roughly keeps up with the GPU. Expect about 35 steps/s, so **100k steps take about 50 min**.

## Task 1: the U-Net (`models/unet.py`)

Write a class `UNet(nn.Module)`:
- **Constructor:** `__init__(self, in_channels=9, out_channels=3, ...)`. Any extra arguments must be saved under `config["model"]` (Task 2), because `evaluate.py` builds the model with `UNet(**config["model"])`.
- **Forward:** `forward(x)`, input `(N, 9, H, W)` → output `(N, 3, H, W)` in **log space**. Don't put an activation on the output.
- **Size:** H and W must be multiples of 16. Training crops (128) and frames (512) are; `evaluate.py` pads other sizes.

### Architecture

A **U-Net** encodes the image down to a small, wide representation, then decodes it back up. At each level it concatenates ("skips") the encoder's features of the same size, so fine detail isn't lost. "Conv" below is `3×3 conv (padding 1) → ReLU`.

```
input 9 x 128 x 128
 enc1: conv 9->32,   conv 32->32     -----------------------------> skip1 (32 x 128^2)
   max-pool 2x2  -> 64^2
 enc2: conv 32->48,  conv 48->48     ----------------------> skip2 (48 x 64^2)
   max-pool      -> 32^2
 enc3: conv 48->64,  conv 64->64     ---------------> skip3 (64 x 32^2)
   max-pool      -> 16^2
 enc4: conv 64->96,  conv 96->96     --------> skip4 (96 x 16^2)
   max-pool      -> 8^2
 bottleneck: conv 96->128, conv 128->128          (8 x 8)
   upsample x2 -> 16^2, concat skip4 (128+96)  -> conv 224->96, conv 96->96
   upsample x2 -> 32^2, concat skip3 (96+64)   -> conv 160->64, conv 64->64
   upsample x2 -> 64^2, concat skip2 (64+48)   -> conv 112->48, conv 48->48
   upsample x2 -> 128^2, concat skip1 (48+32)  -> conv 80->32,  conv 32->32
 output: 1x1 conv 32->3      (no activation)
```

The sizes are halved 4 times (128 → 8), which is why sides must be multiples of 2⁴ = 16. With these widths the model has **about 1.0M parameters**.

Hints:
- **Building blocks:** `nn.Conv2d(cin, cout, 3, padding=1)` keeps the size; `F.max_pool2d(x, 2)` halves it; `F.interpolate(x, scale_factor=2, mode="nearest")` doubles it; `torch.cat([a, b], dim=1)` concatenates channels.
- **Repeated levels:** a helper `def double_conv(cin, cout)` returning an `nn.Sequential`, plus `nn.ModuleList`s for the encoder and decoder levels, keeps the code short. Making the widths a constructor argument (e.g. `widths=(32, 48, 64, 96)`) makes experiment 3 easy.
- **Phase 3 compatibility:** only these standard layers, so the model exports to ONNX. Avoid Python logic that depends on tensor values inside `forward`.

Test it before training:

```python
import torch
from unet import UNet
m = UNet()
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")  # about 1.0
print(m(torch.randn(2, 9, 128, 128)).shape)  # torch.Size([2, 3, 128, 128])
print(m(torch.randn(1, 9, 512, 512)).shape)  # torch.Size([1, 3, 512, 512])
```

## Task 2: the training loop (`models/train.py`)

### Settings

Keep the settings in one dict at the top. It's saved in every checkpoint:

```python
CONFIG = {
    "run": "unet_v1",            # names runs/<run> and models/checkpoints/<run>
    "steps": 100_000,
    "frames_per_batch": 2,       # x 8 crops per frame = 16 crops per step
    "lr": 1e-3, "lr_min": 1e-5, "warmup": 1_000,
    "grad_clip": 1.0,
    "val_every": 2_000,
    "workers": 8,
    "seed": 0,
    "model": {"in_channels": 9, "out_channels": 3},   # passed to UNet(**...)
}
```

### Outline

1. **Windows guard:** everything goes inside `def main()`, called under `if __name__ == "__main__":`. On Windows, DataLoader workers re-import the file, and without the guard each worker would start training too.
2. **Seed:** `torch.manual_seed(CONFIG["seed"])`.
3. **Data:**
   ```python
   DataLoader(TrainCrops(), batch_size=CONFIG["frames_per_batch"], shuffle=True,
              num_workers=CONFIG["workers"], collate_fn=collate_crops,
              persistent_workers=True, pin_memory=True, drop_last=True)
   ```
   Wrap it in a generator that loops over the loader forever, since training goes by steps:
   ```python
   def forever(loader):
       while True:
           yield from loader
   ```
   - The validation set is `FullFrames("val", num_frames=12)`, with a `DataLoader(batch_size=1, num_workers=2)`.
4. **Model and optimizer:**
   - `model = UNet(**CONFIG["model"]).cuda()`, `opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])` and `scaler = torch.amp.GradScaler()`.
   - **Learning rate:** linear warm-up from 0 to `lr` over `warmup` steps, then a cosine decay down to `lr_min` at `steps`:
     ```
     s < warmup:  lr * s / warmup
     otherwise:   lr_min + 0.5 * (lr - lr_min) * (1 + cos(pi * (s - warmup) / (steps - warmup)))
     ```
     Implement it as a function `lr_at(step)`, and set it each step: `for g in opt.param_groups: g["lr"] = lr_at(step)`. Or use `LambdaLR`.
5. **Resume:** if `models/checkpoints/<run>/last.pth` exists, load the model, optimizer, scaler and step from it, and continue from there.
6. **One training step:**
   ```
   x, y = next(batches); move both to the GPU (non_blocking=True)
   with torch.autocast("cuda", dtype=torch.float16):
       loss = L1 between model(x) and y          # both in log space
   opt.zero_grad(set_to_none=True)
   scaler.scale(loss).backward()
   scaler.unscale_(opt)                          # so clipping sees the real gradients
   torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
   scaler.step(opt); scaler.update()
   ```
   - **fp16 autocast** runs the model in half precision (faster, less memory).
   - **GradScaler** scales the loss up so small fp16 gradients don't round to zero.
   - **Clipping** stops a rare huge gradient (e.g. from a firefly) from wrecking the weights.
7. **Logging:** about every 100 steps, write `train/loss` and `train/lr` to TensorBoard (`torch.utils.tensorboard.SummaryWriter(f"runs/{run}")`) and print the step, loss and steps/s.
8. **Validation, every `val_every` steps:**
   1. Switch to `model.eval()` inside `torch.no_grad()`.
   2. For each val item, compute `ours = from_output(model(input).float())`, then `psnr(ours, gt)` and `ssim(ours, gt)`.
   3. Average them **per spp level**, and log `val/psnr_1spp` … `val/psnr_16spp`, the same for SSIM, and `val/psnr_mean`.
   4. Log one image with `writer.add_image("val/noisy_ours_gt", torch.cat([tonemap(noisy), tonemap(ours), tonemap(gt)], dim=2), step)`. That's `(3, H, 3W)` with values 0..1, which is useful for spotting blur or colour shifts.
   5. Switch back to `model.train()`.
9. **Checkpoints:**
   - After each validation, save `last.pth`. Also save `best.pth` if `val/psnr_mean` is the best so far.
   - Format, which `evaluate.py` relies on:
     ```python
     {"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
      "step": step, "config": CONFIG, "best_psnr": best}
     ```

### Training runs

1. **Smoke test** (`steps=2_000`, `val_every=500`, about 1 min): the loss should drop clearly in the first few hundred steps, and val PSNR should rise above the noisy baseline. If not, check the data flow before running longer.
2. **Short run** (`steps=20_000`, about 10 min): curves are smooth and val keeps improving. Look at the preview images.
3. **Main run** (`steps=100_000`–`200_000`, about 1–2 h): the baseline model `unet_v1`. Then copy `best.pth` to `models/trained/unet_v1.pth`.

What to watch in TensorBoard:
- **Train loss** falling fast and then slowly is normal.
- **Val PSNR flattening** while the train loss still falls means the model is starting to memorise; more steps won't help.
- **1 spp improves the most** (most noise to remove). 16 spp improves the least, but must still beat its noisy input.

### Scores to beat (noisy input, no denoising)

These are the scores of the noisy input itself, measured with `evaluate.py` and a stand-in model that returns its input unchanged. Your model must beat them. Each cell is PSNR dB / SSIM.

| spp | val: all | val: blender-4.3-splash | val: layouts | test: all | test: classroom | test: layouts |
|---|---|---|---|---|---|---|
| 1 | 13.36 / 0.203 | 14.88 / 0.311 | 11.84 / 0.094 | 10.18 / 0.078 | 9.43 / 0.069 | 10.92 / 0.087 |
| 4 | 23.44 / 0.522 | 26.06 / 0.666 | 20.82 / 0.378 | 16.81 / 0.259 | 14.09 / 0.151 | 19.53 / 0.366 |
| 8 | 27.11 / 0.678 | 30.13 / 0.836 | 24.10 / 0.521 | 20.05 / 0.366 | 17.15 / 0.231 | 22.94 / 0.502 |
| 16 | 30.00 / 0.783 | 32.97 / 0.922 | 27.04 / 0.643 | 23.45 / 0.485 | 20.81 / 0.347 | 26.10 / 0.623 |

- **Classroom is hard:** it's lit mostly by bounced light, so even at 16 spp it's very noisy (20.8 dB). Expect the biggest gains there, and also the hardest cases.
- **The scale of improvement:** a good denoiser typically adds 10+ dB at low spp. OIDN is the reference in Phase 4.
- **Val vs test:** val is easier than test, so don't expect test numbers to match val.

## Experiments after the baseline

Change **one thing at a time**, train with the same steps, and compare val PSNR/SSIM per spp with `evaluate.py --split val`. Keep a short log of what you tried and the numbers; it becomes the blog post.

1. **Perceptual loss:** add `w × L1(VGG features of ours, VGG features of gt)` to the loss, with `w` around 0.05–0.1.
   - Features: frozen `torchvision.models.vgg16(weights="DEFAULT").features[:16]` (up to relu3_3), downloaded once (about 0.5 GB).
   - Feed it smoothly tonemapped images (`x / (1 + x)` of the linear images), normalised with the ImageNet mean/std.
   - It usually makes textures look sharper, even if PSNR barely moves.
2. **Depth input:** add `depth.exr` as a 10th channel, normalised per image (e.g. `depth / median(depth)`, clamped). This needs changes in `dataset.py`; ask me if you want this one.
3. **Width:** try `widths=(48, 64, 96, 128)` (about 2× the parameters). Is the quality gain worth the slower inference in Phase 3?
4. **Telling the model the noise level:** add a constant channel `log2(spp) / 4` to the input. The model then doesn't have to guess how noisy the input is.
5. **Loss space:** L1 on `x / (1 + x)` of the linear images instead of on log space.

## Evaluation (`evaluate.py`)

```powershell
python models/evaluate.py --checkpoint models/checkpoints/unet_v1/best.pth --split val
python models/evaluate.py --checkpoint models/trained/unet_v1.pth --split test --amp --save-exr
```

It writes to `models/eval/<checkpoint>_<split>/`:
- `metrics.csv`: one row per image, with the scene, frame, spp, PSNR/SSIM for noisy and ours, and the time;
- `summary.md`: means per spp for all scenes, demo scenes and layouts;
- `images/`: `noisy | ours | gt` PNGs for the first 8 frames;
- `exr/`: our outputs, with `--save-exr` (needed in Phase 4 to compare with OIDN).

Useful options:
- `--num-frames N` uses a quick subset;
- `--amp` runs in fp16;
- `--model-file` points at a different model file.

## Phase 2 is done when

- [ ] `models/trained/unet_v1.pth` exists, trained with the main run.
- [ ] `evaluate.py --split test` shows **ours beating noisy at every spp**, on both PSNR and SSIM, for demo scenes and layouts.
- [ ] The comparison images look right: no colour shifts, no blotches, edges kept.
- [ ] Inference time per 512×512 frame is noted (from `summary.md`).
- [ ] At least the perceptual-loss experiment is tried and compared on val.

That gives you the resume line's numbers: "trained a U-Net on 8,400 rendered pairs, reaching X dB PSNR at Y ms per frame".

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Training starts several times, or `RuntimeError ... bootstrapping phase` | Missing `if __name__ == "__main__":` (Windows DataLoader workers) |
| Loss is `nan` | `scaler` not used, or the learning rate is too high; try `lr=5e-4` |
| Loss stuck from the start | Output has an activation (e.g. ReLU or sigmoid), the learning rate isn't set, or `x`/`y` are swapped |
| Output too dark or too bright | Forgot `from_output` before the metrics, or applied it twice |
| GPU usage low (Task Manager → GPU → Cuda) | Fewer than 8 workers; or another program is reading the disk |
| Out of memory | Unlikely at 0.4 GB; check you aren't keeping tensors (e.g. `losses.append(loss)` instead of `loss.item()`) |

## Checks done (plumbing)

- **`make_splits.py`:** each scene is in one split only, and classroom is only in test.
- **`check_dataset.py`:**
  - shapes and value ranges are correct, with no NaN or inf;
  - noisy and gt crops line up (correlation 0.985, vs 0.863 when shifted by 8 px);
  - the flip negates normal X;
  - the preview PNG shows matching noisy, albedo, normal and gt crops.
- **`metrics.py`:** the self-check passes, and PSNR and SSIM match scikit-image to 6 digits.
- **`evaluate.py`:**
  - with a stand-in identity model (output = noisy input), "ours" equals the noisy scores exactly;
  - with an untrained spec-sized U-Net: checkpoint loading, padding to multiples of 16 and fp16 inference all run.
