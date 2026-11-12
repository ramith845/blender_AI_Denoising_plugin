"""Check the data pipeline before training, and measure how fast it loads.

    python models/check_dataset.py

1. Shapes and value ranges of a training item.
2. Crops line up: the noisy and gt crops show the same pixels.
3. A horizontal flip mirrors the normals and negates their X component.
4. Loading speed with 0 / 4 / 8 DataLoader workers (crops per second).
5. Saves models/check_dataset.png: rows of crops showing noisy | albedo | normal | gt.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FullFrames, TrainCrops, augment, collate_crops, frames_of, load_frame
from exr_io import write_png
from metrics import tonemap

OUT = Path(__file__).resolve().parent / "check_dataset.png"


def check_item():
    ds = TrainCrops()
    x, y = ds[0]
    print(f"train frames: {len(ds)}; one item: input {tuple(x.shape)}, target {tuple(y.shape)}")
    groups = {"log noisy": x[:, 0:3], "albedo": x[:, 3:6], "normal": x[:, 6:9], "log gt": y}
    for name, t in groups.items():
        print(f"  {name:9s} min {t.min():8.3f}  max {t.max():8.3f}  mean {t.mean():7.3f}")
    assert x.shape[1:] == (9, 128, 128) and y.shape[1:] == (3, 128, 128)
    assert torch.isfinite(x).all() and torch.isfinite(y).all(), "NaN or inf in a sample"


def check_alignment():
    """Noisy (16 spp) and gt of the same area should be strongly correlated; gt shifted
    by 8 pixels should match worse. Averaging 8x8 blocks first removes most of the noise,
    which would otherwise hide the match."""
    scene, frame = frames_of("train")[0]
    images = load_frame(scene, frame, 16)
    noisy, gt = np.log1p(images["noisy"]), np.log1p(images["gt"])

    def blocks(img):  # 256x256 area -> 32x32 block means
        return img.reshape(32, 8, 32, 8, -1).mean(axis=(1, 3)).ravel()

    same = np.corrcoef(blocks(noisy[128:384, 128:384]), blocks(gt[128:384, 128:384]))[0, 1]
    shifted = np.corrcoef(blocks(noisy[128:384, 128:384]), blocks(gt[128:384, 136:392]))[0, 1]
    print(f"alignment ({scene}/{frame}): correlation noisy vs gt {same:.3f}, vs gt shifted 8 px {shifted:.3f}")
    assert same > 0.9 and same > shifted, "noisy and gt crops don't line up"


def check_flip():
    """Force a flip (exposure 0, colour order unchanged) and compare normals."""
    scene, frame = frames_of("train")[0]
    images = load_frame(scene, frame, 16)

    class FlipRng:  # stands in for numpy's generator: no exposure change, same colours, flip
        def uniform(self, low, high):
            return 0.0

        def permutation(self, n):
            return np.arange(n)

        def random(self):
            return 0.0

    flipped = augment(images, FlipRng())
    expected = images["normal"][:, ::-1] * np.array([-1, 1, 1], dtype=np.float32)
    assert np.array_equal(flipped["normal"], expected)
    assert np.array_equal(flipped["gt"], images["gt"][:, ::-1])
    print("flip: normals mirrored with X negated, images mirrored")


def check_speed(batch_frames=4, batches=25):
    """Crops per second the DataLoader delivers (the GPU needs this many to stay busy)."""
    ds = TrainCrops()
    for workers in (0, 4, 8):
        loader = DataLoader(ds, batch_size=batch_frames, shuffle=True, num_workers=workers,
                            collate_fn=collate_crops, persistent_workers=workers > 0)
        it = iter(loader)
        next(it)  # warm-up: workers start and open files
        start = time.perf_counter()
        crops = 0
        for _ in range(batches):
            x, _ = next(it)
            crops += x.shape[0]
        rate = crops / (time.perf_counter() - start)
        print(f"loader with {workers} workers: {rate:6.0f} crops/s ({rate / x.shape[0]:5.1f} batches of {x.shape[0]}/s)")
        del it, loader


def save_preview(rows=6):
    ds = TrainCrops()
    x, y = ds[1]
    tiles = []
    for k in range(min(rows, x.shape[0])):
        noisy = tonemap(torch.expm1(x[k, 0:3]))
        albedo = x[k, 3:6] ** (1 / 2.2)
        normal = x[k, 6:9] * 0.5 + 0.5  # -1..1 shown as 0..1
        gt = tonemap(torch.expm1(y[k]))
        tiles.append(torch.cat([noisy, albedo, normal, gt], dim=2))
    grid = torch.cat(tiles, dim=1).permute(1, 2, 0).numpy()
    write_png(OUT, grid)
    print(f"preview: {OUT} (columns: noisy | albedo | normal | gt)")


def check_val():
    ds = FullFrames("val", num_frames=12)
    item = ds[0]
    print(f"val subset: {len(ds)} images (12 frames x 4 spp); one: input {tuple(item['input'].shape)}, "
          f"{item['scene']}/{item['frame']} at {item['spp']} spp")


if __name__ == "__main__":  # needed on Windows: DataLoader workers re-import this file
    check_item()
    check_alignment()
    check_flip()
    check_val()
    save_preview()
    check_speed()
