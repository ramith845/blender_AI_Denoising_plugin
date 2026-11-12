"""PyTorch datasets for the denoiser.

Two datasets read the frames listed in data/dataset/splits.json (make_splits.py):

- TrainCrops: for training. Each item is K random 128x128 crops from one random-spp
  render of a frame, with augmentation. Shapes (K, 9, 128, 128) and (K, 3, 128, 128);
  use collate_crops in the DataLoader to get (batch * K, ...).
- FullFrames: for validation and testing. One item per (frame, spp level), the whole
  512x512 image, plus the linear noisy and gt images for the metrics.

The model never sees raw radiance: to_input / to_target map it to log space, and
from_output maps the model's output back (see models/README.md, "Data").
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from exr_io import read_exr

DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset"
SPP_LEVELS = (1, 4, 8, 16)
IN_CHANNELS = 9  # noisy RGB + albedo RGB + normal XYZ
OUT_CHANNELS = 3


# ---------------------------------------------------------------------------
# Transforms between radiance and what the network sees
# ---------------------------------------------------------------------------


def to_input(noisy, albedo, normal):
    """Network input, (9, H, W) float32 tensor, from (H, W, 3) float arrays.

    Radiance spans 0 to thousands (lamps, sky), so it's squashed with log(1 + x).
    Albedo is already 0..1 and normals -1..1, so they're only clamped."""
    x = np.concatenate(
        [np.log1p(np.maximum(noisy, 0)), np.clip(albedo, 0, 1), np.clip(normal, -1, 1)],
        axis=2,
    )
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def to_target(gt):
    """Training target, (3, H, W): the clean image in the same log space as the input."""
    return torch.from_numpy(np.ascontiguousarray(np.log1p(np.maximum(gt, 0)).transpose(2, 0, 1)))


def from_output(y):
    """Model output (log space, torch) back to linear radiance. Negative outputs mean
    'less than black', so they're clamped to 0 first."""
    return torch.expm1(y.clamp(min=0))


# ---------------------------------------------------------------------------
# Loading frames
# ---------------------------------------------------------------------------


def load_splits():
    path = DATASET / "splits.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run models/make_splits.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def frames_of(split):
    """[(scene, frame folder)] of every complete frame in a split ('train', 'val', 'test')."""
    info = load_splits()
    return [(scene, frame) for scene in info["splits"][split] for frame in info["frames"][scene]]


def load_frame(scene, frame, spp):
    """One frame's images as float32 (H, W, C) arrays: noisy at `spp`, gt, albedo, normal."""
    folder = DATASET / scene / frame
    return {
        "noisy": read_exr(folder / f"noisy_{spp}spp.exr"),
        "gt": read_exr(folder / "gt.exr"),
        "albedo": read_exr(folder / "albedo.exr"),
        "normal": read_exr(folder / "normal.exr"),
    }


# ---------------------------------------------------------------------------
# Augmentation (training only)
# ---------------------------------------------------------------------------


def augment(images, rng, exposure_stops=1.5):
    """Random variations that don't change what the right answer is:

    - exposure: noisy and gt scaled by the same 2^U(-stops, stops), so the model
      handles brighter and darker renders (albedo and normals don't change with light)
    - colour: the same random order of R, G, B for noisy, gt and albedo
    - horizontal flip: everything mirrored. Normals are in camera space, so their
      X component (pointing right) must change sign too.
    """
    images = dict(images)
    scale = np.float32(2.0 ** rng.uniform(-exposure_stops, exposure_stops))
    images["noisy"] = images["noisy"] * scale
    images["gt"] = images["gt"] * scale

    order = rng.permutation(3)
    for key in ("noisy", "gt", "albedo"):
        images[key] = images[key][..., order]

    if rng.random() < 0.5:
        images = {key: img[:, ::-1] for key, img in images.items()}
        images["normal"] = images["normal"] * np.array([-1, 1, 1], dtype=np.float32)
    return images


def random_crop(images, size, rng):
    """The same random size x size window from every image."""
    height, width = images["gt"].shape[:2]
    top = rng.integers(0, height - size + 1)
    left = rng.integers(0, width - size + 1)
    return {key: img[top : top + size, left : left + size] for key, img in images.items()}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class TrainCrops(Dataset):
    """Random training crops. One epoch = every frame of the split once, each time at a
    random spp level, giving `crops_per_frame` crops (reading the 4 EXRs of a frame is
    the slow part, so each read is used several times)."""

    # 8 crops per frame read: measured ~580 crops/s with 8 workers, about what the GPU
    # trains on (README, "Speed"); with 4 it was ~375 and the GPU waited on data
    def __init__(self, split="train", crop_size=128, crops_per_frame=8, augment=True):
        self.frames = frames_of(split)
        self.crop_size = crop_size
        self.crops_per_frame = crops_per_frame
        self.augment = augment
        self._calls = 0

    def __len__(self):
        return len(self.frames)

    def _rng(self, index):
        # torch.initial_seed() differs per DataLoader worker and per epoch, so every
        # worker draws different crops; the call counter covers num_workers=0.
        self._calls += 1
        return np.random.default_rng([torch.initial_seed() % 2**63, index, self._calls])

    def __getitem__(self, index):
        rng = self._rng(index)
        scene, frame = self.frames[index]
        spp = SPP_LEVELS[rng.integers(len(SPP_LEVELS))]
        images = load_frame(scene, frame, spp)
        inputs, targets = [], []
        for _ in range(self.crops_per_frame):
            crop = random_crop(images, self.crop_size, rng)
            if self.augment:
                crop = augment(crop, rng)
            inputs.append(to_input(crop["noisy"], crop["albedo"], crop["normal"]))
            targets.append(to_target(crop["gt"]))
        return torch.stack(inputs), torch.stack(targets)


def collate_crops(batch):
    """DataLoader collate_fn for TrainCrops: joins the crops of all frames into one
    batch, giving inputs (batch * K, 9, S, S) and targets (batch * K, 3, S, S)."""
    inputs, targets = zip(*batch)
    return torch.cat(inputs), torch.cat(targets)


class FullFrames(Dataset):
    """Whole frames for validation / testing: one item per (frame, spp level).

    num_frames picks a fixed random subset of the split's frames (the same every
    time, e.g. for quick validation during training); None uses all of them."""

    def __init__(self, split="val", spp_levels=SPP_LEVELS, num_frames=None, seed=0):
        frames = frames_of(split)
        if num_frames is not None and num_frames < len(frames):
            frames = sorted(random.Random(seed).sample(frames, num_frames))
        self.items = [(scene, frame, spp) for scene, frame in frames for spp in spp_levels]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        scene, frame, spp = self.items[index]
        images = load_frame(scene, frame, spp)
        return {
            "input": to_input(images["noisy"], images["albedo"], images["normal"]),
            "target": to_target(images["gt"]),
            # Linear images, for the metrics (models/metrics.py)
            "noisy": torch.from_numpy(np.ascontiguousarray(images["noisy"].transpose(2, 0, 1))),
            "gt": torch.from_numpy(np.ascontiguousarray(images["gt"].transpose(2, 0, 1))),
            "scene": scene,
            "frame": frame,
            "spp": spp,
        }
