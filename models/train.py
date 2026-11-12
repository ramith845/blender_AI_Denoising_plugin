import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import FullFrames, TrainCrops, collate_crops, from_output
from metrics import psnr, ssim, tonemap
from unet import UNet

CONFIG = {
    "run": "unet_v1",  # names runs/<run> and models/checkpoints/<run>
    "steps": 100_000,
    "frames_per_batch": 2,  # x 8 crops per frame = 16 crops per step
    "lr": 1e-3,
    "lr_min": 1e-5,
    "warmup": 1_000,
    "grad_clip": 1.0,
    "val_every": 2_000,
    "workers": 8,
    "seed": 0,
    "model": {"in_channels": 9, "out_channels": 3},  # passed to UNet(**...)
}

LOG_EVERY = 100  # steps between progress prints


def forever(loader):
    """Training goes by steps, not epochs: loop over the loader again and again."""
    while True:
        yield from loader


def main():
    # Step 1: setup
    torch.manual_seed(CONFIG["seed"])
    device = "cuda"
    torch.backends.cudnn.benchmark = True  # input size never changes, so let cuDNN pick the fastest conv algorithms

    # Step 2: data. Each loader item is one frame giving 8 crops, and collate_crops joins
    # the crops of frames_per_batch frames into one batch.
    train_loader = DataLoader(
        TrainCrops(),
        batch_size=CONFIG["frames_per_batch"],
        shuffle=True,
        num_workers=CONFIG["workers"],
        collate_fn=collate_crops,
        persistent_workers=True,  # keep workers alive between epochs (slow to start on Windows)
        pin_memory=True,  # faster host -> GPU copies
        drop_last=True,
    )
    batches = forever(train_loader)

    # Step 3: model, optimizer, gradient scaler
    model = UNet(**CONFIG["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scaler = torch.amp.GradScaler()  # scales the loss up so small fp16 gradients don't round to 0
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Step 4: training loop
    start = 0  # the resume code (step 8) will set this from last.pth
    model.train()
    loss_sum = torch.zeros((), device=device)  # summed on the GPU; .item() every step would stall it
    t0 = time.perf_counter()

    for step in range(start, CONFIG["steps"]):
        x, y = next(batches)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Forward pass in fp16 (faster, less memory). Model output and target are both in
        # log space, so the L1 loss compares them directly.
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.l1_loss(model(x), y)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)  # back to real gradient sizes, so clipping uses the true norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        scaler.step(opt)  # skips the update if the gradients overflowed (inf/nan)
        scaler.update()  # adjusts the scale factor for the next step

        loss_sum += loss.detach()
        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.perf_counter() - t0
            print(f"step {step + 1:>7}  loss {loss_sum.item() / LOG_EVERY:.4f}  "
                  f"{LOG_EVERY / elapsed:.1f} steps/s")
            loss_sum.zero_()
            t0 = time.perf_counter()


if __name__ == "__main__":
    main()
