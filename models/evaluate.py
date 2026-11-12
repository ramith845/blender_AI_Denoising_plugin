"""Score a trained checkpoint on the val or test split, at every spp level.

    python models/evaluate.py --checkpoint models/checkpoints/<run>/best.pth --split test

For every (frame, spp) it runs the model on the whole frame and compares
noisy-vs-gt and ours-vs-gt with PSNR and SSIM (models/metrics.py). Writes to
models/eval/<checkpoint name>_<split>/:

- metrics.csv     one row per image: scene, frame, spp, PSNR/SSIM noisy and ours, time
- summary.md      mean scores per spp level, for all scenes / demo scenes / layouts
- images/         noisy | ours | gt comparison PNGs for the first --images frames
- exr/            our denoised EXRs, with --save-exr (e.g. for Phase 4 comparisons)

The model class is loaded from --model-file (default models/unet.py), which must
define UNet; it's built with the "model" settings saved in the checkpoint's config.
"""

import argparse
import csv
import importlib.util
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import FullFrames, from_output
from exr_io import write_exr, write_png
from metrics import psnr, ssim, tonemap

HERE = Path(__file__).resolve().parent
MULTIPLE = 16  # a U-Net with 4 downsamplings needs sides divisible by 2^4


def load_model(checkpoint_path, model_file, device):
    """Build UNet from model_file with the checkpoint's settings and load its weights."""
    spec = importlib.util.spec_from_file_location("unet_module", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    settings = checkpoint.get("config", {}).get("model", {})
    model = module.UNet(**settings).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("step")


def run_model(model, x, amp):
    """Pad the image to a multiple of 16 (mirroring the border), run the model, crop back."""
    height, width = x.shape[-2:]
    pad_h, pad_w = -height % MULTIPLE, -width % MULTIPLE
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else x
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        y = model(x)
    return y[..., :height, :width].float()


def summarize(rows):
    """Mean of each metric per spp level, for all scenes, demo scenes and layouts."""
    lines = []
    groups = {
        "all scenes": rows,
        "demo scenes": [r for r in rows if not r["scene"].startswith("layout_")],
        "procedural layouts": [r for r in rows if r["scene"].startswith("layout_")],
    }
    for name, group in groups.items():
        if not group:
            continue
        lines += [f"### {name} ({len({(r['scene'], r['frame']) for r in group})} frames)", "",
                  "| spp | PSNR noisy | PSNR ours | gain | SSIM noisy | SSIM ours | ms/frame |",
                  "|---|---|---|---|---|---|---|"]
        by_spp = defaultdict(list)
        for r in group:
            by_spp[r["spp"]].append(r)
        for spp in sorted(by_spp):
            rs = by_spp[spp]
            mean = lambda key: sum(r[key] for r in rs) / len(rs)  # noqa: E731
            lines.append(
                f"| {spp} | {mean('psnr_noisy'):.2f} | {mean('psnr_ours'):.2f} | "
                f"{mean('psnr_ours') - mean('psnr_noisy'):+.2f} | {mean('ssim_noisy'):.4f} | "
                f"{mean('ssim_ours'):.4f} | {mean('ms'):.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Score a checkpoint on val/test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model-file", default=str(HERE / "unet.py"))
    parser.add_argument("--num-frames", type=int, default=None, help="random subset of frames (default: all)")
    parser.add_argument("--images", type=int, default=8, help="frames to save comparison PNGs for")
    parser.add_argument("--save-exr", action="store_true")
    parser.add_argument("--amp", action="store_true", help="run the model in float16 (faster)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, step = load_model(args.checkpoint, args.model_file, device)
    name = f"{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}_{args.split}"
    out = Path(args.out) if args.out else HERE / "eval" / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    if args.save_exr:
        (out / "exr").mkdir(exist_ok=True)

    data = FullFrames(args.split, num_frames=args.num_frames)
    loader = DataLoader(data, batch_size=1, num_workers=args.workers)
    preview_frames = sorted({(s, f) for s, f, _ in data.items})[: args.images]
    rows = []
    print(f"{args.checkpoint} (step {step}) on {args.split}: {len(data)} images", flush=True)

    with torch.no_grad():
        for i, item in enumerate(loader):
            x = item["input"].to(device)
            noisy, gt = item["noisy"].to(device), item["gt"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            ours = from_output(run_model(model, x, args.amp))
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - start) * 1000

            scene, frame, spp = item["scene"][0], item["frame"][0], int(item["spp"][0])
            rows.append({
                "scene": scene, "frame": frame, "spp": spp,
                "psnr_noisy": float(psnr(noisy, gt)[0]), "psnr_ours": float(psnr(ours, gt)[0]),
                "ssim_noisy": float(ssim(noisy, gt)[0]), "ssim_ours": float(ssim(ours, gt)[0]),
                "ms": ms,
            })
            if (scene, frame) in preview_frames:
                row = torch.cat([tonemap(noisy[0]), tonemap(ours[0]), tonemap(gt[0])], dim=2)
                write_png(out / "images" / f"{scene}_{frame}_{spp}spp.png", row.permute(1, 2, 0).cpu().numpy())
            if args.save_exr:
                write_exr(out / "exr" / f"{scene}_{frame}_{spp}spp.exr", ours[0].permute(1, 2, 0).cpu().numpy())
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(data)}", flush=True)

    rows[0]["ms"] = rows[1]["ms"] if len(rows) > 1 else rows[0]["ms"]  # first run includes GPU warm-up
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = f"# {name} (step {step})\n\nPSNR and SSIM on tonemapped images (models/metrics.py).\n\n" + summarize(rows)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"results in {out}")


if __name__ == "__main__":  # needed on Windows: DataLoader workers re-import this file
    main()
