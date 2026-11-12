"""Image quality metrics, scored on what a user actually sees.

Every metric takes LINEAR radiance tensors (N, 3, H, W), applies the same simple
display transform (tonemap) to both images, and returns one value per image (N,).
Use the same functions for the noisy input, our output and other denoisers (OIDN in
Phase 4), so all numbers are comparable.

    python models/metrics.py     # runs a few self-checks
"""

import math

import torch
import torch.nn.functional as F


def tonemap(x):
    """Linear radiance -> display values 0..1: clip highlights at 1, then gamma 2.2.

    For metrics and previews only. Its gradient is infinite at 0, so for a loss in
    display space use tonemap(x + 1e-4) or a smooth curve like x / (1 + x)."""
    return x.clamp(0, 1) ** (1 / 2.2)


def psnr(pred, target):
    """Peak signal-to-noise ratio in dB (higher is better; +3 dB = half the error).
    Identical images give inf."""
    mse = (tonemap(pred) - tonemap(target)).pow(2).mean(dim=(1, 2, 3))
    return 10 * torch.log10(1.0 / mse)


def _gaussian_window(size=11, sigma=1.5, channels=3, device=None):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]  # 2D Gaussian, sums to 1
    return window.expand(channels, 1, size, size).contiguous()


def ssim(pred, target):
    """Structural similarity (Wang et al. 2004), 0..1, higher is better. It compares
    local brightness, contrast and structure in 11x11 Gaussian windows, per channel,
    and averages over the image. Matches skimage's structural_similarity with
    gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1."""
    x, y = tonemap(pred).float(), tonemap(target).float()
    window = _gaussian_window(channels=x.shape[1], device=x.device)
    blur = lambda img: F.conv2d(img, window, groups=img.shape[1])  # noqa: E731, valid area only
    c1, c2 = 0.01**2, 0.03**2  # stabilisers for data range 1
    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x**2
    var_y = blur(y * y) - mu_y**2
    cov = blur(x * y) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return ssim_map.mean(dim=(1, 2, 3))


def _self_check():
    torch.manual_seed(0)
    gt = torch.rand(2, 3, 64, 64) * 0.8  # low enough that gt + the shift below isn't clipped
    assert torch.isinf(psnr(gt, gt)).all(), "identical images should give inf PSNR"
    assert torch.allclose(ssim(gt, gt), torch.ones(2)), "identical images should give SSIM 1"

    # Known error: shift the display image by d everywhere -> MSE = d^2, PSNR = -20 log10(d)
    d = 0.05
    shifted = (tonemap(gt) + d) ** 2.2
    expected = -20 * math.log10(d)
    assert torch.allclose(psnr(shifted, gt), torch.full((2,), expected), atol=1e-3)

    noisy = gt + 0.2 * torch.randn_like(gt)
    less_noisy = gt + 0.05 * torch.randn_like(gt)
    assert (psnr(less_noisy, gt) > psnr(noisy, gt)).all()
    assert (ssim(less_noisy, gt) > ssim(noisy, gt)).all()
    print(f"metrics OK: PSNR of a {d} shift = {psnr(shifted, gt)[0]:.2f} dB (expected {expected:.2f}), "
          f"SSIM noisy {ssim(noisy, gt)[0]:.3f} < less noisy {ssim(less_noisy, gt)[0]:.3f}")


if __name__ == "__main__":
    _self_check()
