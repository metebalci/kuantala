"""Pure-PyTorch PSNR and SSIM metrics — no new dependencies."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(ref: torch.Tensor, test: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio between two tensors.

    Returns value in dB. Higher is better.
    """
    mse = (ref.float() - test.float()).pow(2).mean()
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(data_range**2 / mse)).item()


def ssim(
    ref: torch.Tensor,
    test: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Compute Structural Similarity Index between two 4D tensors [B,C,H,W].

    Returns value in [0, 1]. Higher is better.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ref = ref.float()
    test = test.float()
    channels = ref.shape[1]

    # Create Gaussian kernel
    kernel_1d = _gaussian_kernel_1d(window_size, sigma, ref.device)
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel = kernel_2d.expand(channels, 1, window_size, window_size)

    pad = window_size // 2

    mu_ref = F.conv2d(ref, kernel, padding=pad, groups=channels)
    mu_test = F.conv2d(test, kernel, padding=pad, groups=channels)

    mu_ref_sq = mu_ref.pow(2)
    mu_test_sq = mu_test.pow(2)
    mu_ref_test = mu_ref * mu_test

    sigma_ref_sq = F.conv2d(ref * ref, kernel, padding=pad, groups=channels) - mu_ref_sq
    sigma_test_sq = F.conv2d(test * test, kernel, padding=pad, groups=channels) - mu_test_sq
    sigma_ref_test = F.conv2d(ref * test, kernel, padding=pad, groups=channels) - mu_ref_test

    numerator = (2 * mu_ref_test + C1) * (2 * sigma_ref_test + C2)
    denominator = (mu_ref_sq + mu_test_sq + C1) * (sigma_ref_sq + sigma_test_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean().item()


def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma**2))
    return g / g.sum()


def compute_metrics_per_frame(
    ref: torch.Tensor, test: torch.Tensor, data_range: float = 1.0
) -> list[dict[str, float]]:
    """Compute PSNR and SSIM per frame.

    For 5D video tensors [B,C,T,H,W], computes metrics per frame.
    For 4D image tensors [B,C,H,W], returns a single-element list.
    """
    if ref.dim() == 5:
        # Video: [B, C, T, H, W] → iterate over T
        num_frames = ref.shape[2]
        results = []
        for t in range(num_frames):
            frame_ref = ref[:, :, t, :, :]
            frame_test = test[:, :, t, :, :]
            results.append({
                "psnr": psnr(frame_ref, frame_test, data_range),
                "ssim": ssim(frame_ref, frame_test, data_range),
            })
        return results
    else:
        # Image: [B, C, H, W]
        return [{
            "psnr": psnr(ref, test, data_range),
            "ssim": ssim(ref, test, data_range),
        }]
