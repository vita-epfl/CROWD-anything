"""Per-frame image metrics between generated and reference frames, on batches of float tensors [B, 3, H, W] in
[0, 1]. Each function returns one value per frame [B], so errors can be followed over time and averaged later.

PSNR and SSIM follow the usual definitions (SSIM: 11x11 Gaussian window, sigma 1.5, K1 0.01, K2 0.03, averaged
over the RGB channels, as skimage's structural_similarity with gaussian_weights and channel_axis). LPIPS uses the
reference implementation (`lpips` package, AlexNet), the variant most video-generation papers report.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 frames [B, H, W, 3] -> float [B, 3, H, W] in [0, 1] on `device`."""
    return torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().div_(255.0)


def psnr(generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mse = (generated - reference).pow(2).flatten(1).mean(1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-coords.pow(2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(generated: torch.Tensor, reference: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    window = _gaussian_window(window_size, sigma, generated.device)

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, window, groups=3)  # valid convolution, as skimage crops the borders

    c1, c2 = 0.01**2, 0.03**2
    mu_x, mu_y = blur(generated), blur(reference)
    # sample covariance (skimage's default use_sample_covariance), unbiased over the window's weights
    n = window_size * window_size
    cov_norm = n / (n - 1)
    var_x = cov_norm * (blur(generated * generated) - mu_x.pow(2))
    var_y = cov_norm * (blur(reference * reference) - mu_y.pow(2))
    cov_xy = cov_norm * (blur(generated * reference) - mu_x * mu_y)
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2))
    return ssim_map.flatten(1).mean(1)


class Lpips:
    """LPIPS distance (lower is more similar). The network is loaded once, on first use."""

    def __init__(self, device: torch.device, net: str = "alex"):
        self.device = device
        self.net = net
        self._model: Optional[torch.nn.Module] = None

    def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            import lpips  # optional dependency, only needed for this metric

            self._model = lpips.LPIPS(net=self.net, verbose=False).to(self.device).eval()
        with torch.no_grad():
            return self._model(generated * 2 - 1, reference * 2 - 1).flatten()


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
