"""
Denoising metrics implemented with PyTorch only.

This file provides two metrics implemented using torch operations:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index (fast torch-based implementation)

Both classes inherit from the project's Metric base class and accept
batch tensors in (B, C, H, W) format. Inputs are converted to float tensors
and moved to the same device for calculations.
"""
from typing import Optional

import torch
import torch.nn.functional as F

from pytorcher.metrics import Metric


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    """Convert input to torch float tensor.

    Accepts numpy arrays or torch tensors. Returns a torch.float32 tensor.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.to(dtype=torch.float32)


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, dtype: torch.dtype = torch.float32, device=None):
    """Create 2D gaussian kernel for SSIM (separable).

    Returns a tensor of shape (1, 1, window_size, window_size).
    """
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel2d = g[:, None] @ g[None, :]
    kernel2d = kernel2d / kernel2d.sum()
    return kernel2d.unsqueeze(0).unsqueeze(0)


class PSNR(Metric):
    """
    Peak Signal-to-Noise Ratio computed with torch ops.

    Args:
        name: optional metric name
        max_val: maximum possible pixel value (e.g., 1.0 or 255)
        eps: small epsilon to avoid divide-by-zero
    """

    def __init__(self, name: Optional[str] = None, max_val: float = 1.0, eps: float = 1e-10):
        super().__init__(name or "psnr")
        self.max_val = float(max_val)
        self.eps = float(eps)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to torch float tensors
        y_true_t = _as_float_tensor(y_true)
        y_pred_t = _as_float_tensor(y_pred)

        # Ensure shape (B, C, H, W)
        if y_true_t.dim() == 3:  # (C, H, W) -> (1, C, H, W)
            y_true_t = y_true_t.unsqueeze(0)
        if y_pred_t.dim() == 3:
            y_pred_t = y_pred_t.unsqueeze(0)

        # Move predictions to same device as targets
        y_pred_t = y_pred_t.to(device=y_true_t.device)

        # compute per-image MSE over channels and spatial dims
        mse = torch.mean((y_true_t - y_pred_t) ** 2, dim=(1, 2, 3))
        psnr = 20.0 * torch.log10(torch.tensor(self.max_val, dtype=mse.dtype, device=mse.device) / torch.sqrt(mse + self.eps))

        batch_total = float(torch.sum(psnr).item())
        batch_count = float(psnr.numel())

        self._total += batch_total
        self._count += batch_count


class SSIM(Metric):
    """
    Structural Similarity Index (SSIM) implemented using torch operations.

    This implementation follows the standard windowed SSIM formula using a
    separable Gaussian window and computes mean SSIM per image across
    channels and spatial dimensions.

    Args:
        name: optional metric name
        max_val: maximum possible pixel value (e.g., 1.0 or 255)
        window_size: size of gaussian window (odd integer, default 11)
        sigma: gaussian sigma used for window (default 1.5)
    """

    def __init__(self, name: Optional[str] = None, max_val: float = 1.0, window_size: int = 11, sigma: float = 1.5):
        super().__init__(name or "ssim")
        self.max_val = float(max_val)
        self.window_size = int(window_size)
        self.sigma = float(sigma)

    def _ssim_map(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute SSIM map for two batches X and Y with shape (B, C, H, W)."""
        # kernel
        device = X.device
        dtype = X.dtype
        kernel = _gaussian_kernel(self.window_size, self.sigma, dtype=dtype, device=device)

        B, C, H, W = X.shape

        # convolve per-channel using groups
        kernel = kernel.expand(C, 1, self.window_size, self.window_size)

        # padding to keep same size
        padding = self.window_size // 2

        mu_x = F.conv2d(X, kernel, groups=C, padding=padding)
        mu_y = F.conv2d(Y, kernel, groups=C, padding=padding)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(X * X, kernel, groups=C, padding=padding) - mu_x_sq
        sigma_y_sq = F.conv2d(Y * Y, kernel, groups=C, padding=padding) - mu_y_sq
        sigma_xy = F.conv2d(X * Y, kernel, groups=C, padding=padding) - mu_xy

        # constants to stabilize the division with weak denominators
        c1 = (0.01 * self.max_val) ** 2
        c2 = (0.03 * self.max_val) ** 2

        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

        ssim_map = numerator / (denominator + 1e-12)
        return ssim_map

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to float tensors
        X = _as_float_tensor(y_true)
        Y = _as_float_tensor(y_pred)

        if X.dim() == 3:
            X = X.unsqueeze(0)
        if Y.dim() == 3:
            Y = Y.unsqueeze(0)

        Y = Y.to(device=X.device)

        # ensure shape (B, C, H, W)
        if X.shape[1] != Y.shape[1]:
            # try to handle (B, H, W, C) layout by permuting
            if X.dim() == 4 and X.shape[-1] == Y.shape[1]:
                X = X.permute(0, 3, 1, 2)
            if Y.dim() == 4 and Y.shape[-1] == X.shape[1]:
                Y = Y.permute(0, 3, 1, 2)

        # compute ssim map and average per-image
        ssim_map = self._ssim_map(X, Y)
        # mean over channels and spatial dims -> per-image value
        ssim_per_image = torch.mean(ssim_map, dim=(1, 2, 3))

        batch_total = float(torch.sum(ssim_per_image).item())
        batch_count = float(ssim_per_image.numel())

        self._total += batch_total
        self._count += batch_count
