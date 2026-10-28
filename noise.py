"""Re-noising the lifted video latent at the hand-over sigma.

noise_keep = 0: fresh noise only (what the old SelfLift sampler did).
noise_keep > 0: the low frequency band of the new noise (the frequencies the low-res grid can hold) comes from
the noise the low-res stage actually carried, recovered from its state, so the full-res stage continues that
trajectory; the band above stays fresh. The split is done exactly in the 2D spectrum (orthonormal FFT), where
white noise has independent coefficients, so blending the low band as keep * carried + sqrt(1 - keep^2) * fresh
keeps the noise white with variance 1.
"""

from __future__ import annotations

import math

import torch


def implied_noise(state: torch.Tensor, x0: torch.Tensor, sigma: float) -> torch.Tensor:
    """Rectified flow: state = sigma * eps + (1 - sigma) * x0, so eps = (state - (1 - sigma) * x0) / sigma."""
    return (state.float() - (1.0 - sigma) * x0.float()) / sigma


def _band(H: int, W: int, h: int, w: int):
    """Row indices and column count of the rfft2 coefficients of an HxW grid that an hxw grid also has."""
    rows = list(range(0, (h + 1) // 2)) + list(range(H - h // 2, H))
    return torch.tensor(rows), w // 2 + 1


def _white(x: torch.Tensor) -> torch.Tensor:
    """Normalise each channel to unit variance (the recovered noise is only approximately white)."""
    return (x - x.mean(dim=(0, 2, 3, 4), keepdim=True)) / x.std(dim=(0, 2, 3, 4), keepdim=True).clamp(min=1e-6)


def mix(fresh: torch.Tensor, carried_low: torch.Tensor, keep: float) -> torch.Tensor:
    """New [B, C, T, H, W] noise: low band blended from `carried_low` ([B, C, T, h, w]), high band from `fresh`."""
    if keep <= 0.0:
        return fresh
    keep = min(keep, 1.0)
    H, W = fresh.shape[-2:]
    h, w = carried_low.shape[-2:]
    rows, cols = _band(H, W, h, w)
    spec_f = torch.fft.rfft2(fresh.float(), norm="ortho")
    spec_c = torch.fft.rfft2(_white(carried_low.float().to(fresh.device)), norm="ortho")
    rows = rows.to(fresh.device)
    low_rows = torch.cat([torch.arange(0, (h + 1) // 2), torch.arange(h - h // 2, h)]).to(fresh.device)
    # With the orthonormal FFT every white unit-variance grid has unit-variance coefficients, whatever its size,
    # so carried coefficients replace fresh ones at the same frequency with no gain.
    out = spec_f.clone()
    out[..., rows, :cols] = keep * spec_c[..., low_rows, :cols] + math.sqrt(1.0 - keep * keep) * spec_f[..., rows, :cols]
    return torch.fft.irfft2(out, s=(H, W), norm="ortho")
