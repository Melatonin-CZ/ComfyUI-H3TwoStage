"""Size rules for H3 two-stage sampling.

H3's DiT groups the video latent into 2x2 patches and pads an odd side circularly (the right/bottom edge then
samples next to content wrapped from the other side). The VAE downscales 16x. So:
- both stages need even latent sides;
- the low-res grid should be exactly `scale` times the target grid, or the lift stretches the picture.
"""

from __future__ import annotations

from fractions import Fraction

SPATIAL = 16  # H3 video VAE downscale


def low_side(side: int, scale: float) -> int | None:
    """Low-res latent side for a target side, or None when scale * side is not an even whole number."""
    exact = Fraction(scale).limit_denominator(64) * side
    if exact.denominator != 1 or exact.numerator % 2 or exact.numerator < 2:
        return None
    return exact.numerator


def rounded_low_side(side: int, scale: float) -> int:
    """The rounding the old SelfLift sampler used (nearest even), for non-strict runs."""
    return max(2, round(side * scale / 2) * 2)


def nearest_valid_side(side: int, scale: float) -> int:
    """Nearest latent side that is even and whose low-res side is an exact even number."""
    for delta in range(0, 256):
        for cand in (side - delta, side + delta):
            if cand >= 2 and cand % 2 == 0 and low_side(cand, scale) is not None:
                return cand
    raise ValueError(f"no valid latent side near {side} for scale {scale}")


def check(target_hw: tuple[int, int], scale: float, strict: bool) -> tuple[int, int]:
    """Low-res latent (h, w) for a target latent (H, W). Strict mode refuses sizes that are not exact."""
    H, W = target_hw
    if strict:
        lows = (low_side(H, scale), low_side(W, scale))
        if H % 2 or W % 2 or None in lows:
            good = (nearest_valid_side(H, scale), nearest_valid_side(W, scale))
            raise ValueError(
                f"H3 two-stage: target {W * SPATIAL}x{H * SPATIAL} px (latent {W}x{H}) does not fit scale {scale}: "
                f"latent sides must be even and give an exact even low-res side. "
                f"Nearest valid size: {good[1] * SPATIAL}x{good[0] * SPATIAL} px. Use the 'H3 Two-Stage Size' node.")
        return lows[0], lows[1]
    return rounded_low_side(H, scale), rounded_low_side(W, scale)
