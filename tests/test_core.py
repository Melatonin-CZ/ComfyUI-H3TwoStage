"""Offline checks for the pieces that need no model: sizes, noise mixing, hand-over maths.

Run: python_embeded/python.exe -s -m pytest custom_nodes/ComfyUI-H3TwoStage/tests -q
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(f"h3ts_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sizes = _load("sizes")
noise = _load("noise")


def test_exact_size_passes():
    assert sizes.check((64, 112), 0.75, strict=True) == (48, 84)


def test_inexact_default_size_is_refused_with_suggestion():
    with pytest.raises(ValueError, match="1792x1024"):
        sizes.check((64, 114), 0.75, strict=True)


def test_odd_latent_side_is_refused():
    with pytest.raises(ValueError):
        sizes.check((112, 63), 0.75, strict=True)


def test_non_strict_matches_old_rounding():
    assert sizes.check((64, 114), 0.75, strict=False) == (48, 86)


def test_nearest_valid_side():
    assert sizes.nearest_valid_side(114, 0.75) == 112
    assert sizes.nearest_valid_side(63, 0.75) == 64
    assert sizes.nearest_valid_side(64, 0.5) == 64


def test_mix_zero_returns_fresh_noise():
    fresh = torch.randn(1, 24, 3, 64, 112)
    carried = torch.randn(1, 24, 3, 48, 84)
    assert noise.mix(fresh, carried, 0.0) is fresh


@pytest.mark.parametrize("keep", [0.5, 1.0])
def test_mix_keeps_unit_variance(keep):
    torch.manual_seed(0)
    out = noise.mix(torch.randn(1, 24, 9, 64, 112), torch.randn(1, 24, 9, 48, 84), keep)
    assert abs(out.std().item() - 1.0) < 0.03
    assert abs(out.mean().item()) < 0.02


def test_mix_low_band_follows_carried_noise():
    torch.manual_seed(1)
    carried = torch.randn(1, 24, 4, 48, 84)
    out = noise.mix(torch.randn(1, 24, 4, 64, 112), carried, 1.0)
    # bring the result back onto the low grid by keeping only its low band (exact inverse of the band copy)
    spec = torch.fft.rfft2(out, norm="ortho")
    low = torch.cat([spec[..., :24, :43], spec[..., 64 - 24:, :43]], dim=-2)
    back = torch.fft.irfft2(low, s=(48, 84), norm="ortho")
    corr = torch.corrcoef(torch.stack([back.flatten(), carried.flatten()]))[0, 1].item()
    assert corr > 0.95


def test_mix_half_keep_correlation_is_half():
    torch.manual_seed(3)
    carried = torch.randn(1, 24, 4, 48, 84)
    out = noise.mix(torch.randn(1, 24, 4, 64, 112), carried, 0.5)
    spec = torch.fft.rfft2(out, norm="ortho")
    back = torch.fft.irfft2(torch.cat([spec[..., :24, :43], spec[..., 40:, :43]], dim=-2), s=(48, 84), norm="ortho")
    corr = torch.corrcoef(torch.stack([back.flatten(), carried.flatten()]))[0, 1].item()
    assert 0.42 < corr < 0.58


def test_mix_output_is_white():
    """Neighbouring cells of the mixed noise stay uncorrelated (no smoothing or blocking)."""
    torch.manual_seed(4)
    out = noise.mix(torch.randn(1, 24, 4, 64, 112), torch.randn(1, 24, 4, 48, 84), 1.0)
    a, b = out[..., :, :-1].flatten(), out[..., :, 1:].flatten()
    assert abs(torch.corrcoef(torch.stack([a, b]))[0, 1].item()) < 0.03


def test_implied_noise_recovers_eps():
    eps, x0, sigma = torch.randn(2, 8), torch.randn(2, 8), 0.95
    state = sigma * eps + (1 - sigma) * x0
    assert torch.allclose(noise.implied_noise(state, x0, sigma), eps, atol=1e-5)


def test_handover_equals_old_renoise_then_euler():
    """Old sampler: noise at sigma_k, then one Euler step with the lifted x0 to sigma_next."""
    torch.manual_seed(2)
    eps, z0 = torch.randn(1000), torch.randn(1000)
    sk, sn = 0.9524, 0.9231
    old_state = sk * eps + (1 - sk) * z0
    old_next = old_state + (old_state - z0) * ((sn - sk) / sk)
    new_next = sn * eps + (1 - sn) * z0
    assert torch.allclose(old_next, new_next, atol=1e-5)
    assert math.isclose(sn, 0.9231)
