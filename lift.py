"""Latent lift from the low-res grid to the target grid.

The learned lift uses the MiniMax H3 latent upscaler by LBH-123-AI, loaded through that author's own plugin
(custom_nodes/Comfyui_Minimax_h3_latent_Upscaler) so its code is not copied here. Checked 2026-10-28 against
ground-truth encodings: it halves the bilinear error, its extra mean/std normalisation is how it was trained,
and running the whole clip in one pass matches the chunked result.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys

import torch
import torch.nn.functional as F

import comfy.model_management
import folder_paths

FOLDER = "latent_upscale_models"
_PLUGIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "Comfyui_Minimax_h3_latent_Upscaler", "nodes", "minimax_h3_latent_upscaler_3d.py")
_module = None


def _upscaler_module():
    global _module
    if _module is None:
        if not os.path.isfile(_PLUGIN):
            raise FileNotFoundError(
                "H3 two-stage: the learned lift needs the plugin Comfyui_Minimax_h3_latent_Upscaler "
                f"(expected {_PLUGIN}); use lift 'bilinear' or install it")
        spec = importlib.util.spec_from_file_location("h3twostage_lbh_upscaler3d", _PLUGIN)
        _module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _module
        spec.loader.exec_module(_module)
    return _module


if FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER))
DEFAULT_MODEL = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
_models: dict[str, torch.nn.Module] = {}


def choices() -> tuple[list[str], str]:
    """Every model file in models/latent_upscale_models (all registered paths, subfolders included), like the
    old SelfLift node, plus 'bilinear'. Default: the H3 3D upscaler if present."""
    names = sorted(n for n in folder_paths.get_filename_list(FOLDER)
                   if os.path.splitext(n)[1].lower() in (".pth", ".safetensors"))
    h3 = [n for n in names if "h3" in n.lower() and "lms" not in n.lower()]
    default = DEFAULT_MODEL if DEFAULT_MODEL in names else (h3[0] if h3 else "bilinear")
    return names + ["bilinear"], default


def _load(name: str) -> torch.nn.Module:
    """Build the upscaler with the LBH plugin's own architecture and loaders; fp16, cached on the CPU."""
    if name in _models:
        return _models[name]
    path = folder_paths.get_full_path(FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"H3 two-stage: upscaler '{name}' not found in models/{FOLDER}")
    mod = _upscaler_module()
    sd = mod._extract_upscaler_sd(mod._load_raw_sd(path))
    cfg = mod._detect_arch(sd)
    if cfg["in_channels"] != 24:
        raise ValueError(f"H3 two-stage: '{name}' takes {cfg['in_channels']} latent channels, H3 has 24 - "
                         "pick an H3 latent upscaler")
    model = mod.LatentResizer3D(**{k: cfg[k] for k in ("in_channels", "in_blocks", "out_blocks", "channels",
                                                       "dropout", "attn", "temporal_every", "temporal_kernel")})
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise ValueError(f"H3 two-stage: '{name}' is not an H3 3D latent upscaler ({str(e)[:200]})") from e
    model = model.eval().to(torch.float16)
    _models[name] = model
    logging.info("[H3 two-stage] loaded upscaler %s from %s", name, path)
    return model


def lift(z0_low: torch.Tensor, target_hw: tuple[int, int], method: str) -> torch.Tensor:
    """[B, 24, T, h, w] clean video latent -> [B, 24, T, H, W] on the intermediate device."""
    H, W = target_hw
    T = z0_low.shape[2]
    if method == "bilinear":
        return F.interpolate(z0_low.float(), size=(T, H, W), mode="trilinear", align_corners=False)
    mod = _upscaler_module()
    model = _load(method)
    device = comfy.model_management.get_torch_device()
    # weights (~0.7 GB) plus a few live 512-channel fp16 feature maps of the whole clip at the target size
    feature = 512 * T * H * W * z0_low.shape[0] * 2
    comfy.model_management.free_memory(6 * feature + 1024 ** 3, device)
    model.to(device)
    try:
        dtype = next(model.parameters()).dtype
        mean, std = mod._make_norm_tensors(device, dtype)
        scale = (H / z0_low.shape[-2] + W / z0_low.shape[-1]) / 2.0
        with torch.no_grad():
            x = (z0_low.to(device, dtype) - mean) / std
            out = model(x, scale=scale, target_size=(T, H, W))
            out = (out * std + mean).float()
    finally:
        model.to("cpu")
    logging.info("[H3 two-stage] lift %s %s -> %s scale %.4f", method, tuple(z0_low.shape[-2:]), (H, W), scale)
    return out.to(comfy.model_management.intermediate_device())
