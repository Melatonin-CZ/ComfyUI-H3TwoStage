"""Latent lift from the low-res grid to the target grid.

The learned lift is the MiniMax H3 latent upscaler (weights by LBH-123-AI), run through ComfyUI's own code: the
"Load Latent Upscale Model" loader and the "MiniMax H3 Latent Upscale With Model" node, called here directly so no
extra nodes are wired. ComfyUI's version processes the clip in 16-frame slabs with clip-wide normalisation
statistics (same result as one pass, less VRAM) and loads and offloads like any other ComfyUI model.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F

import comfy.model_management
import folder_paths

FOLDER = "latent_upscale_models"
SPATIAL = 16  # H3 video latent: one latent pixel per 16 image pixels

if FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER))
DEFAULT_MODEL = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
_models: dict[str, object] = {}


def choices() -> tuple[list[str], str]:
    """Every model file in models/latent_upscale_models (all registered paths, subfolders included), like the
    old SelfLift node, plus 'bilinear'. Default: the H3 3D upscaler if present."""
    names = sorted(n for n in folder_paths.get_filename_list(FOLDER)
                   if os.path.splitext(n)[1].lower() in (".pth", ".safetensors"))
    h3 = [n for n in names if "h3" in n.lower() and "lms" not in n.lower()]
    default = DEFAULT_MODEL if DEFAULT_MODEL in names else (h3[0] if h3 else "bilinear")
    return names + ["bilinear"], default


def _load(name: str):
    """ComfyUI's latent upscale model loader; only its MiniMax H3 upscaler is accepted. Cached per file."""
    if name in _models:
        return _models[name]
    from comfy.ldm.minimax.latent_upscaler import MiniMaxH3LatentUpscaler
    from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader

    if folder_paths.get_full_path(FOLDER, name) is None:
        raise FileNotFoundError(f"H3 two-stage: upscaler '{name}' not found in models/{FOLDER}")
    loaded = LatentUpscaleModelLoader.execute(name).result
    patcher = loaded[0] if loaded else None
    if not isinstance(getattr(patcher, "model", None), MiniMaxH3LatentUpscaler):
        raise ValueError(f"H3 two-stage: '{name}' is not a MiniMax H3 latent upscaler - pick an H3 upscaler "
                         "or 'bilinear'")
    _models[name] = patcher
    logging.info("[H3 two-stage] loaded upscaler %s (ComfyUI MiniMaxH3LatentUpscaler, %s)", name,
                 patcher.model.get_dtype())
    return patcher


def lift(z0_low: torch.Tensor, target_hw: tuple[int, int], method: str) -> torch.Tensor:
    """[B, 24, T, h, w] clean video latent (latent format) -> [B, 24, T, H, W] on the intermediate device."""
    H, W = target_hw
    T = z0_low.shape[2]
    if method == "bilinear":
        return F.interpolate(z0_low.float(), size=(T, H, W), mode="trilinear", align_corners=False)
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3LatentUpscaleWithModel

    model = _load(method)
    with torch.no_grad():  # ComfyUI runs nodes in inference mode; outside it autograd would keep ~10x the activations
        out = MiniMaxH3LatentUpscaleWithModel.execute(model, {"samples": z0_low}, W * SPATIAL, H * SPATIAL).result
    lifted = out[0]["samples"]
    if tuple(lifted.shape[-3:]) != (T, H, W):
        raise RuntimeError(f"H3 two-stage: upscaler returned {tuple(lifted.shape)}, expected (..., {T}, {H}, {W})")
    logging.info("[H3 two-stage] lift %s %s -> %s", method, tuple(z0_low.shape[-2:]), (H, W))
    return lifted.float().to(comfy.model_management.intermediate_device())
