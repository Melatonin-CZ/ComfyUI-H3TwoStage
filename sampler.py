"""Two-stage H3 sampling: low-res prefix -> latent lift -> re-noise -> full-res finish, one decode at the end.

Audio is never lifted: at the hand-over it keeps the low-res stage's own last step (the sampler's rule) and continues
on its own schedule (H3 carries audio scaled by sigma_v / sigma_a, so the AV pack is one ordinary flow latent).

Only memoryless samplers can be split in two: each step may depend on the current latent and the model's prediction,
never on earlier steps. Euler (an ODE step along the prediction) and LCM (jump to the prediction, add fresh noise:
the rule few-step distilled LoRAs such as DMAD were trained with) qualify; multistep solvers keep history.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

from . import lift as lift_mod
from . import noise as noise_mod
from . import sizes

LOG = "[H3 two-stage]"
META_KEY = "h3_two_stage"
DEFAULT_SCALE = 0.75
SAMPLERS = ("euler", "lcm")  # memoryless k-diffusion samplers, see the module docstring


@dataclass(frozen=True)
class Settings:
    split_step: int
    lowres_scale: float
    lift: str
    noise_keep: float
    strict_size: bool


def settings_from_latent(latent_image, split_step, lift, noise_keep, legacy=None) -> Settings:
    """lowres_scale comes with the latent from H3 Two-Stage Size (exact sizes). A plain empty latent (old graphs,
    comparisons with old renders) falls back to an old-style input or 0.75 with the old rounding."""
    meta = latent_image.get(META_KEY)
    if meta:
        return Settings(split_step, float(meta["lowres_scale"]), lift, noise_keep, True)
    legacy = legacy or {}
    scale = float(legacy.get("lowres_scale", DEFAULT_SCALE))
    logging.info("%s latent not from H3 Two-Stage Size: lowres_scale %.2f, old size rounding", LOG, scale)
    return Settings(split_step, scale, lift, noise_keep, bool(legacy.get("strict_size", False)))


def _streams(x):
    return (list(x.unbind()), True) if x.is_nested else ([x], False)


def _pack(streams, nested):
    return comfy.nested_tensor.NestedTensor(streams) if nested else streams[0]


def _validate(model, sampler_name, sigmas, latent_image, s: Settings) -> None:
    if not isinstance(model.get_model_object("model_sampling"), comfy.model_sampling.CONST):
        raise ValueError(f"{LOG} needs a rectified-flow model (MiniMax H3)")
    if sampler_name not in SAMPLERS:
        raise ValueError(f"{LOG} sampler '{sampler_name}' cannot be split in two; use one of {', '.join(SAMPLERS)}")
    if sigmas.ndim != 1 or sigmas.numel() < 3:
        raise ValueError(f"{LOG} needs a schedule of at least 2 steps")
    if not 1 <= s.split_step <= sigmas.numel() - 2:
        raise ValueError(f"{LOG} split_step {s.split_step} must leave at least one step on each side "
                         f"of a {sigmas.numel() - 1}-step schedule")
    if not 0.25 <= s.lowres_scale < 1.0:
        raise ValueError(f"{LOG} lowres_scale must be in [0.25, 1)")
    if latent_image.get("noise_mask") is not None:
        raise ValueError(f"{LOG} noise masks are not supported; pass first/last frames through conditioning")
    for st in _streams(latent_image["samples"])[0]:
        if torch.count_nonzero(st) != 0:
            raise ValueError(f"{LOG} latent_image must be an empty (all-zero) size template")


def _resize_keyframes(cond, hw):
    """Keyframe latents follow the size the conditioning was built at; resize them to each stage's grid."""
    out = []
    for tensor, d in cond:
        kfs = d.get("minimax_keyframes")
        if kfs is None:
            out.append((tensor, d))
            continue
        resized = []
        for kf in kfs:
            lat = kf.get("latent")
            if lat is not None and tuple(lat.shape[-2:]) != tuple(hw):
                kf = {**kf, "latent": F.interpolate(lat.float(), size=(lat.shape[2], *hw), mode="trilinear",
                                                    align_corners=False).to(lat)}
            resized.append(kf)
        out.append((tensor, {**d, "minimax_keyframes": resized}))
    return out


REF_AREA_TOLERANCE = 1.15


def _check_reference_size(cond, low_hw) -> None:
    """References are sized by the conditioning node to its own width x height. Built at the full size they cost
    a few percent more time per step than references built at the low-res size, with no visible difference
    (one test clip, 3 seeds). Note it in the log."""
    low_area = low_hw[0] * low_hw[1]
    areas = [r["latent_h"] * r["latent_w"] for _, d in cond for r in d.get("minimax_refs") or []
             if r.get("kind") == "image" and "latent_h" in r]
    if areas and max(areas) > REF_AREA_TOLERANCE * low_area:
        logging.info("%s reference images are %.1fx the low-res frame's area; building the conditioning at the "
                     "low-res size is slightly faster", LOG, max(areas) / low_area)


class _StepLog:
    """Per-step log: video sigma, audio sigma, and the noise level actually present in each stream."""

    def __init__(self, stage, sigmas, model, offset, total, preview):
        ms = model.get_model_object("model_sampling")
        self.stage, self.sigmas, self.offset, self.total, self.preview = stage, sigmas, offset, total, preview
        self.shift, self.audio_shift = getattr(ms, "shift", None), getattr(ms, "audio_shift", None)
        self.count = 0
        self.last = None
        self.t0 = self.prev = time.perf_counter()

    def __call__(self, step, x0, x, total):
        i = self.count
        self.count += 1
        sigma = float(self.sigmas[i])
        self.last = (x, x0, sigma)
        parts = []
        for name, xs, x0s in zip(("video", "audio"), _streams(x)[0], _streams(x0)[0]):
            eps = noise_mod.implied_noise(xs, x0s, max(sigma, 1e-6))
            parts.append(f"{name} noise std {eps.std().item():.3f}")
        audio_sigma = ""
        if self.audio_shift:
            from comfy.ldm.minimax.model import time_shift_sigma
            audio_sigma = f" sigma_a {float(time_shift_sigma(torch.tensor(sigma), self.shift, self.audio_shift)):.4f}"
        now = time.perf_counter()
        logging.info("%s %s step %d/%d sigma_v %.4f%s | %s | %.1fs", LOG, self.stage, i + 1,
                     len(self.sigmas) - 1, sigma, audio_sigma, ", ".join(parts), now - self.prev)
        self.prev = now
        return self.preview(i + self.offset, x0, x, self.total)


def sample(model, positive, negative, latent_image, sampler_name, sigmas, seed, cfg, s: Settings, model_low=None):
    """Returns (result latent, low-res x0 latent, lifted latent) — the last two video only, for inspection."""
    _validate(model, sampler_name, sigmas, latent_image, s)
    sampler = comfy.samplers.ksampler(sampler_name)
    model_low = model_low or model
    device = comfy.model_management.intermediate_device()
    streams, nested = _streams(comfy.sample.fix_empty_latent_channels(
        model, latent_image["samples"], latent_image.get("downscale_ratio_spacial"),
        latent_image.get("downscale_ratio_temporal")))
    video = streams[0]
    if video.ndim != 5:
        raise ValueError(f"{LOG} expects a 5D video latent")
    b, c, t, H, W = video.shape
    h, w = sizes.check((H, W), s.lowres_scale, s.strict_size)
    k = s.split_step
    sigma_k, sigma_next = float(sigmas[k - 1]), float(sigmas[k])
    logging.info("%s plan target %dx%d low %dx%d (lift %.4f x %.4f) frames %d | low steps %d, full steps %d | "
                 "hand-over predicts at sigma %.4f, resumes at %.4f | sampler %s, lift %s, noise_keep %.2f",
                 LOG, W, H, w, h, W / w, H / h, t, k, sigmas.numel() - 1 - k, sigma_k, sigma_next, sampler_name,
                 s.lift, s.noise_keep)
    _check_reference_size(positive, (h, w))

    total = sigmas.numel() - 1
    preview = latent_preview.prepare_callback(model, total)
    no_bar = not comfy.utils.PROGRESS_BAR_ENABLED
    batch_index = latent_image.get("batch_index")

    # 1. low-res prefix
    low_latent = _pack([torch.zeros((b, c, t, h, w), device=device)] +
                       [torch.zeros_like(x) for x in streams[1:]], nested)
    noise_low = comfy.sample.prepare_noise(low_latent, seed, batch_index)
    low_log = _StepLog("low", sigmas[:k + 1], model_low, 0, total, preview)
    low_out = comfy.samplers.sample(model_low, noise_low, _resize_keyframes(positive, (h, w)),
                                    _resize_keyframes(negative, (h, w)), cfg, model_low.load_device, sampler,
                                    sigmas[:k + 1], model_low.model_options, latent_image=low_latent,
                                    callback=low_log, disable_pbar=no_bar, seed=seed)
    if low_log.count != k:
        raise RuntimeError(f"{LOG} expected {k} low-res steps, saw {low_log.count}")
    state, x0, _ = low_log.last
    state_streams, _ = _streams(state)
    x0_streams, _ = _streams(x0)
    # the low-res stage's last step already took audio to sigma_next by this sampler's own rule (latent format)
    audio_resume = [x.to(device) for x in _streams(low_out)[0][1:]]
    del low_latent, noise_low, low_out

    # 2. lift the predicted clean video
    t_lift = time.perf_counter()
    latent_format = model.get_model_object("latent_format")
    x0_video_low = latent_format.process_out(x0_streams[0].float()).to(device)
    lifted = latent_format.process_in(lift_mod.lift(x0_video_low, (H, W), s.lift)).to(device)

    # 3. re-noise video at the resume sigma (fresh noise around the lifted prediction is itself an LCM step)
    fresh = comfy.sample.prepare_noise(lifted, (seed + 1) % (1 << 64), batch_index).to(lifted)
    carried = noise_mod.implied_noise(state_streams[0].to(device), x0_streams[0].to(device), sigma_k)
    eps = noise_mod.mix(fresh, carried, s.noise_keep).to(lifted)
    video_next = sigma_next * eps + (1.0 - sigma_next) * lifted
    # bring the video into latent format through the pack (the audio placeholders are dropped right after)
    ms = model.get_model_object("model_sampling")
    pack = [ms.inverse_noise_scaling(torch.tensor(sigma_next), x)
            for x in [video_next] + [st.to(device) for st in state_streams[1:]]]
    video_resume = _streams(model.model.process_latent_out(_pack(pack, nested)))[0][0]
    resume = [video_resume.to(device)] + audio_resume
    resume_latent = _pack(resume, nested)
    resume_noise = _pack([torch.zeros_like(x) for x in resume], nested)
    logging.info("%s hand-over %.1fs (noise std %.3f)", LOG, time.perf_counter() - t_lift, eps.std().item())
    low_preview = {"samples": x0_video_low.cpu()}
    lifted_preview = {"samples": latent_format.process_out(lifted).cpu()}
    del state, x0, state_streams, x0_streams, carried, fresh, eps, video_next, pack, video_resume, audio_resume, resume

    # 4. full-res finish
    high_log = _StepLog("full", sigmas[k:], model, k, total, preview)
    out = comfy.samplers.sample(model, resume_noise, _resize_keyframes(positive, (H, W)),
                                _resize_keyframes(negative, (H, W)), cfg, model.load_device, sampler,
                                sigmas[k:], model.model_options, latent_image=resume_latent, callback=high_log,
                                disable_pbar=no_bar, seed=seed)
    if high_log.count != total - k:
        raise RuntimeError(f"{LOG} expected {total - k} full-res steps, saw {high_log.count}")
    logging.info("%s low %.1fs, full %.1fs", LOG, low_log.prev - low_log.t0, high_log.prev - high_log.t0)
    result = {k: v for k, v in latent_image.items() if k != META_KEY}
    result = {**result, "samples": out.to(device=device, dtype=comfy.model_management.intermediate_dtype())}
    return result, low_preview, lifted_preview
