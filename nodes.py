"""ComfyUI nodes: H3 Two-Stage Size and H3 Two-Stage Sampler."""

from __future__ import annotations

from . import lift as lift_mod
from . import sampler as engine
from . import sizes


ASPECTS = {  # same choices as ComfyUI's Resolution Selector
    "1:1 (Square)": (1, 1), "2:3 (Portrait Photo)": (2, 3), "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4), "4:3 (Standard)": (4, 3), "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9), "21:9 (Ultrawide)": (21, 9),
}
FPS = 24


def frame_count(seconds: float) -> int:
    """Frames for a duration at 24 fps, rounded up to H3's 17k+5 grid (the old math-expression node)."""
    n = max(5, round(seconds * FPS))
    return n + (5 - n % 17) % 17


class H3TwoStageSize:
    """Everything the graph needs about size and length in one node (replaces Resolution Selector, the duration
    value and the frame-count expression): aspect + megapixels -> nearest exact two-stage size; seconds -> frame
    count. Outputs the empty full-size AV latent for the sampler (it carries lowres_scale) and the sizes and
    length for the conditioning node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "aspect_ratio": (list(ASPECTS), {"default": "16:9 (Widescreen)"}),
            "megapixels": ("FLOAT", {"default": 1.8, "min": 0.1, "max": 16.0, "step": 0.1,
                                     "tooltip": "Target size (1 MP = 1024x1024); snapped to an exact two-stage size."}),
            "seconds": ("FLOAT", {"default": 5.0, "min": 0.2, "max": 30.0, "step": 0.1,
                                  "tooltip": "Clip length; frames are rounded up to H3's 17k+5 grid at 24 fps."}),
            "lowres_scale": ("FLOAT", {"default": 0.75, "min": 0.25, "max": 0.95, "step": 0.05}),
        }}

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "low_width", "low_height", "length", "info")
    OUTPUT_TOOLTIPS = (
        "Empty AV latent at the full size -> sampler latent_image.",
        "Full width (conditioning node, only for clips with a first/last frame).", "Full height.",
        "Low-res width -> conditioning node width (reference-only clips).", "Low-res height.",
        "Frame count -> conditioning node length.", "Summary.")
    FUNCTION = "pick"
    CATEGORY = "sampling/minimax"

    def pick(self, aspect_ratio, megapixels, seconds, lowres_scale):
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent

        wr, hr = ASPECTS[aspect_ratio]
        unit = (megapixels * 1024 * 1024 / (wr * hr)) ** 0.5
        lw = sizes.nearest_valid_side(round(wr * unit / sizes.SPATIAL), lowres_scale)
        lh = sizes.nearest_valid_side(round(hr * unit / sizes.SPATIAL), lowres_scale)
        low_w, low_h = sizes.low_side(lw, lowres_scale), sizes.low_side(lh, lowres_scale)
        W, H = lw * sizes.SPATIAL, lh * sizes.SPATIAL
        length = frame_count(seconds)
        latent, frames = _empty_av_latent(W, H, length)
        latent[engine.META_KEY] = {"lowres_scale": lowres_scale}
        info = (f"{W}x{H} px, {frames} frames ({frames / FPS:.2f} s); low-res stage "
                f"{low_w * sizes.SPATIAL}x{low_h * sizes.SPATIAL} px")
        return (latent, W, H, low_w * sizes.SPATIAL, low_h * sizes.SPATIAL, frames, info)


class H3TwoStageSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "Model for the full-res stage (and the low-res stage unless model_low is set)."}),
            "positive": ("CONDITIONING", {"tooltip": "Build it at the low-res size (Size node low_width / "
                                                     "low_height) so references match the low-res frame."}),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT", {"tooltip": "From H3 Two-Stage Size (empty full-size AV latent; it also "
                                                   "carries lowres_scale)."}),
            "sampler": ("SAMPLER", {"tooltip": "Plain euler only."}),
            "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "split_step": ("INT", {"default": 6, "min": 1, "max": 1000,
                                   "tooltip": "Steps at low resolution; the rest run at full resolution."}),
            "lift": (lift_mod.choices()[0], {"default": lift_mod.choices()[1],
                     "tooltip": "Latent upscaler from models/latent_upscale_models (same folder as before), "
                                "or plain bilinear."}),
            "noise_keep": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                     "tooltip": "0 = fresh noise at the hand-over (old behaviour). Higher keeps the "
                                                "low frequencies of the low-res stage's own noise, so the full-res "
                                                "stage continues the same trajectory."}),
        }, "optional": {
            "model_low": ("MODEL", {"tooltip": "Optional separate model for the low-res stage (e.g. another LoRA strength)."}),
        }}

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("latent", "low_res_x0", "lifted")
    FUNCTION = "run"
    CATEGORY = "sampling/minimax"

    def run(self, model, positive, negative, latent_image, sampler, sigmas, seed, cfg, split_step, lift, noise_keep,
            model_low=None, **legacy):
        settings = engine.settings_from_latent(latent_image, split_step, lift, noise_keep, legacy)
        return engine.sample(model, positive, negative, latent_image, sampler, sigmas, seed, cfg, settings,
                             model_low)


NODE_CLASS_MAPPINGS = {
    "H3TwoStageSize": H3TwoStageSize,
    "H3TwoStageSampler": H3TwoStageSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TwoStageSize": "H3 Two-Stage Size",
    "H3TwoStageSampler": "H3 Two-Stage Sampler",
}
