# ComfyUI-H3TwoStage

[中文说明](README_zh.md)

Two-stage sampling for the **MiniMax H3** audio-video model in ComfyUI: the first steps run at low resolution
(where layout and motion are decided), the predicted clean latent is upscaled in latent space with a learned 3D
upscaler, re-noised, and the remaining steps run at full resolution. One VAE decode at the end; the audio stream
continues on its own schedule through both stages.

It is a lean alternative to the H3 sampler in [comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) for
the way that sampler is usually run with H3 (`rho = 0` + an external latent upscaler): the same sampling path,
without the unused SelfLift correction, plus exact size handling and diagnostics.

Two nodes, both under `sampling/minimax`:

- **H3 Two-Stage Size**: aspect ratio + megapixels + seconds → an exact two-stage size, the frame count and the
  empty full-size AV latent.
- **H3 Two-Stage Sampler**: low-res steps → latent lift → re-noise → full-res steps.

## Wiring

```
H3 Two-Stage Size ──latent──────────────────────────► H3 Two-Stage Sampler.latent_image
                  ├─low_width / low_height / length ─► H3 conditioning node width / height / length ─► Sampler.positive
                  └─width / height (instead of low_*, for clips with a pinned first/last frame)
```

- Sampler: `euler` (KSamplerSelect), sigmas e.g. BasicScheduler `simple` 8 steps (+ an optional refiner step).
- Reference-only clips: build the conditioning at the **low-res size**. References are sized by the conditioning
  node to its own width × height; at the low-res size each step is ~5–8 % faster with no visible difference.
- Clips with a first/last frame: build the conditioning at the **full size** so keyframes are encoded at full
  resolution; the sampler shrinks them for the low-res stage.

## H3 Two-Stage Size

| Input | Type | Default | Meaning |
|---|---|---|---|
| aspect_ratio | combo | 16:9 (Widescreen) | Same choices as ComfyUI's Resolution Selector |
| megapixels | FLOAT | 1.8 | Target area (1 MP = 1024 × 1024); snapped to the nearest exact two-stage size |
| seconds | FLOAT | 5.0 | Clip length; frames at 24 fps, rounded up to H3's 17k + 5 grid |
| lowres_scale | FLOAT | 0.75 | Low-res stage scale per side; travels inside the latent to the sampler |

| Output | Type | Meaning |
|---|---|---|
| latent | LATENT (AV) | Empty full-size latent → sampler `latent_image` |
| width, height | INT | Full size (conditioning node, when a first/last frame is pinned) |
| low_width, low_height | INT | Low-res size (conditioning node, reference-only clips) |
| length | INT | Frame count → conditioning node `length` |
| info | STRING | Summary |

"Exact" means both latents have even sides (H3 patches the latent 2 × 2 and pads odd sides by wrapping, which shows
up as coloured edges) and the low-res side is exactly `lowres_scale` × the full side. Example: 1.8 MP 16:9 →
1792 × 1024, low-res 1344 × 768 (H3's native 768p canvas).

## H3 Two-Stage Sampler

| Input | Type | Default | Meaning |
|---|---|---|---|
| model | MODEL | | H3 model (with LoRA); full-res stage, and low-res stage unless `model_low` is set |
| positive / negative | CONDITIONING | | From the H3 conditioning node |
| latent_image | LATENT | | From H3 Two-Stage Size (must be empty; no noise mask). A plain empty latent also works (scale 0.75, old rounding) |
| sampler | SAMPLER | | Plain `euler` only (other solvers carry history and cannot be split) |
| sigmas | SIGMAS | | The full schedule |
| seed | INT | 0 | Low-res noise; the re-noise at the hand-over uses seed + 1 |
| cfg | FLOAT | 1.0 | As in KSampler |
| split_step | INT | 6 | Steps at low resolution; the rest run at full resolution (at least one each side) |
| lift | combo | `minimax_h3_latent_upscaler_3d_fp16.safetensors` | Any file in `models/latent_upscale_models/`, or `bilinear` (no model). Only 24-channel H3 upscalers load; others are refused with a clear message |
| noise_keep | FLOAT | 0 | 0 = fresh noise at the hand-over (recommended). > 0 keeps the low frequencies of the low-res stage's own noise (exact orthonormal-FFT band split). It made no measurable difference in our tests |
| model_low (optional) | MODEL | | Separate model for the low-res stage |

Outputs: `latent` (AV, to the VAE decoders), `low_res_x0` and `lifted` (video only, for inspection).

Every step logs a line `[H3 two-stage]` with stage, video/audio sigma, the measured noise level of each stream and
the time per step; the first line states the plan (sizes, steps, hand-over sigma, upscaler).

## Requirements

- A ComfyUI build with native MiniMax H3 support (`comfy_extras/nodes_minimax_h3.py`).
- For the learned lift: the plugin **[Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)**
  by **LBH-123-AI**, installed next to this one in `custom_nodes/`, and its model weights
  (e.g. `minimax_h3_latent_upscaler_3d_fp16.safetensors`) from
  [huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
  in `ComfyUI/models/latent_upscale_models/`. Without them only `lift = bilinear` works.

The upscaler's architecture and loading code are **not** included here: this node imports them at run time from
that plugin, which carries no license file, so its code and weights stay under its author's terms. Its authors
credit [ComfyUi_NNLatentUpscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale) by Ttl and the LTX 2.3 spatial
upscaler as foundations.

## Notes from testing (H3, 4-step LoRA, 8 + 1 steps)

- The 3D upscaler is accurate: against ground-truth encodings it halves the error of bilinear upscaling, including
  low frequencies; its extra mean/std normalisation of ComfyUI latents is how it was trained; running a whole clip
  in one pass matches the chunked result.
- Same seed, same settings: output is equivalent to the SelfLift H3 sampler (`rho = 0`).
- `split_step` 4 / 5 / 6 of 8 gives near-identical clips; 6 is a little faster.
- Layout depends on the starting noise, which depends on size and seed together: the same seed at another size
  gives another composition. Compare seeds at a fixed size.

## Tests

Offline tests for sizes, noise mixing and the hand-over maths (no model needed):

```
python_embeded/python.exe -s -m pytest custom_nodes/ComfyUI-H3TwoStage/tests -q
```

## Credits

- [comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) (facok): the two-stage H3 sampling path this
  node streamlines. No code is copied from it.
- [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) and
  [the model weights](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) (LBH-123-AI): the learned
  3D latent upscaler, loaded from that plugin at run time.

## License

MIT for the code in this repository (see [LICENSE](LICENSE)). Third-party plugins and model weights it loads keep
their own terms.
