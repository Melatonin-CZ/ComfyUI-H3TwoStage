# ComfyUI-H3TwoStage（中文说明）

[English README](README.md)

MiniMax H3 两段采样：低分辨率先采几步（构图和动作在这里定）→ 用 3D latent 放大模型放大预测的干净 latent → 重新加噪 →
全分辨率采完。最后只解码一次；音频在两段里按自己的噪声表连续采。

它是 [comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) 的 H3 采样节点在常见用法（`rho = 0` + 外部 latent 放大模型）
下的精简替代：采样路径相同，去掉没用到的 SelfLift 修正，加上精确尺寸和诊断日志。两个节点都在 `sampling/minimax` 分类下。

## 连线

```
H3 Two-Stage Size ──latent─────────────────────────► H3 Two-Stage Sampler.latent_image
                  ├─low_width / low_height / length ─► 条件节点 width / height / length ──► Sampler.positive
                  └─width / height（有首帧 / 尾帧的片段用它代替 low_*）
```

- 采样器在节点上直接选（`euler` 或 `lcm`，不再接 KSamplerSelect）；噪声表如 BasicScheduler `simple` 8 步（可加一步 refiner）。
- **只用参考图的片段，条件节点接低分辨率尺寸**：参考图按条件节点的宽高缩放，按低尺寸每步约快 5–8%，画面没有可见差别。
- **有首帧 / 尾帧的片段，条件节点接 width / height（成片尺寸）**，首尾帧按全分辨率编码，采样节点自动为低分辨率阶段缩小。
- 构图由尺寸和种子共同决定的初始噪声决定：同一种子换尺寸，构图会变。比较种子时固定尺寸。

## H3 Two-Stage Size（尺寸与时长）

| 输入 | 格式 | 默认 | 作用 |
|---|---|---|---|
| aspect_ratio | 下拉 | 16:9 (Widescreen) | 宽高比，选项同 Resolution Selector |
| megapixels | FLOAT | 1.8 | 目标面积（1 MP = 1024×1024），对齐到最近的精确两段尺寸 |
| seconds | FLOAT | 5.0 | 片长（秒），按 24 fps 向上对齐到 H3 的 17k+5 帧 |
| lowres_scale | FLOAT | 0.75 | 低分辨率阶段的缩放；随 latent 传给采样节点 |

| 输出 | 格式 | 作用 |
|---|---|---|
| latent | LATENT（AV） | 成片尺寸的空 latent → 采样节点 latent_image |
| width, height | INT | 成片尺寸 → 有首尾帧时接条件节点 |
| low_width, low_height | INT | 低分辨率尺寸 → 只用参考图时接条件节点 |
| length | INT | 帧数 → 条件节点 length |
| info | STRING | 尺寸、帧数、低分辨率尺寸 |

“精确”指两个 latent 的边长都是偶数（H3 按 2×2 打包 latent，奇数边会环绕补边，成片边缘出彩边），且低分辨率边长正好是
成片边长 × lowres_scale。例：1.8 MP 16:9 → 1792×1024，低分辨率 1344×768（H3 的原生 768p 画布）。

## H3 Two-Stage Sampler（两段采样）

| 输入 | 格式 | 默认 | 作用 |
|---|---|---|---|
| model | MODEL | | H3 模型（含 LoRA）；用于全分辨率阶段，没接 model_low 时低分辨率阶段也用它 |
| positive / negative | CONDITIONING | | H3 条件节点输出 |
| latent_image | LATENT | | 接 H3 Two-Stage Size（必须是空 latent，不支持 noise mask）。直接接普通空 latent 时按 0.75 和旧取整处理 |
| sampler_name | 下拉 | euler | 只列出能拆成两段的采样器，即每一步不依赖之前步骤的：`euler`（沿预测方向走一步的 ODE 采样）和 `lcm`（直接取预测的干净画面再加新噪声，DMAD 这类少步蒸馏 LoRA 训练时用的就是这个规则）。多步求解器不提供 |
| sigmas | SIGMAS | | 完整噪声表 |
| seed | INT | 0 | 低分辨率阶段的种子；接回时的新噪声用 seed+1 |
| cfg | FLOAT | 1.0 | 和 KSampler 一样 |
| split_step | INT | 6 | 低分辨率下采的步数，其余在全分辨率下采（两边至少各 1 步） |
| lift | 下拉 | `minimax_h3_latent_upscaler_3d_fp16.safetensors` | `models/latent_upscale_models/` 里的任意文件，或 `bilinear`（不加载模型）。只有 MiniMax H3 放大模型能加载，选错会给出说明 |
| noise_keep | FLOAT | 0 | 0 = 接回时全新噪声（推荐）；大于 0 时低频部分沿用低分辨率阶段的噪声（正交 FFT 频带拆分）。测试中没有可测改善 |
| model_low（选填） | MODEL | | 低分辨率阶段单独用的模型 |

输出：`latent`（AV，接 VAE 解码），`low_res_x0`、`lifted`（仅视频，用于检查）。

每一步在 ComfyUI 日志里打印一行 `[H3 two-stage]`：阶段、视频 / 音频 sigma、两路实测噪声强度、每步耗时；开头一行是整体计划。

## 依赖

- 带原生 MiniMax H3 支持的 ComfyUI（`comfy_extras/nodes_minimax_h3.py`）。
- 学习型放大需要带原生 MiniMax H3 latent 放大的 ComfyUI（PR #17412：`comfy/ldm/minimax/latent_upscaler.py`，
  节点 "MiniMax H3 Latent Upscale With Model"），以及模型权重（如 `minimax_h3_latent_upscaler_3d_fp16.safetensors`，LBH-123-AI 训练，
  下载：[huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)），
  放在 `ComfyUI/models/latent_upscale_models/`。没有它们时只能用 `lift = bilinear`。

采样节点在内部直接调用 ComfyUI 自己的加载器和放大节点，不用另外接节点，也不再需要第三方插件。和之前用的插件
（Comfyui_Minimax_h3_latent_Upscaler）相比，结果只差 fp16 舍入（相对 RMS 差 0.05%），峰值显存更低
（5 秒 1792×1024：1.6 GB 对 2.0 GB），因为 ComfyUI 的实现按 16 帧分块计算。

## 测试结论（H3，4 步加速 LoRA，8+1 步）

- 3D 放大模型准确：对照真值编码，误差约为双线性放大的一半（包括低频）；它对 ComfyUI latent 额外做的均值 / 方差归一化正是训练时的做法；整段一次放大和分块结果一致。
- 同种子同设置，和 SelfLift 的 H3 采样节点（`rho = 0`）结果等价。
- split_step 取 8 步中的 4 / 5 / 6，结果几乎一样；6 稍快。
- `lcm` 对 `euler`（4 步 LoRA 强度 0.4，8+1 步，split 6，2 个种子）：`lcm` 能正常跑完两段，但同种子构图不同，画面更亮（亮度 +4 到 +9），纹理闪烁约为 1.5–2 倍，动作略少。LoRA 只开部分强度时用 `euler`；`lcm` 适合蒸馏 LoRA 按其原设置使用（如 DMAD 强度 1、4 步）。
- 采样不能逐位复现：同一个图重跑，PSNR 约 26–28 dB，镜头相同。

## 致谢

- [comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift)（facok）：本节点精简的两段 H3 采样路径。未复制其代码。
- [Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)（LBH-123-AI）：学习型 3D latent 放大的模型权重，由 ComfyUI 原生实现运行。

## 许可

本仓库代码使用 MIT 许可（见 [LICENSE](LICENSE)）。它加载的模型权重遵循各自的条款。
