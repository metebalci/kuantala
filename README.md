# Kuantala

Kuantala quantizes diffusers-format generative models to NVFP4 (or FP8) using NVIDIA Model Optimizer. It can also convert the quantized output to ComfyUI-compatible format, evaluate quantization quality, and inspect model components.

NVFP4 is W4A4 (4-bit weights, 4-bit activations). Weights are stored as FP4 E2M1 packed in uint8, with FP8 per-block scales. Activation scales are calibrated during quantization, enabling native FP4 tensor core math on Blackwell+ GPUs for both memory savings and compute speedups.

FP8 is W8A8 (8-bit weights, 8-bit activations). Weights are stored as float8 E4M3 with per-tensor scales. Activation scales are calibrated during quantization, enabling native FP8 tensor core math on Hopper+ GPUs.

## Installation

Requires Python ≤ 3.12 due to `nvidia-modelopt` constraints. Tested on Linux.

```bash
python -m venv .venv
source .venv/bin/activate

# First install PyTorch with CUDA from https://pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu130

pip install kuantala
```

Kuantala requires models in **diffusers format** (with `model_index.json`). If a model has both a raw and a diffusers variant on HuggingFace, use the diffusers one (typically suffixed with `-Diffusers`).

## Quick Start

```bash
# Show supported formats and default keep presets
kuantala info

# Quantize to FP8 (~50% size, ~2x inference speed on Hopper+/Blackwell)
kuantala quantize Wan-AI/Wan2.2-I2V-A14B-Diffusers --dtype FP8

# Quantize to NVFP4 (~75% size, fastest on Blackwell)
kuantala quantize ./local-model --dtype NVFP4

# Inspect model components
kuantala components Wan-AI/Wan2.2-I2V-A14B-Diffusers

# Show model architecture from config (no weights downloaded)
kuantala config Wan-AI/Wan2.2-I2V-A14B-Diffusers

# Estimate output sizes for all formats
kuantala estimate Wan-AI/Wan2.2-I2V-A14B-Diffusers

# Convert NVFP4 output to ComfyUI format
kuantala convert ./output-wan/transformer-NVFP4.safetensors --remap-keys wan

# Evaluate quantization quality (compare original vs quantized)
kuantala eval Wan-AI/Wan2.2-I2V-A14B-Diffusers -q ./output-Wan-AI-Wan2.2-I2V-A14B-Diffusers

# Inspect tensors in a quantized file
kuantala tensors ./output/transformer-FP8.safetensors
```

## How It Works

Kuantala uses NVIDIA Model Optimizer to quantize model components:

1. **Load** the full diffusers pipeline (transformer, text encoder, VAE, etc.) in BF16 on CUDA
2. **Quantize** with Model Optimizer — inserts quantizer nodes and runs calibration by executing the pipeline with prompts from HuggingFace datasets (see [Prompt Sources](#prompt-sources)), producing realistic activations for optimal scale factor estimation
3. **Compress** — converts fake-quantized weights to real low-precision (FP8 or packed FP4)
4. **Save** as safetensors with actual quantized weights

The output files are genuinely smaller and load into VRAM at low precision. FP8 uses `float8_e4m3fn` tensors, NVFP4 uses packed `uint8` with block-wise FP8 scales.

## ComfyUI Conversion

Kuantala's NVFP4 output uses Model Optimizer's internal format. To use NVFP4 models in ComfyUI, convert them first:

```bash
kuantala convert ./output/transformer-NVFP4.safetensors
# Creates ./output/transformer-NVFP4-comfyui.safetensors

# Custom output path
kuantala convert ./output/transformer-NVFP4.safetensors -o ./comfy-model.safetensors
```

For models where diffusers uses different key names than the original (e.g. Wan), use `--remap-keys` to translate back to original names that ComfyUI expects:

```bash
kuantala convert ./wan-nvfp4/transformer-NVFP4.safetensors --remap-keys wan
```

The conversion performs:
- **Nibble swap** — reorders packed FP4 byte layout to match ComfyUI expectations
- **Scale tiling** — converts block scales from plain to cuBLAS tiled layout
- **Quantizer key renaming** — translates Model Optimizer quantizer names to ComfyUI conventions
- **Layer key remapping** — translates diffusers layer names to original names (with `--remap-keys`)
- **Metadata injection** — adds `.comfy_quant` entries required by ComfyUI

Full workflow (Wan example):
```bash
kuantala quantize Wan-AI/Wan2.2-I2V-A14B-Diffusers --dtype NVFP4
kuantala convert ./output-Wan-AI-Wan2.2-I2V-A14B-Diffusers/transformer-NVFP4.safetensors --remap-keys wan
# Load transformer-NVFP4-comfyui.safetensors in ComfyUI
```

## CLI Reference

### `kuantala analyze`

```
kuantala analyze [OPTIONS] MODEL
```

Uses modelopt's `auto_quantize` to find the optimal quantization format per layer given an effective bits constraint. No output files are saved — this is for exploring what format each layer should use.

> **Note:** `analyze` requires significantly more VRAM than `quantize` or `eval` because it computes gradients for sensitivity scoring. As a rough guide, expect ~3-4x the model size in VRAM.

| Option | Description |
|--------|-------------|
| `MODEL` | HuggingFace diffusers model ID or local directory path (required) |
| `-d, --dtypes` | Quantization formats to consider: `FP8`, `NVFP4` (repeatable, default: `NVFP4`) |
| `--effective-bits` | Target average bits per parameter (default: `4.8`) |
| `--nprompts N` | Number of pipeline runs to capture inputs (default: 8) |
| `--nsteps N` | Inference steps per pipeline run (default: 10) |
| `--resolution` | Calibration resolution: `480p`, `540p`, `720p`, `1080p`, `4k`, or `HEIGHTxWIDTH` (default: `480p`) |
| `--offload` | CPU offload mode: `model` (component-level) or `layers` (layer-level, slower but less VRAM) |

```bash
# Analyze optimal format mix between FP8 and NVFP4
kuantala analyze Wan-AI/Wan2.2-I2V-A14B-Diffusers -d FP8 -d NVFP4

# Target higher quality (more bits)
kuantala analyze Wan-AI/Wan2.2-I2V-A14B-Diffusers -d FP8 -d NVFP4 --effective-bits 6.0
```

### `kuantala components`

```
kuantala components [OPTIONS] MODEL
```

| Option | Description |
|--------|-------------|
| `MODEL` | HuggingFace diffusers model ID or local directory path (required) |

### `kuantala config`

```
kuantala config MODEL
```

Shows model architecture from config: full module hierarchy with layer types, shapes, and parameter counts. Only downloads `config.json` files — no model weights are downloaded.

### `kuantala convert`

```
kuantala convert [OPTIONS] INPUT
```

| Option | Description |
|--------|-------------|
| `INPUT` | Path to a Model Optimizer NVFP4 `.safetensors` file (required) |
| `-o, --output` | Output file path (default: `{input_stem}-comfyui.safetensors`) |
| `--remap-keys` | Remap diffusers key names to original: `wan` |

### `kuantala estimate`

```
kuantala estimate MODEL
```

Estimates output sizes for all formats from parameter counts. No actual quantization is performed. VAE is excluded (skipped by default). The model must be downloaded locally.

### `kuantala eval`

```
kuantala eval [OPTIONS] MODEL
kuantala evaluate [OPTIONS] MODEL
```

Compares original vs quantized pipeline outputs using PSNR and SSIM metrics. Runs the pipeline with fixed seeds on both original and quantized models to measure quality loss. Eval prompts come from HuggingFace datasets selected by `--psrc` (see [Prompt Sources](#prompt-sources)).

| Option | Description |
|--------|-------------|
| `MODEL` | HuggingFace diffusers model ID or local directory path (required) |
| `-q, --quantized-dir` | Directory with quantized safetensors from `kuantala quantize` (required) |
| `--prompts FILE` | File with eval prompts, one per line (default: HF dataset test split) |
| `--nprompts N` | Number of eval prompts (default: 16) |
| `--nsteps N` | Inference steps (default: 30) |
| `--resolution` | Resolution: `480p`, `540p`, `720p`, `1080p`, `4k`, or `HEIGHTxWIDTH` (default: `480p`) |
| `--decode` | Also compare decoded pixel-space outputs (default: latent only) |
| `--psrc` | Prompt source: `t2i`, `t2v`, `i2v`, `ti2i` (auto-detected for known HF model IDs) |
| `--offset N` | Dataset offset for eval prompts to avoid overlap with calibration (default: 1024) |
| `--offload` | CPU offload mode: `model` (component-level) or `layers` (layer-level, slower but less VRAM) |

```bash
# Basic eval
kuantala eval Wan-AI/Wan2.2-I2V-A14B-Diffusers -q ./output-wan

# With pixel-space comparison
kuantala eval Wan-AI/Wan2.2-I2V-A14B-Diffusers -q ./output-wan --decode

# Custom prompts
kuantala eval ./local-model -q ./output --prompts eval_prompts.txt --nprompts 8
```

### `kuantala info`

```
kuantala info
```

Shows supported quantization formats, default keep presets, and known models.

### `kuantala quantize`

```
kuantala quantize [OPTIONS] MODEL
```

| Option | Description |
|--------|-------------|
| `MODEL` | HuggingFace diffusers model ID or local directory path (required) |
| `-d, --dtype` | Target format: `FP8`, `NVFP4` (default: `NVFP4`) |
| `-o, --output` | Output directory (default: `output-<MODEL_ID>`) |
| `--vae-dtype` | VAE dtype (default: `skip`). Same choices as `--dtype` plus `skip` |
| `--te-dtype` | Text encoder dtype (default: `skip`) |
| `--ie-dtype` | Image encoder dtype (default: `skip`) |
| `--cfg` | Quantization config preset: `default`, `awq_lite`, `awq_clip`, `awq_full` (default: `default`, see below) |
| `--algorithm` | Advanced: override calibration algorithm (default: auto, see below) |

Quantization config presets (`--cfg`):
- `default` — Standard quantization with `max` calibration. Fastest, good baseline.
- `awq_lite` — Activation-aware weight quantization with lightweight search. Better quality than default, moderate overhead.
- `awq_clip` — AWQ with clipping-based optimization. Good quality/speed tradeoff.
- `awq_full` — AWQ with full search. Slowest, best quality. NVFP4 only.

The `--algorithm` option is an advanced setting that overrides the calibration algorithm used by the selected config preset. Most users should use `--cfg` instead. Available algorithms: `max`, `smoothquant`, `awq_lite`, `awq_full`, `mse`.
| `--keep PATTERN` | Disable quantization on layers matching this glob pattern (repeatable) |
| `--use-default-keeps` | Apply preset keep patterns: `wan`, `flux`, `ltx`, `z-image`, `qwen-image` (auto-detected for known HF model IDs) |
| `--no-default-keeps` | Disable auto-detected default keep patterns |
| `--prompts FILE` | File with calibration prompts, one per line (default: HF dataset) |
| `--nprompts N` | Number of calibration prompts (default: 32) |
| `--nsteps N` | Inference steps per calibration prompt (default: 30) |
| `--resolution` | Calibration resolution: `480p`, `540p`, `720p`, `1080p`, `4k`, or `HEIGHTxWIDTH` (default: `480p`) |
| `--psrc` | Prompt source: `t2i`, `t2v`, `i2v`, `ti2i` (auto-detected for known HF model IDs) |
| `--offload` | CPU offload mode: `model` (component-level) or `layers` (layer-level, slower but less VRAM) |

### `kuantala tensors`

```
kuantala tensors FILE_PATH
```

Shows per-tensor detail: name, dtype, shape, and parameter count. Also shows a dtype summary.

### Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable debug logging |

## Per-Component Control

Kuantala detects and quantizes the following component types from diffusers models:

| Component | Flag | Default |
|-----------|------|---------|
| Transformer / UNet | `--dtype` | `NVFP4` |
| Text encoder | `--te-dtype` | `skip` |
| Image encoder | `--ie-dtype` | `skip` |
| VAE | `--vae-dtype` | `skip` |

Schedulers, tokenizers, and other non-neural components are always skipped. VAE is skipped by default because quantizing it causes visible artifacts.

```bash
kuantala quantize black-forest-labs/FLUX.1-dev \
    --dtype FP8 \
    --vae-dtype skip \
    --te-dtype FP8
```

## Layer-Level Control

Use `--keep` to disable quantization on specific layers by glob pattern. Matched layers stay at their original precision. Time embeddings, conditioning projections, and input/output layers are small but sensitive — keeping them unquantized has negligible size impact but helps quality.

## Default Keeps

For known HuggingFace model IDs, kuantala automatically applies preset keep patterns that disable quantization on sensitive layers. These are based on [NVIDIA Model Optimizer's example settings](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/diffusers/quantization/utils.py).

| Preset | Models | Kept layers |
|--------|--------|-------------|
| `wan` | Wan2.2-I2V-A14B, Wan2.2-T2V-A14B | patch_embedding, condition_embedder, proj_out, first/last 3 blocks |
| `flux` | FLUX.1-dev/schnell/Kontext-dev/Krea-dev, FLUX.2-dev/klein | proj_out, time_text_embed, context_embedder, x_embedder, norm_out |
| `ltx` | LTX-2 | proj_in, time_embed, caption_projection, proj_out, patchify_proj, adaln_single |
| `cogvideox` | CogVideoX-5b, CogVideoX-5b-I2V | patch_embed, time_embedding, norm_final, norm_out, proj_out |
| `z-image` | Z-Image, Z-Image-Turbo | t_embedder, cap_embedder, all_x_embedder, all_final_layer, first/last 3 layers |
| `qwen-image` | Qwen-Image-2512, Qwen-Image-Edit-2511 | time_text_embed, img_in, txt_in, txt_norm, norm_out, proj_out, first/last 3 transformer_blocks |
| `sdxl` | SDXL 1.0, SDXL Turbo, SVD | time_emb_proj, time_embedding, conv_in, conv_out, conv_shortcut, add_embedding, pos_embed |

Use `--use-default-keeps <preset>` to explicitly select a preset (e.g. for local paths). Use `--no-default-keeps` to disable auto-detection.

```bash
# Wan 2.2 I2V / T2V 14B (keeps auto-detected)
kuantala quantize Wan-AI/Wan2.2-I2V-A14B-Diffusers --dtype NVFP4

# FLUX.1 dev
kuantala quantize black-forest-labs/FLUX.1-dev --dtype NVFP4

# FLUX.2 dev
kuantala quantize black-forest-labs/FLUX.2-dev --dtype NVFP4

# LTX-2
kuantala quantize Lightricks/LTX-2 --dtype NVFP4

# Z-Image
kuantala quantize Tongyi-MAI/Z-Image --dtype NVFP4

# Qwen-Image-2512
kuantala quantize Qwen/Qwen-Image-2512 --dtype NVFP4

# CogVideoX-5b
kuantala quantize zai-org/CogVideoX-5b --dtype FP8

# SDXL 1.0
kuantala quantize stabilityai/stable-diffusion-xl-base-1.0 --dtype FP8

# With AWQ config preset for better quality
kuantala quantize Wan-AI/Wan2.2-I2V-A14B-Diffusers --dtype NVFP4 --cfg awq_lite
```

Norms (`FP32LayerNorm`, `RMSNorm`) and embeddings (`WanRotaryPosEmbed`) are already kept at original precision by Model Optimizer.

## Prompt Sources

Calibration and evaluation prompts are loaded from HuggingFace datasets. Use `--psrc` to select the source, or let kuantala auto-detect it from known model IDs.

| Source | Dataset | License | Use case |
|--------|---------|---------|----------|
| `t2i` | [poloclub/diffusiondb](https://huggingface.co/datasets/poloclub/diffusiondb) | CC0-1.0 | Text-to-image models (FLUX, Z-Image, Qwen-Image) |
| `t2v` | [WenhaoWang/VidProM](https://huggingface.co/datasets/WenhaoWang/VidProM) | CC-BY-NC-4.0 | Text-to-video models (Wan T2V, LTX) |
| `i2v` | [WenhaoWang/TIP-I2V](https://huggingface.co/datasets/WenhaoWang/TIP-I2V) | CC-BY-NC-4.0 | Image-to-video models (Wan I2V) |
| `ti2i` | [UCSC-VLAA/HQ-Edit](https://huggingface.co/datasets/UCSC-VLAA/HQ-Edit) | CC-BY-NC-4.0 | Text+image-to-image models (Qwen-Image-Edit) |

The first 10240 entries from each dataset are used for calibration, the second 10240 for evaluation. For I2V and TI2I sources, the dataset provides both text prompts and conditioning images.

Datasets are downloaded on demand from HuggingFace and cached locally by the `datasets` library. Custom prompts can be provided via `--prompts FILE` to override the default dataset.

## Python API

```python
from pathlib import Path
from kuantala import QuantConfig, quantize

config = QuantConfig(
    model_source="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    dtype="NVFP4",
    output_dir=Path("./output-wan"),
    keep=[
        "*patch_embedding*", "*condition_embedder*", "*proj_out*",
        "*blocks.0.*", "*blocks.1.*", "*blocks.2.*",
        "*blocks.37.*", "*blocks.38.*", "*blocks.39.*",
    ],
)
output_files = quantize(config)
```

## Supported Formats

| Format | Description | Size vs FP16 | GPU Requirement |
|--------|-------------|--------------|-----------------|
| FP8 | 8-bit floating point (E4M3) | ~50% | Hopper+ (RTX 4000+) |
| NVFP4 | NVIDIA 4-bit floating point | ~25% | Blackwell+ (RTX 5000+) |

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev]"
pytest tests/
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
