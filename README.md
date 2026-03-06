# Kuantala

Quantize diffusion models (Wan2.x, FLUX, Stable Diffusion, etc.) to FP8 and NVFP4 formats using NVIDIA modelopt.

## Installation

```bash
# First install PyTorch with CUDA from https://pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu130

pip install kuantala
```

Requires Python ≤ 3.12 due to `nvidia-modelopt` constraints.

Kuantala requires models in **diffusers format** (with `model_index.json`). If a model has both a raw and a diffusers variant on HuggingFace, use the diffusers one (typically suffixed with `-Diffusers`).

## Quick Start

```bash
# Quantize to FP8 (~50% size, ~2x inference speed on Hopper+/Blackwell)
kuantala quantize Wan-AI/Wan2.1-I2V-14B-Diffusers --dtype FP8 --output ./wan-fp8

# Quantize to NVFP4 (~75% size, fastest on Blackwell)
kuantala quantize ./local-model --dtype NVFP4 --output ./model-fp4

# Convert FP32 model to FP16
kuantala quantize ./old-model --dtype FP16 --output ./model-fp16

# Inspect model components
kuantala components Wan-AI/Wan2.1-I2V-14B-Diffusers

# Show model architecture from config (no weights downloaded)
kuantala config Wan-AI/Wan2.1-I2V-14B-Diffusers

# Estimate output sizes for all formats
kuantala estimate Wan-AI/Wan2.1-I2V-14B-Diffusers

# Inspect tensors in a quantized file
kuantala tensors ./output/transformer-FP8.safetensors

# List available formats
kuantala formats
```

## How It Works

Kuantala uses NVIDIA modelopt to quantize diffusion model components:

1. **Load** each component (transformer, text encoder, etc.) via diffusers/transformers
2. **Quantize** with modelopt — inserts quantizer nodes and runs calibration forward passes with random data to determine optimal scale factors
3. **Compress** — converts fake-quantized weights to real low-precision (FP8 or packed FP4)
4. **Save** as safetensors with actual quantized weights

The output files are genuinely smaller and load into VRAM at low precision. FP8 uses `float8_e4m3fn` tensors, NVFP4 uses packed `uint8` with block-wise FP8 scales.

## ComfyUI Conversion

Kuantala's NVFP4 output uses modelopt's internal format. To use NVFP4 models in ComfyUI, convert them first:

```bash
kuantala convert ./output/transformer-NVFP4.safetensors
# Creates ./output/transformer-NVFP4-comfyui.safetensors

# Custom output path
kuantala convert ./output/transformer-NVFP4.safetensors -o ./comfy-model.safetensors
```

The conversion performs:
- **Nibble swap** — reorders packed FP4 byte layout to match ComfyUI expectations
- **Scale tiling** — converts block scales from plain to cuBLAS tiled layout
- **Key renaming** — translates modelopt tensor names to ComfyUI conventions
- **Metadata injection** — adds `.comfy_quant` entries required by ComfyUI

Full workflow:
```bash
kuantala quantize Wan-AI/Wan2.1-I2V-14B-Diffusers --dtype NVFP4 --output ./wan-nvfp4
kuantala convert ./wan-nvfp4/transformer-NVFP4.safetensors
# Load transformer-NVFP4-comfyui.safetensors in ComfyUI
```

## CLI Reference

### `kuantala quantize`

```
kuantala quantize [OPTIONS] MODEL
```

| Option | Description |
|--------|-------------|
| `MODEL` | HuggingFace diffusers model ID or local directory path (required) |
| `-d, --dtype` | Target format: `FP8`, `NVFP4`, `FP16`, `BF16` (required) |
| `-o, --output` | Output directory (default: `./output`) |
| `--vae-dtype` | VAE dtype (default: `skip`). Same choices as `--dtype` plus `skip` |
| `--te-dtype` | Text encoder dtype (default: `skip`) |
| `--ie-dtype` | Image encoder dtype (default: same as `--dtype`) |
| `--keep PATTERN` | Disable quantization on layers matching this glob pattern (repeatable) |

### `kuantala components`

```
kuantala components [OPTIONS] MODEL
```

| Option | Description |
|--------|-------------|
| `MODEL` | HuggingFace diffusers model ID or local directory path (required) |

### `kuantala estimate`

```
kuantala estimate MODEL
```

Estimates output sizes for all formats from parameter counts. No actual quantization is performed. VAE is excluded (skipped by default). The model must be downloaded locally.

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
| `INPUT` | Path to a modelopt NVFP4 `.safetensors` file (required) |
| `-o, --output` | Output file path (default: `{input_stem}-comfyui.safetensors`) |

### `kuantala tensors`

```
kuantala tensors FILE_PATH
```

Shows per-tensor detail: name, dtype, shape, and parameter count. Also shows a dtype summary.

### `kuantala formats`

```
kuantala formats
```

Lists available quantization formats with descriptions.

### Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable debug logging |

## Per-Component Control

Kuantala detects and quantizes the following component types from diffusers models:

| Component | Flag | Default |
|-----------|------|---------|
| Transformer / UNet | `--dtype` | required |
| Text encoder | `--te-dtype` | `skip` |
| Image encoder | `--ie-dtype` | same as `--dtype` |
| VAE | `--vae-dtype` | `skip` |

Schedulers, tokenizers, and other non-neural components are always skipped. VAE is skipped by default because quantizing it causes visible artifacts.

```bash
kuantala quantize black-forest-labs/FLUX.1-dev \
    --dtype FP8 \
    --vae-dtype skip \
    --te-dtype FP16
```

## Layer-Level Control

Use `--keep` to disable quantization on specific layers by glob pattern. Matched layers stay at their original precision (FP16/BF16). Time embeddings, conditioning projections, and input/output layers are small but sensitive — keeping them unquantized has negligible size impact but helps quality.

## Model Recipes

### Wan 2.2 I2V 14B

```bash
kuantala quantize Wan-AI/Wan2.2-I2V-A14B-Diffusers --dtype FP8 \
    --keep "*condition_embedder*" \
    --keep "*patch_embedding*"
```

Key layers kept at full precision:
- `condition_embedder` — time embedding, text projection, conditioning (232M params)
- `patch_embedding` — input Conv3d

Norms (`FP32LayerNorm`, `RMSNorm`) and embeddings (`WanRotaryPosEmbed`) are already kept at original precision by modelopt.

## Python API

```python
from pathlib import Path
from kuantala import QuantConfig, quantize

config = QuantConfig(
    model_source="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    dtype="FP8",
    output_dir=Path("./output"),
    keep=["*condition_embedder*", "*patch_embedding*"],
)
output_files = quantize(config)
```

## Supported Formats

| Format | Description | Size vs FP16 | GPU Requirement |
|--------|-------------|--------------|-----------------|
| FP8 | 8-bit floating point (E4M3) | ~50% | Hopper+ (RTX 4000+) |
| NVFP4 | NVIDIA 4-bit floating point | ~25% | Blackwell (RTX 5000+) |
| FP16 | 16-bit float (passthrough) | 100% | Any |
| BF16 | Brain float 16 (passthrough) | 100% | Ampere+ |

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
