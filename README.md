# Kuantala

Quantize diffusion models (Wan2.x, FLUX, Stable Diffusion, etc.) to GGUF, MXFP8, and NVFP4 formats.

## Installation

```bash
# GGUF quantization (no torch required)
pip install kuantala[gguf]

# With HuggingFace Hub download support
pip install kuantala[gguf,hub]

# NVIDIA MXFP8/NVFP4
# First install PyTorch with CUDA from https://pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install kuantala[nvidia]

# Optional (Windows only): install Triton for faster quantization simulations
pip install triton-windows

# Everything (hub, gguf, nvidia)
pip install kuantala[all]
```

For local models only, `huggingface-hub` is not needed. PyTorch is not declared as a dependency since it requires a platform-specific CUDA build — install it manually before `kuantala[nvidia]` or `kuantala[all]`.

> **Note:** The NVIDIA backend requires Python ≤ 3.12 due to `nvidia-modelopt` package constraints. The GGUF backend works with any Python version.

Kuantala requires models in **diffusers format** (with `model_index.json`). If a model has both a raw and a diffusers variant on HuggingFace, use the diffusers one (typically suffixed with `-Diffusers`).

## Quick Start

```bash
# Quantize to GGUF Q4_K
kuantala quantize Wan-AI/Wan2.1-I2V-14B-Diffusers --dtype Q4_K --output ./wan-q4

# Quantize to NVIDIA MXFP8
kuantala quantize ./local-model --dtype MXFP8 --output ./model-fp8

# Inspect model components
kuantala components Wan-AI/Wan2.1-I2V-14B-Diffusers

# Show model architecture from its config (requires torch + diffusers)
kuantala config Wan-AI/Wan2.1-I2V-14B-Diffusers

# Estimate output sizes for all formats
kuantala estimate Wan-AI/Wan2.1-I2V-14B-Diffusers

# Inspect tensors in a quantized file
kuantala tensors ./output/transformer-Q4_K.gguf

# List available formats
kuantala formats
```

## CLI Reference

### `kuantala quantize`

```
kuantala quantize [OPTIONS] MODEL
```

| Option | Description |
|--------|-------------|
| `MODEL` | HuggingFace diffusers model ID (e.g. `Wan-AI/Wan2.1-I2V-14B-Diffusers`) or local directory path in diffusers format (required) |
| `-d, --dtype` | Target quantization type (required). GGUF: `Q2_K`, `Q3_K`, `Q4_0`, `Q4_K`, `Q5_0`, `Q5_K`, `Q6_K`, `Q8_0`. NVIDIA: `MXFP8`, `NVFP4` |
| `-o, --output` | Output directory (default: `./output`) |
| `--vae-dtype` | VAE quantization dtype (default: `skip`). Accepts any dtype above plus `F16`, `F32`, `BF16`, `skip` |
| `--te-dtype` | Text encoder quantization dtype (default: same as `--dtype`). Same choices as `--vae-dtype` |
| `--ie-dtype` | Image encoder quantization dtype (default: same as `--dtype`). Same choices as `--vae-dtype` |
| `--no-heuristics` | Disable heuristic-based mixed precision (on by default) |
| `--statistics LEVEL` | Preserve statistically sensitive layers: `low`, `medium`, or `high` (off by default) |
| `--no-calibration` | Disable calibration forward passes (on by default for NVIDIA backend) |
| `--calibration-data PATH` | Path to calibration data directory |
| `--keep TEXT` | Manual layer override: `pattern:dtype` (repeatable) |

### `kuantala components`

```
kuantala components [OPTIONS] MODEL
```

| Option | Description |
|--------|-------------|
| `MODEL` | HuggingFace diffusers model ID (e.g. `Wan-AI/Wan2.1-I2V-14B-Diffusers`) or local directory path in diffusers format (required) |

### `kuantala estimate`

```
kuantala estimate [OPTIONS] MODEL
```

| Option | Description |
|--------|-------------|
| `MODEL` | HuggingFace diffusers model ID (e.g. `Wan-AI/Wan2.1-I2V-14B-Diffusers`) or local directory path in diffusers format (required) |

Estimates output file sizes for common quantization formats: key GGUF types (Q4_K, Q5_K, Q6_K, Q8_0) and NVIDIA types (MXFP8, NVFP4) if torch + nvidia-modelopt are installed. Shows columns for heuristics-only and heuristics + statistics at each level (low/medium/high). VAE is excluded (skipped by default). Reads actual weight values to compute statistics, so the model must be downloaded.

### `kuantala config`

```
kuantala config [OPTIONS] MODEL
```

| Option | Description |
|--------|-------------|
| `MODEL` | HuggingFace diffusers model ID (e.g. `Wan-AI/Wan2.1-I2V-14B-Diffusers`) or local directory path in diffusers format (required) |

Shows the model architecture from its config: full module hierarchy with layer types, shapes, and parameter counts. Only downloads `config.json` files — no model weights are downloaded or loaded. Requires `torch` and `diffusers`/`transformers` to be installed.

### `kuantala formats`

```
kuantala formats
```

Lists all available quantization formats with their backend and description.

### `kuantala tensors`

```
kuantala tensors [OPTIONS] FILE_PATH
```

| Option | Description |
|--------|-------------|
| `FILE_PATH` | Path to a `.safetensors` or `.gguf` file (required) |

Shows per-tensor detail: name, dtype, shape, and parameter count. Also shows a dtype summary with parameter distribution.

### Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable debug logging |

## Per-Component Control

Kuantala detects and quantizes the following component types from diffusers models:

| Component | Flag | Default |
|-----------|------|---------|
| Transformer / UNet | `--dtype` | required |
| Text encoder | `--te-dtype` | same as `--dtype` |
| Image encoder | `--ie-dtype` | same as `--dtype` |
| VAE | `--vae-dtype` | `skip` |

Schedulers, tokenizers, and other non-neural components are always skipped. VAE is skipped by default because quantizing it below FP16 causes visible artifacts.

```bash
kuantala quantize black-forest-labs/FLUX.1-dev \
    --dtype Q4_K \
    --vae-dtype skip \
    --te-dtype Q8_0
```

## Mixed Quantization

Kuantala supports three methods for mixed-precision quantization, which can be combined. When multiple methods are active, a layer is preserved at higher precision if *any* method flags it. Manual `--keep` rules always take highest priority.

### Heuristics (on by default)

Preserves known-sensitive layer types at F16 based on diffusion model research: normalization layers, attention QKV projections, timestep embeddings, and input/output projections. These layers are small but disproportionately affect output quality.

Disable with `--no-heuristics`.

### Statistics (off by default)

Analyzes the actual weight values in each layer to find statistical outliers — layers with unusually high outlier ratios (many weights far from the mean) are most harmed by quantization and get preserved at F16.

The sensitivity level controls the threshold:

| Level | Threshold | Effect |
|-------|-----------|--------|
| `low` | > 2.5 std devs | Only extreme outliers, fewest layers preserved |
| `medium` | > 2.0 std devs | Clear outliers |
| `high` | > 1.5 std devs | Moderate outliers, most layers preserved |

Enable with `--statistics medium` (or `low`/`high`).

### Calibration (on by default, NVIDIA backend only)

Runs forward passes with random data through the model so that quantizers can observe actual activation ranges and set optimal scale factors. This doesn't decide *which* layers to preserve — it improves *how* the quantized layers are scaled.

Not applicable to GGUF (which is weight-only quantization with scales computed directly from weight values). Disable with `--no-calibration`.

### Examples

```bash
# Default: heuristics + calibration (NVIDIA) enabled
kuantala quantize model --dtype MXFP8

# Add statistics-based layer preservation
kuantala quantize model --dtype Q4_K --statistics medium

# Disable defaults if needed
kuantala quantize model --dtype Q4_K --no-heuristics
kuantala quantize model --dtype MXFP8 --no-calibration

# Manual overrides (always highest priority)
kuantala quantize model --dtype Q4_K \
    --keep "norm_*:F16" --keep "attn_*:Q8_0"
```

## Python API

```python
from pathlib import Path
from kuantala import QuantConfig, quantize

config = QuantConfig(
    model_source="Wan-AI/Wan2.1-I2V-14B-Diffusers",
    dtype="Q4_K",
    vae_dtype="skip",
    output_dir=Path("./output"),
    # heuristics=True and calibration=True by default
    statistics="medium",
    keep=["norm_*:F16"],
)
output_files = quantize(config)
```

## Supported Formats

### GGUF (for llama.cpp / stable-diffusion.cpp)

| Format | Description |
|--------|-------------|
| Q2_K   | 2-bit K-means quantization |
| Q3_K   | 3-bit K-means quantization |
| Q4_0   | 4-bit basic quantization |
| Q4_K   | 4-bit K-means quantization |
| Q5_0   | 5-bit basic quantization |
| Q5_K   | 5-bit K-means quantization |
| Q6_K   | 6-bit K-means quantization |
| Q8_0   | 8-bit basic quantization |

### NVIDIA (requires torch + nvidia-modelopt)

| Format | Description |
|--------|-------------|
| MXFP8  | Microscaling FP8 (requires Hopper+) |
| NVFP4  | NVIDIA FP4 (requires Blackwell) |

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
