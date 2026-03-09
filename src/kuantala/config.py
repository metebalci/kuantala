"""Configuration dataclass for quantization jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DTYPES = ["FP8", "NVFP4"]

# Types allowed as per-component overrides (including skip)
COMPONENT_DTYPES = DTYPES + ["skip"]

# Calibration algorithms supported by modelopt
CALIB_ALGORITHMS = ["max", "smoothquant", "awq_lite", "awq_full", "mse"]

# Prompt sources — determines which HF dataset to use for calibration and eval
PROMPT_SOURCES = ["t2i", "t2v", "i2v", "ti2i"]


# Default keep patterns per model family.
# Based on NVIDIA modelopt examples:
# https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/diffusers/quantization/utils.py
DEFAULT_KEEPS: dict[str, list[str]] = {
    "wan": [
        "*patch_embedding*",
        "*condition_embedder*",
        "*proj_out*",
        "*blocks.[0-2].*",
        "*blocks.3[7-9].*",
    ],
    "flux": [
        "*proj_out*",
        "*time_text_embed*",
        "*context_embedder*",
        "*x_embedder*",
        "*norm_out*",
    ],
    "ltx": [
        "*proj_in*",
        "*time_embed*",
        "*caption_projection*",
        "*proj_out*",
        "*patchify_proj*",
        "*adaln_single*",
    ],
}

DEFAULT_KEEPS["cogvideox"] = [
    "*patch_embed*",
    "*time_embedding*",
    "*norm_final*",
    "*norm_out*",
    "*proj_out*",
]

DEFAULT_KEEPS["lumina-image"] = [
    "*x_embedder*",
    "*time_caption_embed*",
    "*norm_out*",
]

DEFAULT_KEEPS["omnigen"] = [
    "*patch_embedding*",
    "*time_token*",
    "*t_embedder*",
    "*embed_tokens*",
    "*norm_out*",
    "*proj_out*",
    "*layers.[0-2].*",
    "*layers.3[5-7].*",
]

DEFAULT_KEEPS["z-image"] = [
    "*t_embedder*",
    "*cap_embedder*",
    "*all_x_embedder*",
    "*all_final_layer*",
    "*layers.[0-2].*",
    "*layers.2[7-9].*",
]

DEFAULT_KEEPS["qwen-image"] = [
    "*time_text_embed*",
    "*img_in*",
    "*txt_in*",
    "*txt_norm*",
    "*norm_out*",
    "*proj_out*",
    "*transformer_blocks.[0-2].*",
    "*transformer_blocks.5[7-9].*",
]

DEFAULT_KEEPS_NAMES = list(DEFAULT_KEEPS.keys())

# Map HuggingFace model IDs to keep presets
_MODEL_ID_TO_KEEPS: dict[str, str] = {
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": "wan",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "wan",
    "black-forest-labs/FLUX.2-dev": "flux",
    "black-forest-labs/FLUX.1-Krea-dev": "flux",
    "Lightricks/LTX-2": "ltx",
    "Tongyi-MAI/Z-Image": "z-image",
    "Qwen/Qwen-Image-2512": "qwen-image",
    "Qwen/Qwen-Image-Edit-2511": "qwen-image",
    "Alpha-VLLM/Lumina-Image-2.0": "lumina-image",
    "THUDM/CogVideoX-2b": "cogvideox",
    "THUDM/CogVideoX-5b": "cogvideox",
    "THUDM/CogVideoX-2b-I2V": "cogvideox",
    "THUDM/CogVideoX-5b-I2V": "cogvideox",
    "Shitao/OmniGen-v1-diffusers": "omnigen",
}


def detect_default_keeps(model_source: str) -> str | None:
    """Auto-detect default keeps preset from HuggingFace model ID."""
    return _MODEL_ID_TO_KEEPS.get(model_source)


# Map HuggingFace model IDs to prompt sources
_MODEL_ID_TO_PROMPT_SOURCE: dict[str, str] = {
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": "i2v",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "t2v",
    "black-forest-labs/FLUX.2-dev": "t2i",
    "black-forest-labs/FLUX.1-Krea-dev": "t2i",
    "Lightricks/LTX-2": "t2v",
    "Tongyi-MAI/Z-Image": "t2i",
    "Qwen/Qwen-Image-2512": "t2i",
    "Qwen/Qwen-Image-Edit-2511": "ti2i",
    "Alpha-VLLM/Lumina-Image-2.0": "t2i",
    "THUDM/CogVideoX-2b": "t2v",
    "THUDM/CogVideoX-5b": "t2v",
    "THUDM/CogVideoX-2b-I2V": "i2v",
    "THUDM/CogVideoX-5b-I2V": "i2v",
    "Shitao/OmniGen-v1-diffusers": "ti2i",
}


def detect_prompt_source(model_source: str) -> str | None:
    """Auto-detect prompt source from HuggingFace model ID."""
    return _MODEL_ID_TO_PROMPT_SOURCE.get(model_source)


@dataclass
class QuantConfig:
    """Configuration for a quantization job."""

    model_source: str
    dtype: str
    output_dir: Path = Path("./output")

    # Per-component overrides ("skip" = don't touch, None = use main dtype)
    vae_dtype: str | None = "skip"
    te_dtype: str | None = "skip"
    ie_dtype: str | None = "skip"

    # Calibration
    algorithm: str = "max"  # calibration algorithm (max, smoothquant, awq_lite, awq_full, mse)
    calib_size: int = 256  # number of calibration prompts to use
    calib_steps: int = 30  # number of inference steps per prompt
    calib_resolution: tuple[int, int] = (480, 848)  # (height, width) for calibration
    calib_prompts: list[str] | None = None  # custom prompts (default: HF dataset)

    # Prompt source: t2i, t2v, i2v (auto-detected for known model IDs)
    prompt_source: str | None = None

    # Default keep preset (e.g. "wan", "flux", "ltx"); auto-detected for known model IDs
    default_keeps: str | None = None
    no_default_keeps: bool = False

    # Manual layer overrides: disable quantization on matched layer names
    keep: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.dtype not in DTYPES:
            raise ValueError(
                f"Unknown dtype {self.dtype!r}. "
                f"Choose from: {', '.join(DTYPES)}"
            )
        if self.algorithm not in CALIB_ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm {self.algorithm!r}. "
                f"Choose from: {', '.join(CALIB_ALGORITHMS)}"
            )
        for field_name, value in [("vae_dtype", self.vae_dtype), ("te_dtype", self.te_dtype), ("ie_dtype", self.ie_dtype)]:
            if value is not None and value not in COMPONENT_DTYPES:
                raise ValueError(
                    f"Unknown {field_name} {value!r}. "
                    f"Choose from: {', '.join(COMPONENT_DTYPES)}"
                )
