"""Configuration dataclass for quantization jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DTYPES = ["FP8", "NVFP4"]

# Types allowed as per-component overrides (including skip)
COMPONENT_DTYPES = DTYPES + ["skip"]

# Quantization config presets (maps to modelopt CFG dicts)
QUANT_CONFIGS = ["default", "awq_lite", "awq_clip", "awq_full"]

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

DEFAULT_KEEPS["sdxl"] = [
    "*time_emb_proj*",
    "*time_embedding*",
    "*conv_in*",
    "*conv_out*",
    "*conv_shortcut*",
    "*add_embedding*",
    "*pos_embed*",
]

DEFAULT_KEEPS_NAMES = list(DEFAULT_KEEPS.keys())


# Per-model defaults: keeps, prompt source, resolution, steps, num_frames.
# resolution is (height, width).
MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    # Wan models
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": {
        "keeps": "wan", "psrc": "i2v", "size": "14B*",
        "resolution": (720, 1280), "steps": 40, "num_frames": 81,
    },
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": {
        "keeps": "wan", "psrc": "t2v", "size": "14B*",
        "resolution": (720, 1280), "steps": 40, "num_frames": 81,
    },
    # FLUX models
    "black-forest-labs/FLUX.1-dev": {
        "keeps": "flux", "psrc": "t2i", "size": "12B",
        "resolution": (1024, 1024), "steps": 50,
    },
    "black-forest-labs/FLUX.1-schnell": {
        "keeps": "flux", "psrc": "t2i", "size": "12B",
        "resolution": (1024, 1024), "steps": 4,
    },
    "black-forest-labs/FLUX.1-Kontext-dev": {
        "keeps": "flux", "psrc": "ti2i", "size": "12B",
        "resolution": (1024, 1024), "steps": 28,
    },
    "black-forest-labs/FLUX.1-Krea-dev": {
        "keeps": "flux", "psrc": "t2i", "size": "12B",
        "resolution": (1024, 1024), "steps": 28,
    },
    "black-forest-labs/FLUX.2-dev": {
        "keeps": "flux", "psrc": "t2i", "size": "32B",
        "resolution": (1024, 1024), "steps": 28,
    },
    "black-forest-labs/FLUX.2-klein-base-9B": {
        "keeps": "flux", "psrc": "t2i", "size": "9B",
        "resolution": (1024, 1024), "steps": 50,
    },
    # Stability AI models
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "keeps": "sdxl", "psrc": "t2i", "size": "2.6B",
        "resolution": (1024, 1024), "steps": 30,
    },
    "stabilityai/sdxl-turbo": {
        "keeps": "sdxl", "psrc": "t2i", "size": "2.6B",
        "resolution": (512, 512), "steps": 1,
    },
    "stabilityai/stable-video-diffusion-img2vid-xt-1-1": {
        "keeps": "sdxl", "psrc": "i2v", "size": "1.5B",
        "resolution": (576, 1024), "steps": 25, "num_frames": 25,
    },
    # CogVideoX models
    "zai-org/CogVideoX-5b": {
        "keeps": "cogvideox", "psrc": "t2v", "size": "5B",
        "resolution": (480, 720), "steps": 50, "num_frames": 49,
    },
    "zai-org/CogVideoX-5b-I2V": {
        "keeps": "cogvideox", "psrc": "i2v", "size": "5B",
        "resolution": (480, 720), "steps": 50, "num_frames": 49,
    },
    # Other models
    "HiDream-ai/HiDream-I1-Full": {
        "psrc": "t2i", "size": "17B",
        "resolution": (1024, 1024), "steps": 50,
    },
    "HiDream-ai/HiDream-E1-1": {
        "psrc": "ti2i", "size": "17B",
        "resolution": (768, 768), "steps": 28,
    },
    "Lightricks/LTX-2": {
        "keeps": "ltx", "psrc": "t2v", "size": "19B",
        "resolution": (512, 768), "steps": 40, "num_frames": 121,
    },
    "nvidia/Cosmos-Predict2-14B-Text2Image": {
        "psrc": "t2i", "size": "14B",
        "resolution": (768, 1360), "steps": 35,
    },
    "nvidia/Cosmos-Predict2-14B-Video2World": {
        "psrc": "i2v", "size": "14B",
        "resolution": (704, 1280), "steps": 35, "num_frames": 93,
    },
    "Qwen/Qwen-Image-2512": {
        "keeps": "qwen-image", "psrc": "t2i", "size": "20B",
        "resolution": (1328, 1328), "steps": 50,
    },
    "Qwen/Qwen-Image-Edit-2511": {
        "keeps": "qwen-image", "psrc": "ti2i", "size": "20B",
        "resolution": (1328, 1328), "steps": 50,
    },
    "Tongyi-MAI/Z-Image": {
        "keeps": "z-image", "psrc": "t2i", "size": "6B",
        "resolution": (720, 1280), "steps": 50,
    },
    "Tongyi-MAI/Z-Image-Turbo": {
        "keeps": "z-image", "psrc": "t2i", "size": "6B",
        "resolution": (1024, 1024), "steps": 9,
    },
}


def get_model_defaults(model_source: str) -> dict[str, Any]:
    """Get all defaults for a known model, or empty dict for unknown models."""
    return MODEL_DEFAULTS.get(model_source, {})


def detect_default_keeps(model_source: str) -> str | None:
    """Auto-detect default keeps preset from HuggingFace model ID."""
    return MODEL_DEFAULTS.get(model_source, {}).get("keeps")


def detect_prompt_source(model_source: str) -> str | None:
    """Auto-detect prompt source from HuggingFace model ID."""
    return MODEL_DEFAULTS.get(model_source, {}).get("psrc")


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
    cfg: str = "default"  # quantization config preset (default, awq_lite, awq_clip, awq_full)
    alpha_step: float | None = None  # awq_full: search step size (default: 0.1, smaller = finer search)
    calib_size: int = 32  # number of calibration prompts to use
    calib_steps: int = 30  # number of inference steps per prompt
    calib_resolution: tuple[int, int] = (480, 848)  # (height, width) for calibration
    num_frames: int | None = None  # video models: frames per generation (auto-detected)
    offload: str | None = None  # "model" (component-level) or "layers" (layer-level) CPU offload
    calib_prompts: list[str] | None = None  # custom prompts (default: HF dataset)
    calib_images: list[str | None] | None = None  # custom images for i2v/ti2i (from prompts file)

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
        if self.cfg not in QUANT_CONFIGS:
            raise ValueError(
                f"Unknown cfg {self.cfg!r}. "
                f"Choose from: {', '.join(QUANT_CONFIGS)}"
            )
        for field_name, value in [("vae_dtype", self.vae_dtype), ("te_dtype", self.te_dtype), ("ie_dtype", self.ie_dtype)]:
            if value is not None and value not in COMPONENT_DTYPES:
                raise ValueError(
                    f"Unknown {field_name} {value!r}. "
                    f"Choose from: {', '.join(COMPONENT_DTYPES)}"
                )
