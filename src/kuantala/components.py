"""Detect model components from model_index.json or directory structure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kuantala.utils import get_logger

log = get_logger(__name__)


@dataclass
class ModelComponent:
    """A single component of a diffusion pipeline."""

    name: str  # e.g. "transformer", "vae", "text_encoder"
    path: Path  # directory containing safetensors
    component_type: str  # "transformer", "unet", "vae", "text_encoder", "scheduler", "other"
    library: str | None = None  # e.g. "diffusers", "transformers"
    class_name: str | None = None  # e.g. "AutoencoderKL", "CLIPTextModel"


@dataclass
class ModelInfo:
    """Parsed model structure."""

    root: Path
    components: list[ModelComponent]
    model_type: str | None = None  # e.g. "FluxPipeline", "WanPipeline"

    def get(self, component_type: str) -> ModelComponent | None:
        for c in self.components:
            if c.component_type == component_type:
                return c
        return None

    def get_all(self, component_type: str) -> list[ModelComponent]:
        return [c for c in self.components if c.component_type == component_type]


# Known class mappings: "library.ClassName" -> component_type
_CLASS_MAP: dict[str, str] = {
    # Transformers (diffusers)
    "diffusers.Transformer2DModel": "transformer",
    "diffusers.WanTransformer3DModel": "transformer",
    "diffusers.FluxTransformer2DModel": "transformer",
    "diffusers.SD3Transformer2DModel": "transformer",
    "diffusers.DiTTransformer2DModel": "transformer",
    "diffusers.HunyuanVideoTransformer3DModel": "transformer",
    "diffusers.CogVideoXTransformer3DModel": "transformer",
    "diffusers.PixArtTransformer2DModel": "transformer",
    "diffusers.LTXVideoTransformer3DModel": "transformer",
    "diffusers.MochiTransformer3DModel": "transformer",
    # UNets (diffusers)
    "diffusers.UNet2DConditionModel": "unet",
    "diffusers.UNet2DModel": "unet",
    "diffusers.UNet3DConditionModel": "unet",
    # VAEs (diffusers)
    "diffusers.AutoencoderKL": "vae",
    "diffusers.AutoencoderKLWan": "vae",
    "diffusers.AutoencoderDC": "vae",
    "diffusers.AutoencoderTiny": "vae",
    "diffusers.VQModel": "vae",
    # Text encoders (transformers)
    "transformers.CLIPTextModel": "text_encoder",
    "transformers.CLIPTextModelWithProjection": "text_encoder",
    "transformers.T5EncoderModel": "text_encoder",
    "transformers.UMT5EncoderModel": "text_encoder",
    "transformers.ChatGLMModel": "text_encoder",
    # Tokenizers (transformers)
    "transformers.CLIPTokenizer": "tokenizer",
    "transformers.T5Tokenizer": "tokenizer",
    "transformers.T5TokenizerFast": "tokenizer",
    "transformers.AutoTokenizer": "tokenizer",
    # Schedulers (diffusers)
    "diffusers.FlowMatchEulerDiscreteScheduler": "scheduler",
    "diffusers.EulerDiscreteScheduler": "scheduler",
    "diffusers.EulerAncestralDiscreteScheduler": "scheduler",
    "diffusers.DPMSolverMultistepScheduler": "scheduler",
    "diffusers.PNDMScheduler": "scheduler",
    "diffusers.DDIMScheduler": "scheduler",
    "diffusers.UniPCMultistepScheduler": "scheduler",
    "diffusers.FlowMatchHeunDiscreteScheduler": "scheduler",
    # Image encoders
    "transformers.CLIPVisionModelWithProjection": "other",
    "transformers.SiglipVisionModel": "other",
}


def _classify_component(name: str, class_name: str | None = None, library: str | None = None) -> str:
    """Classify a component by its library.class_name, falling back to directory name."""
    if library and class_name:
        key = f"{library}.{class_name}"
        if key in _CLASS_MAP:
            return _CLASS_MAP[key]
        log.warning("Unknown model class: %s — classifying as 'other'", key)
    return "other"


def _has_safetensors(path: Path) -> bool:
    return any(path.glob("*.safetensors"))


def detect_components(model_dir: Path) -> ModelInfo:
    """Detect components from a diffusion model directory.

    Requires model_index.json (standard HuggingFace diffusers layout).
    """
    index_path = model_dir / "model_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No model_index.json found in {model_dir}. "
            "The model directory must follow the HuggingFace diffusers layout."
        )

    with open(index_path) as f:
        index = json.load(f)

    model_type = index.get("_class_name")
    components: list[ModelComponent] = []

    for key, value in index.items():
        if key.startswith("_"):
            continue
        # value is [module_path, class_name] or None
        if value is None:
            continue
        comp_dir = model_dir / key
        if comp_dir.is_dir() and _has_safetensors(comp_dir):
            # value is [library, class_name]
            library = value[0] if isinstance(value, list) and len(value) >= 1 else None
            class_name = value[1] if isinstance(value, list) and len(value) >= 2 else None
            comp_type = _classify_component(key, class_name, library)
            components.append(ModelComponent(
                name=key,
                path=comp_dir,
                component_type=comp_type,
                library=library,
                class_name=class_name,
            ))
            log.debug("Found component: %s (%s) at %s", key, comp_type, comp_dir)

    log.info(
        "Detected %d quantizable components in %s pipeline",
        len(components),
        model_type or "unknown",
    )
    return ModelInfo(root=model_dir, components=components, model_type=model_type)
