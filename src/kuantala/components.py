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


# Classification of component names to types
_TYPE_MAP = {
    "transformer": "transformer",
    "unet": "unet",
    "vae": "vae",
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder",
    "text_encoder_3": "text_encoder",
    "tokenizer": "tokenizer",
    "tokenizer_2": "tokenizer",
    "tokenizer_3": "tokenizer",
    "scheduler": "scheduler",
    "image_encoder": "other",
    "feature_extractor": "other",
    "safety_checker": "other",
}


def _classify_component(name: str) -> str:
    return _TYPE_MAP.get(name, "other")


def _has_safetensors(path: Path) -> bool:
    return any(path.glob("*.safetensors"))


def detect_components(model_dir: Path) -> ModelInfo:
    """Detect components from a diffusion model directory.

    Looks for model_index.json first (standard diffusers layout),
    then falls back to directory scanning.
    """
    index_path = model_dir / "model_index.json"
    if index_path.exists():
        return _parse_model_index(model_dir, index_path)
    return _scan_directory(model_dir)


def _parse_model_index(model_dir: Path, index_path: Path) -> ModelInfo:
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
            comp_type = _classify_component(key)
            components.append(ModelComponent(
                name=key,
                path=comp_dir,
                component_type=comp_type,
            ))
            log.debug("Found component: %s (%s) at %s", key, comp_type, comp_dir)

    log.info(
        "Detected %d quantizable components in %s pipeline",
        len(components),
        model_type or "unknown",
    )
    return ModelInfo(root=model_dir, components=components, model_type=model_type)


def _scan_directory(model_dir: Path) -> ModelInfo:
    """Fallback: scan subdirectories for safetensors files."""
    components: list[ModelComponent] = []

    # Check if safetensors are in root (single-component model)
    if _has_safetensors(model_dir):
        components.append(ModelComponent(
            name="model",
            path=model_dir,
            component_type="transformer",
        ))
        log.info("Single-component model detected at %s", model_dir)
        return ModelInfo(root=model_dir, components=components)

    for subdir in sorted(model_dir.iterdir()):
        if subdir.is_dir() and _has_safetensors(subdir):
            comp_type = _classify_component(subdir.name)
            components.append(ModelComponent(
                name=subdir.name,
                path=subdir,
                component_type=comp_type,
            ))

    log.info("Directory scan found %d components", len(components))
    return ModelInfo(root=model_dir, components=components)
