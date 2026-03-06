"""Main quantization orchestrator."""

from __future__ import annotations

from pathlib import Path

from kuantala.backends import get_backend
from kuantala.components import ModelComponent, detect_components
from kuantala.config import QuantConfig
from kuantala.mixed import compute_layer_overrides
from kuantala.model_loader import resolve_model_path
from kuantala.utils import get_logger

log = get_logger(__name__)


def _resolve_component_dtype(
    component: ModelComponent, config: QuantConfig
) -> str | None:
    """Determine what dtype to use for a component. Returns None if skip."""
    if component.component_type == "vae":
        dtype = config.vae_dtype
    elif component.component_type == "text_encoder":
        dtype = config.te_dtype
    else:
        dtype = config.dtype

    if dtype is None:
        dtype = config.dtype

    if dtype == "skip":
        return None

    return dtype


def quantize(config: QuantConfig) -> list[Path]:
    """Run quantization according to the given config.

    Returns list of output file paths.
    """
    # Resolve model to local path
    model_dir = resolve_model_path(config.model_source, config.hf_token)

    # Detect components
    model_info = detect_components(model_dir)
    if not model_info.components:
        raise RuntimeError(f"No quantizable components found in {model_dir}")

    # Get backend
    backend = get_backend(config.backend_name)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []

    for component in model_info.components:
        dtype = _resolve_component_dtype(component, config)
        if dtype is None:
            log.info("Skipping %s (dtype=skip)", component.name)
            continue

        if component.component_type in ("tokenizer", "scheduler", "other"):
            log.debug("Skipping non-quantizable component: %s", component.name)
            continue

        log.info("Processing component: %s (%s -> %s)", component.name, component.component_type, dtype)

        # Compute mixed quantization overrides
        safetensor_files = sorted(component.path.glob("*.safetensors"))
        layer_overrides = compute_layer_overrides(config, safetensor_files)

        output_path = config.output_dir / f"{component.name}-{dtype}"
        output_file = backend.quantize_component(
            component_path=component.path,
            output_path=output_path,
            dtype=dtype,
            config=config,
            layer_overrides=layer_overrides if layer_overrides else None,
        )
        output_files.append(output_file)

    if not output_files:
        log.warning("No components were quantized.")

    return output_files
