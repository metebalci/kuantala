"""Main quantization orchestrator."""

from __future__ import annotations

import gc
import importlib
from pathlib import Path
from typing import Any

import torch
import modelopt.torch.quantization as mtq
from safetensors.torch import save_file

from kuantala.components import ModelComponent, detect_components
from kuantala.config import QuantConfig, is_passthrough_dtype
from kuantala.model_loader import resolve_model_path
from kuantala.utils import get_logger

log = get_logger(__name__)

# modelopt quantization configs for each dtype
_MODELOPT_CONFIGS = {
    "FP8": mtq.FP8_DEFAULT_CFG,
    "NVFP4": mtq.NVFP4_DEFAULT_CFG,
}

_TORCH_DTYPES = {
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
}


def _resolve_component_dtype(
    component: ModelComponent, config: QuantConfig
) -> str | None:
    """Determine what dtype to use for a component. Returns None if skip."""
    if component.component_type == "vae":
        dtype = config.vae_dtype
    elif component.component_type == "text_encoder":
        dtype = config.te_dtype
    elif component.component_type == "image_encoder":
        dtype = config.ie_dtype
    else:
        dtype = config.dtype

    if dtype is None:
        dtype = config.dtype

    if dtype == "skip":
        return None

    return dtype


def _load_model(component: ModelComponent) -> torch.nn.Module:
    """Load a model component via diffusers/transformers from_pretrained."""
    if not component.library or not component.class_name:
        raise RuntimeError(
            f"Cannot load component '{component.name}': missing library/class info"
        )

    lib = importlib.import_module(component.library)
    cls = getattr(lib, component.class_name)

    # diffusers uses torch_dtype, transformers uses dtype
    dtype_kwarg = "dtype" if component.library == "transformers" else "torch_dtype"
    model = cls.from_pretrained(str(component.path), **{dtype_kwarg: torch.float16})
    model = model.cuda()
    log.info("Loaded %s.%s on CUDA", component.library, component.class_name)
    return model


def _make_calibration_fn(model: Any, component: ModelComponent, num_batches: int = 4):
    """Create a calibration forward_loop that feeds random data through the model."""
    import json

    cfg = {}
    cfg_file = component.path / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = json.load(f)

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    def forward_loop(m):
        log.info("Running calibration with random data (%d batches)...", num_batches)
        with torch.no_grad(), torch.autocast("cuda", dtype=model_dtype):
            for _ in range(num_batches):
                _run_random_forward(m, component.component_type, cfg, device, model_dtype)

    return forward_loop


def _run_random_forward(model: Any, component_type: str, cfg: dict, device: Any, model_dtype: Any) -> None:
    """Run a single forward pass with random inputs appropriate for the component type."""
    if component_type == "text_encoder":
        vocab_size = cfg.get("vocab_size", 32128)
        max_length = cfg.get("max_position_embeddings", 77)
        seq_len = min(max_length, 64)
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
        model(input_ids=input_ids)

    elif component_type in ("transformer", "unet"):
        in_channels = cfg.get("in_channels", 4)
        class_name = cfg.get("_class_name", "")

        if "3D" in class_name or "Video" in class_name or "Wan" in class_name:
            hidden_states = torch.randn(1, in_channels, 8, 16, 16, device=device, dtype=model_dtype)
        else:
            sample_size = cfg.get("sample_size", 64)
            if isinstance(sample_size, list):
                sample_size = sample_size[0]
            s = min(sample_size, 32)
            hidden_states = torch.randn(1, in_channels, s, s, device=device, dtype=model_dtype)

        timestep = torch.randint(0, 1000, (1,), device=device)
        text_dim = cfg.get("cross_attention_dim") or cfg.get("text_dim") or cfg.get("encoder_hid_dim", 768)
        encoder_hidden_states = torch.randn(1, 16, text_dim, device=device, dtype=model_dtype)

        try:
            model(hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
        except TypeError:
            try:
                model(hidden_states, timestep, encoder_hidden_states)
            except Exception:
                log.warning("Forward pass failed for %s, falling back to parameter activation", component_type)
                for p in model.parameters():
                    _ = p * 1.0

    elif component_type == "vae":
        in_channels = cfg.get("in_channels", 3)
        sample_size = cfg.get("sample_size", 256)
        if isinstance(sample_size, list):
            sample_size = sample_size[0]
        s = min(sample_size, 64)

        class_name = cfg.get("_class_name", "")
        if "Wan" in class_name or "Video" in class_name:
            x = torch.randn(1, in_channels, 4, s, s, device=device, dtype=model_dtype)
        else:
            x = torch.randn(1, in_channels, s, s, device=device, dtype=model_dtype)

        try:
            model.encode(x)
        except Exception:
            try:
                model(x)
            except Exception:
                log.warning("Forward pass failed for %s, falling back to parameter activation", component_type)
                for p in model.parameters():
                    _ = p * 1.0

    elif component_type == "image_encoder":
        image_size = cfg.get("image_size", 224)
        num_channels = cfg.get("num_channels", 3)
        pixel_values = torch.randn(1, num_channels, image_size, image_size, device=device, dtype=model_dtype)

        try:
            model(pixel_values=pixel_values)
        except TypeError:
            try:
                model(pixel_values)
            except Exception:
                log.warning("Forward pass failed for %s, falling back to parameter activation", component_type)
                for p in model.parameters():
                    _ = p * 1.0

    else:
        for p in model.parameters():
            _ = p * 1.0


def _disable_kv_cache_plugins():
    """Disable modelopt KV cache quantization plugins.

    Diffusion model components don't use KV cache, and the auto-patching
    fails on some attention classes (e.g. UMT5Attention).
    """
    try:
        from modelopt.torch.quantization.plugins import custom as _custom_plugins
        saved = set(_custom_plugins.CUSTOM_MODEL_PLUGINS)
        _custom_plugins.CUSTOM_MODEL_PLUGINS.clear()
        return saved
    except Exception:
        return None


def _restore_kv_cache_plugins(saved):
    """Restore previously saved KV cache plugins."""
    if saved is not None:
        try:
            from modelopt.torch.quantization.plugins import custom as _custom_plugins
            _custom_plugins.CUSTOM_MODEL_PLUGINS.update(saved)
        except Exception:
            pass


def _disable_quantizers_by_pattern(model: torch.nn.Module, patterns: list[str]) -> None:
    """Disable quantizers on layers matching any of the given glob patterns."""
    import fnmatch

    if not patterns:
        return

    def filter_fn(name: str) -> bool:
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    mtq.disable_quantizer(model, filter_fn)
    log.info("Disabled quantizers on layers matching: %s", patterns)


def _build_metadata(component: ModelComponent, dtype: str, config: QuantConfig) -> dict[str, str]:
    """Build safetensors metadata recording how the file was produced."""
    import json
    meta: dict[str, str] = {
        "quantizer": "kuantala",
        "dtype": dtype,
        "model_source": config.model_source,
        "component": component.name,
    }
    if component.class_name:
        meta["class_name"] = component.class_name
    if component.library:
        meta["library"] = component.library
    if config.keep:
        meta["keep"] = json.dumps(config.keep)
    return meta


def _quantize_component(
    component: ModelComponent,
    dtype: str,
    config: QuantConfig,
    output_path: Path,
) -> Path:
    """Quantize a single component and save to safetensors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = output_path.with_suffix(".safetensors")

    model = _load_model(component)

    metadata = _build_metadata(component, dtype, config)

    # Passthrough: just cast and save
    if is_passthrough_dtype(dtype):
        target_dtype = _TORCH_DTYPES[dtype]
        model = model.to(target_dtype)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        save_file(state_dict, str(output_file), metadata=metadata)
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        log.info("Written %s (%.1f MB)", output_file, file_size_mb)
        return output_file

    # Quantize with modelopt
    quant_cfg = _MODELOPT_CONFIGS[dtype]

    def forward_loop(m): pass  # no-op when calibration disabled
    if config.calibration:
        forward_loop = _make_calibration_fn(model, component, config.calib_size)

    saved_plugins = _disable_kv_cache_plugins()
    try:
        log.info("Applying %s quantization via modelopt...", dtype)
        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

        # Disable quantizers on user-specified layers
        if config.keep:
            _disable_quantizers_by_pattern(model, config.keep)

        import warnings
        log.info("Compressing weights to real %s...", dtype)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Real quantization has been applied")
            mtq.compress(model)
    finally:
        _restore_kv_cache_plugins(saved_plugins)

    # Save compressed state dict
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, str(output_file), metadata=metadata)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    log.info("Written %s (%.1f MB)", output_file, file_size_mb)
    del model
    return output_file


def quantize(config: QuantConfig) -> list[Path]:
    """Run quantization according to the given config.

    Returns list of output file paths.
    """
    model_dir = resolve_model_path(config.model_source)
    model_info = detect_components(model_dir)
    if not model_info.components:
        raise RuntimeError(f"No quantizable components found in {model_dir}")

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

        output_path = config.output_dir / f"{component.name}-{dtype}"
        output_file = _quantize_component(component, dtype, config, output_path)
        output_files.append(output_file)
        # Free GPU memory for next component
        torch.cuda.empty_cache()
        gc.collect()

    if not output_files:
        log.warning("No components were quantized.")

    return output_files
