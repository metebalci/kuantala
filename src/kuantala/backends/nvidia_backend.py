"""NVIDIA quantization backend using nvidia-modelopt for MXFP8/NVFP4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kuantala.backends import QuantBackend
from kuantala.config import NVIDIA_TYPES, QuantConfig
from kuantala.utils import get_logger

log = get_logger(__name__)


def _make_random_calibration_fn(model: Any, component_type: str, config_path: Path, num_batches: int = 4):
    """Create a calibration forward_loop that feeds random data through the model."""
    import torch

    cfg = {}
    cfg_file = config_path / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = json.load(f)

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    def forward_loop(m):
        log.info("Running calibration with random data (%d batches)...", num_batches)
        with torch.no_grad():
            for _ in range(num_batches):
                _run_random_forward(m, component_type, cfg, device, model_dtype)

    return forward_loop


def _run_random_forward(model: Any, component_type: str, cfg: dict, device: Any, model_dtype: Any) -> None:
    """Run a single forward pass with random inputs appropriate for the component type."""
    import torch

    if component_type == "text_encoder":
        _forward_text_encoder(model, cfg, device)
    elif component_type in ("transformer", "unet"):
        _forward_transformer_or_unet(model, cfg, device, model_dtype)
    elif component_type == "vae":
        _forward_vae(model, cfg, device, model_dtype)
    elif component_type == "image_encoder":
        _forward_image_encoder(model, cfg, device, model_dtype)
    else:
        # Generic fallback: just touch all parameters
        for p in model.parameters():
            _ = p * 1.0


def _forward_text_encoder(model: Any, cfg: dict, device: Any) -> None:
    """Forward pass for CLIP/T5/UMT5 text encoders."""
    import torch

    # Text encoders take input_ids (integer token indices)
    vocab_size = cfg.get("vocab_size", 32128)
    max_length = cfg.get("max_position_embeddings", 77)
    seq_len = min(max_length, 64)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    model(input_ids=input_ids)


def _forward_transformer_or_unet(model: Any, cfg: dict, device: Any, model_dtype: Any) -> None:
    """Forward pass for diffusion transformers and UNets."""
    import torch

    in_channels = cfg.get("in_channels", 4)
    class_name = cfg.get("_class_name", "")

    if "3D" in class_name or "Video" in class_name or "Wan" in class_name:
        # Video model: (batch, channels, frames, height, width)
        patch_size = cfg.get("patch_size", [1, 2, 2])
        # Use small spatial dims for calibration
        f, h, w = 8, 16, 16
        hidden_states = torch.randn(1, in_channels, f, h, w, device=device, dtype=model_dtype)
    else:
        # 2D model: (batch, channels, height, width)
        sample_size = cfg.get("sample_size", 64)
        if isinstance(sample_size, list):
            sample_size = sample_size[0]
        # Use smaller size for calibration
        s = min(sample_size, 32)
        hidden_states = torch.randn(1, in_channels, s, s, device=device, dtype=model_dtype)

    timestep = torch.randint(0, 1000, (1,), device=device)

    # Encoder hidden states (text conditioning)
    text_dim = cfg.get("cross_attention_dim") or cfg.get("text_dim") or cfg.get("encoder_hid_dim", 768)
    encoder_hidden_states = torch.randn(1, 16, text_dim, device=device, dtype=model_dtype)

    try:
        model(hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
    except TypeError:
        # Some models use positional args
        try:
            model(hidden_states, timestep, encoder_hidden_states)
        except Exception:
            # Last resort
            for p in model.parameters():
                _ = p * 1.0


def _forward_vae(model: Any, cfg: dict, device: Any, model_dtype: Any) -> None:
    """Forward pass for VAE (encode path)."""
    import torch

    in_channels = cfg.get("in_channels", 3)
    sample_size = cfg.get("sample_size", 256)
    if isinstance(sample_size, list):
        sample_size = sample_size[0]
    s = min(sample_size, 64)

    class_name = cfg.get("_class_name", "")
    if "Wan" in class_name or "Video" in class_name:
        # Video VAE: (batch, channels, frames, height, width)
        x = torch.randn(1, in_channels, 4, s, s, device=device, dtype=model_dtype)
    else:
        x = torch.randn(1, in_channels, s, s, device=device, dtype=model_dtype)

    try:
        model.encode(x)
    except Exception:
        try:
            model(x)
        except Exception:
            for p in model.parameters():
                _ = p * 1.0


def _forward_image_encoder(model: Any, cfg: dict, device: Any, model_dtype: Any) -> None:
    """Forward pass for CLIP/SigLIP image encoders."""
    import torch

    image_size = cfg.get("image_size", 224)
    num_channels = cfg.get("num_channels", 3)
    pixel_values = torch.randn(1, num_channels, image_size, image_size, device=device, dtype=model_dtype)

    try:
        model(pixel_values=pixel_values)
    except TypeError:
        try:
            model(pixel_values)
        except Exception:
            for p in model.parameters():
                _ = p * 1.0


def _check_dependencies() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyTorch not installed. Install a CUDA build from https://pytorch.org, e.g.:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu130"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. The NVIDIA backend requires a CUDA-capable GPU "
            "and a CUDA build of PyTorch."
        )
    try:
        import diffusers  # noqa: F401
    except ImportError:
        raise ImportError(
            "diffusers not installed. Install with: pip install kuantala[nvidia]"
        )
    try:
        import modelopt  # noqa: F401
    except ImportError:
        raise ImportError(
            "nvidia-modelopt not installed. Install with: pip install kuantala[nvidia]"
        )


def _get_quant_config(dtype: str) -> dict:
    """Get nvidia-modelopt quantization config for the given dtype."""
    import modelopt.torch.quantization as mtq

    if dtype == "MXFP8":
        return mtq.FP8_DEFAULT_CFG
    elif dtype == "NVFP4":
        return mtq.NVFP4_DEFAULT_CFG
    else:
        raise ValueError(f"Unsupported NVIDIA dtype: {dtype}")


class NvidiaBackend(QuantBackend):
    """Quantize models using nvidia-modelopt for MXFP8/NVFP4."""

    def __init__(self) -> None:
        _check_dependencies()

    def quantize_component(
        self,
        component_path: Path,
        output_path: Path,
        dtype: str,
        config: QuantConfig,
        layer_overrides: dict[str, str] | None = None,
        component: Any = None,
    ) -> Path:
        import importlib

        import torch
        import modelopt.torch.quantization as mtq
        from safetensors.torch import save_file

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.with_suffix(".safetensors")

        # Load model as its actual class (Linear/Conv layers needed for modelopt)
        log.info("Loading model from %s", component_path)
        model = None
        if component and component.library and component.class_name:
            try:
                lib = importlib.import_module(component.library)
                cls = getattr(lib, component.class_name)
                # diffusers uses torch_dtype, transformers uses dtype
                dtype_kwarg = "dtype" if component.library == "transformers" else "torch_dtype"
                model = cls.from_pretrained(str(component_path), **{dtype_kwarg: torch.float16})
                model = model.cuda()
                log.info("Loaded %s.%s", component.library, component.class_name)
            except Exception as e:
                log.warning(
                    "Could not load %s.%s: %s. Falling back to state_dict.",
                    component.library, component.class_name, e,
                )
                model = None

        if model is None:
            # Fallback: build module hierarchy from state_dict
            log.info("Loading as raw state_dict (modelopt may not find quantizable layers)")
            state_dict = {}
            for sf_path in sorted(component_path.glob("*.safetensors")):
                from safetensors.torch import load_file
                state_dict.update(load_file(str(sf_path)))

            model = torch.nn.Module()
            for name, tensor in state_dict.items():
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    if not hasattr(parent, part):
                        parent.add_module(part, torch.nn.Module())
                    parent = getattr(parent, part)
                param = torch.nn.Parameter(tensor.cuda(), requires_grad=False)
                parent.register_parameter(parts[-1], param)

        # Apply quantization
        quant_cfg = _get_quant_config(dtype)
        log.info("Applying %s quantization via nvidia-modelopt...", dtype)

        forward_loop = lambda m: None
        if config.calibration:
            comp_type = component.component_type if component else "other"
            forward_loop = _make_random_calibration_fn(model, comp_type, component_path)

        # Disable HF attention KV cache quantization plugin — diffusion model
        # components don't use KV cache, and the auto-patching fails on some
        # attention classes (e.g. UMT5Attention).
        try:
            from modelopt.torch.quantization.plugins import custom as _custom_plugins
            saved_plugins = set(_custom_plugins.CUSTOM_MODEL_PLUGINS)
            _custom_plugins.CUSTOM_MODEL_PLUGINS.clear()
        except Exception:
            saved_plugins = None

        try:
            mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
        finally:
            if saved_plugins is not None:
                _custom_plugins.CUSTOM_MODEL_PLUGINS.update(saved_plugins)

        # Extract quantized state dict
        quantized_state_dict = {
            name: param.cpu() for name, param in model.named_parameters()
        }

        save_file(quantized_state_dict, str(output_file))

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        log.info("Written %s (%.1f MB)", output_file, file_size_mb)
        return output_file

    def supported_dtypes(self) -> list[str]:
        return list(NVIDIA_TYPES)

    def get_info(self) -> dict[str, Any]:
        return {
            "name": "NVIDIA",
            "description": "NVIDIA native formats (MXFP8/NVFP4) via nvidia-modelopt",
            "dtypes": self.supported_dtypes(),
            "requires_torch": True,
        }
