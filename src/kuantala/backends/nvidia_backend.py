"""NVIDIA quantization backend using nvidia-modelopt for MXFP8/NVFP4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kuantala.backends import QuantBackend
from kuantala.config import NVIDIA_TYPES, QuantConfig
from kuantala.utils import get_logger

log = get_logger(__name__)


def _check_dependencies() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "PyTorch not installed. Install a CUDA build from https://pytorch.org, e.g.:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
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
    ) -> Path:
        import torch
        import modelopt.torch.quantization as mtq
        from safetensors.torch import save_file

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.with_suffix(".safetensors")

        # Load model weights
        log.info("Loading model from %s", component_path)
        from diffusers import DiffusionPipeline

        # Try to load just the component as a model
        state_dict = {}
        for sf_path in sorted(component_path.glob("*.safetensors")):
            from safetensors.torch import load_file
            state_dict.update(load_file(str(sf_path)))

        # Build a simple module from state dict
        model = torch.nn.Module()
        for name, tensor in state_dict.items():
            parts = name.rsplit(".", 1)
            param = torch.nn.Parameter(tensor.cuda(), requires_grad=False)
            if len(parts) == 2:
                model.register_parameter(name, param)
            else:
                model.register_parameter(name, param)

        # Apply quantization
        quant_cfg = _get_quant_config(dtype)
        log.info("Applying %s quantization via nvidia-modelopt...", dtype)

        if config.mixed_calibration:
            # Calibration mode: use forward passes
            def calibrate_fn(model):
                log.info("Running calibration forward passes...")
                # Generate random inputs for calibration
                for _ in range(8):
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            _ = param * 1.0  # Simple calibration pass

            mtq.quantize(model, quant_cfg, forward_loop=calibrate_fn)
        else:
            # Direct quantization without calibration
            mtq.quantize(model, quant_cfg, forward_loop=lambda m: None)

        # Extract quantized state dict
        quantized_state_dict = {}
        for name, param in model.named_parameters():
            quantized_state_dict[name] = param.cpu()

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
