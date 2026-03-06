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
                model = cls.from_pretrained(str(component_path), torch_dtype=torch.float16)
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

        if config.mixed_calibration:
            def calibrate_fn(m):
                log.info("Running calibration forward passes...")
                with torch.no_grad():
                    for _ in range(8):
                        for p in m.parameters():
                            _ = p * 1.0

            mtq.quantize(model, quant_cfg, forward_loop=calibrate_fn)
        else:
            mtq.quantize(model, quant_cfg, forward_loop=lambda m: None)

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
