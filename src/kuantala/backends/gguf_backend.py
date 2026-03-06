"""GGUF quantization backend using the gguf library and numpy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kuantala.backends import QuantBackend
from kuantala.config import GGUF_TYPES, QuantConfig
from kuantala.utils import get_logger, make_progress

log = get_logger(__name__)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import gguf
except ImportError:
    gguf = None  # type: ignore[assignment]

# Mapping from our dtype names to GGUF library types
_GGUF_TYPE_MAP: dict[str, int] = {}


def _init_type_map() -> None:
    """Initialize the GGUF type map lazily."""
    if _GGUF_TYPE_MAP:
        return
    if gguf is None:
        raise ImportError(
            "gguf package not installed. Install with: pip install kuantala[gguf]"
        )
    type_enum = gguf.GGMLQuantizationType
    for name in GGUF_TYPES:
        try:
            _GGUF_TYPE_MAP[name] = type_enum[name]
        except KeyError:
            pass
    # Also map passthrough types
    _GGUF_TYPE_MAP["F16"] = type_enum["F16"]
    _GGUF_TYPE_MAP["F32"] = type_enum["F32"]
    _GGUF_TYPE_MAP["BF16"] = type_enum["BF16"]


def _quantize_tensor_q8_0(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Quantize a tensor to Q8_0 format.

    Q8_0: block size 32, each block has 1 fp16 scale + 32 int8 values.
    """
    data = data.astype(np.float32).flatten()
    block_size = 32

    # Pad to multiple of block_size
    remainder = len(data) % block_size
    if remainder:
        data = np.pad(data, (0, block_size - remainder))

    blocks = data.reshape(-1, block_size)
    scales = np.abs(blocks).max(axis=1) / 127.0
    scales = np.where(scales == 0, 1.0, scales)

    quantized = np.round(blocks / scales[:, None]).astype(np.int8)
    scales_f16 = scales.astype(np.float16)

    # Pack: for each block, 2 bytes scale + 32 bytes quantized
    result = bytearray()
    for i in range(len(scales_f16)):
        result.extend(scales_f16[i].tobytes())
        result.extend(quantized[i].tobytes())

    return np.frombuffer(bytes(result), dtype=np.uint8), gguf.GGMLQuantizationType.Q8_0


def _quantize_tensor_q4_0(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Quantize a tensor to Q4_0 format.

    Q4_0: block size 32, each block has 1 fp16 scale + 16 bytes (32 x 4-bit values).
    """
    data = data.astype(np.float32).flatten()
    block_size = 32

    remainder = len(data) % block_size
    if remainder:
        data = np.pad(data, (0, block_size - remainder))

    blocks = data.reshape(-1, block_size)
    scales = np.abs(blocks).max(axis=1) / 7.0
    scales = np.where(scales == 0, 1.0, scales)

    quantized = np.round(blocks / scales[:, None]).astype(np.int8)
    quantized = np.clip(quantized, -8, 7)

    # Shift to unsigned: 0..15
    quantized_u = (quantized + 8).astype(np.uint8)

    result = bytearray()
    for i in range(len(scales)):
        scale_f16 = np.float16(scales[i])
        result.extend(scale_f16.tobytes())
        # Pack pairs of 4-bit values into bytes
        row = quantized_u[i]
        for j in range(0, block_size, 2):
            byte = (row[j] & 0x0F) | ((row[j + 1] & 0x0F) << 4)
            result.append(byte)

    return np.frombuffer(bytes(result), dtype=np.uint8), gguf.GGMLQuantizationType.Q4_0


def _quantize_tensor_q5_0(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Quantize a tensor to Q5_0 format.

    Q5_0: block size 32, each block has 1 fp16 scale + 4 bytes high bits + 16 bytes low nibbles.
    """
    data = data.astype(np.float32).flatten()
    block_size = 32

    remainder = len(data) % block_size
    if remainder:
        data = np.pad(data, (0, block_size - remainder))

    blocks = data.reshape(-1, block_size)
    scales = np.abs(blocks).max(axis=1) / 15.0
    scales = np.where(scales == 0, 1.0, scales)

    quantized = np.round(blocks / scales[:, None]).astype(np.int8)
    quantized = np.clip(quantized, -16, 15)
    quantized_u = (quantized + 16).astype(np.uint8)

    result = bytearray()
    for i in range(len(scales)):
        scale_f16 = np.float16(scales[i])
        result.extend(scale_f16.tobytes())
        row = quantized_u[i]
        # High bits (bit 4 of each value) packed into 4 bytes
        high_bits = bytearray(4)
        for j in range(block_size):
            if row[j] & 0x10:
                high_bits[j // 8] |= 1 << (j % 8)
        result.extend(high_bits)
        # Low nibbles packed in pairs
        for j in range(0, block_size, 2):
            byte = (row[j] & 0x0F) | ((row[j + 1] & 0x0F) << 4)
            result.append(byte)

    return np.frombuffer(bytes(result), dtype=np.uint8), gguf.GGMLQuantizationType.Q5_0


def _quantize_tensor_q6_k(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Quantize to Q6_K using gguf library's built-in quantization if available,
    otherwise fall back to Q8_0."""
    if hasattr(gguf, 'quants'):
        try:
            data_f32 = data.astype(np.float32).flatten()
            block_size = 256
            remainder = len(data_f32) % block_size
            if remainder:
                data_f32 = np.pad(data_f32, (0, block_size - remainder))
            quantized = gguf.quants.quantize(data_f32, gguf.GGMLQuantizationType.Q6_K)
            return quantized, gguf.GGMLQuantizationType.Q6_K
        except Exception:
            pass
    # Fallback to Q8_0
    log.debug("Q6_K not available via gguf.quants, falling back to Q8_0")
    return _quantize_tensor_q8_0(data)


def _quantize_tensor_kquant(data: np.ndarray, dtype_name: str) -> tuple[np.ndarray, int]:
    """Quantize using gguf library's built-in K-quant support."""
    target_type = _GGUF_TYPE_MAP[dtype_name]
    if hasattr(gguf, 'quants'):
        try:
            data_f32 = data.astype(np.float32).flatten()
            # K-quants use block size 256
            block_size = 256
            remainder = len(data_f32) % block_size
            if remainder:
                data_f32 = np.pad(data_f32, (0, block_size - remainder))
            quantized = gguf.quants.quantize(data_f32, target_type)
            return quantized, target_type
        except Exception as e:
            log.debug("gguf.quants failed for %s: %s, falling back", dtype_name, e)

    # Fallback chain for K-quants
    if "Q2_K" in dtype_name or "Q3_K" in dtype_name:
        return _quantize_tensor_q4_0(data)
    if "Q4_K" in dtype_name:
        return _quantize_tensor_q4_0(data)
    if "Q5_K" in dtype_name:
        return _quantize_tensor_q5_0(data)
    if "Q6_K" in dtype_name:
        return _quantize_tensor_q6_k(data)
    return _quantize_tensor_q8_0(data)


def quantize_tensor(data: np.ndarray, dtype: str) -> tuple[np.ndarray, int]:
    """Quantize a single tensor to the given dtype.

    Returns (quantized_data, ggml_type_enum_value).
    """
    _init_type_map()

    if dtype == "F16":
        return data.astype(np.float16).view(np.uint8), gguf.GGMLQuantizationType.F16
    if dtype == "F32":
        return data.astype(np.float32).view(np.uint8), gguf.GGMLQuantizationType.F32
    if dtype == "BF16":
        return data.astype(np.float32).view(np.uint8), gguf.GGMLQuantizationType.BF16
    if dtype == "Q8_0":
        return _quantize_tensor_q8_0(data)
    if dtype == "Q4_0":
        return _quantize_tensor_q4_0(data)
    if dtype == "Q5_0":
        return _quantize_tensor_q5_0(data)
    if dtype == "Q6_K":
        return _quantize_tensor_q6_k(data)

    # K-quants
    if dtype in _GGUF_TYPE_MAP:
        return _quantize_tensor_kquant(data, dtype)

    raise ValueError(f"Unsupported GGUF dtype: {dtype}")


class GGUFBackend(QuantBackend):
    """Quantize model components to GGUF format."""

    def __init__(self) -> None:
        if np is None:
            raise ImportError(
                "numpy not installed. Install with: pip install kuantala[gguf]"
            )
        if gguf is None:
            raise ImportError(
                "gguf package not installed. Install with: pip install kuantala[gguf]"
            )
        _init_type_map()

    def quantize_component(
        self,
        component_path: Path,
        output_path: Path,
        dtype: str,
        config: QuantConfig,
        layer_overrides: dict[str, str] | None = None,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.with_suffix(".gguf")

        safetensor_files = sorted(component_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(
                f"No safetensors files found in {component_path}"
            )

        # Collect all tensor names and shapes first
        tensor_info: list[tuple[str, Path]] = []
        for sf_path in safetensor_files:
            with safe_open(str(sf_path), framework="numpy") as f:
                for name in f.keys():
                    tensor_info.append((name, sf_path))

        log.info(
            "Quantizing %d tensors to %s -> %s",
            len(tensor_info), dtype, output_file,
        )

        writer = gguf.GGUFWriter(str(output_file), arch="diffusion")

        progress = make_progress()
        with progress:
            task = progress.add_task(
                f"Quantizing to {dtype}", total=len(tensor_info)
            )

            for tensor_name, sf_path in tensor_info:
                with safe_open(str(sf_path), framework="numpy") as f:
                    data = f.get_tensor(tensor_name)

                # Determine effective dtype for this tensor
                effective_dtype = dtype
                if layer_overrides and tensor_name in layer_overrides:
                    effective_dtype = layer_overrides[tensor_name]
                # Also check glob patterns in overrides
                elif layer_overrides:
                    import fnmatch
                    for pattern, override_dtype in layer_overrides.items():
                        if fnmatch.fnmatch(tensor_name, pattern):
                            effective_dtype = override_dtype
                            break

                # Skip 1D tensors (biases, norms) - keep at F16 unless explicitly overridden
                if data.ndim <= 1 and tensor_name not in (layer_overrides or {}):
                    effective_dtype = "F16"

                shape = data.shape
                quantized_data, qtype = quantize_tensor(data, effective_dtype)

                writer.add_tensor(
                    tensor_name,
                    quantized_data,
                    raw_shape=shape,
                    raw_dtype=qtype,
                )

                progress.advance(task)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        log.info("Written %s (%.1f MB)", output_file, file_size_mb)
        return output_file

    def supported_dtypes(self) -> list[str]:
        return list(GGUF_TYPES)

    def get_info(self) -> dict[str, Any]:
        return {
            "name": "GGUF",
            "description": "GGUF format for llama.cpp / stable-diffusion.cpp",
            "dtypes": self.supported_dtypes(),
            "requires_torch": False,
        }
