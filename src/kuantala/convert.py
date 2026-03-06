"""Convert modelopt NVFP4 safetensors to ComfyUI format."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def _ceil_div(a: int, b: int) -> int:
    return math.ceil(a / b)


def _to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """Rearrange block scales from plain row-major to cuBLAS tiled layout.

    Ported from comfy-kitchen/comfy_kitchen/float_utils.py:272-306.
    """
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    padded = F.pad(input_matrix, (0, padded_cols - cols, 0, padded_rows - rows))
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.reshape(padded_rows, padded_cols)


def convert_to_comfyui(input_path: Path, output_path: Path) -> Path:
    """Convert a modelopt NVFP4 safetensors file to ComfyUI format.

    Performs three data transformations plus metadata injection:
    1. Swap nibble order in packed FP4 weight bytes
    2. Convert block scales from plain to cuBLAS tiled layout
    3. Rename modelopt tensor keys to ComfyUI conventions
    4. Add per-layer .comfy_quant metadata tensors

    Returns the output path.
    """
    state_dict = load_file(str(input_path))

    # Find all quantized layer prefixes by looking for weight_quantizer._scale
    quantized_prefixes: set[str] = set()
    for key in state_dict:
        if key.endswith(".weight_quantizer._scale"):
            prefix = key.removesuffix(".weight_quantizer._scale")
            quantized_prefixes.add(prefix)

    comfy_quant_value = torch.tensor(
        list(json.dumps({"format": "nvfp4"}).encode("utf-8")),
        dtype=torch.uint8,
    )

    output: dict[str, torch.Tensor] = {}
    processed_keys: set[str] = set()

    for prefix in quantized_prefixes:
        # Packed FP4 weight: swap nibbles
        weight_key = f"{prefix}.weight"
        if weight_key in state_dict:
            b = state_dict[weight_key]
            output[weight_key] = ((b & 0xF) << 4) | (b >> 4)
            processed_keys.add(weight_key)

        # Block scale: apply tiled layout, rename
        scale_key = f"{prefix}.weight_quantizer._scale"
        if scale_key in state_dict:
            scale = state_dict[scale_key]
            output[f"{prefix}.weight_scale"] = _to_blocked(scale)
            processed_keys.add(scale_key)

        # Tensor scale (double scale): rename
        ds_key = f"{prefix}.weight_quantizer._double_scale"
        if ds_key in state_dict:
            output[f"{prefix}.weight_scale_2"] = state_dict[ds_key]
            processed_keys.add(ds_key)

        # Input scale: rename
        input_key = f"{prefix}.input_quantizer._amax"
        if input_key in state_dict:
            output[f"{prefix}.input_scale"] = state_dict[input_key]
            processed_keys.add(input_key)

        # Weight amax: drop (not needed by ComfyUI)
        amax_key = f"{prefix}.weight_quantizer._amax"
        if amax_key in state_dict:
            processed_keys.add(amax_key)

        # Add comfy_quant metadata
        output[f"{prefix}.comfy_quant"] = comfy_quant_value.clone()

    # Copy non-quantized tensors as-is
    for key, tensor in state_dict.items():
        if key not in processed_keys:
            output[key] = tensor

    save_file(output, str(output_path))
    return output_path
