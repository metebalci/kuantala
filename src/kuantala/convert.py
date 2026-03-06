"""Convert modelopt NVFP4 safetensors to ComfyUI format."""

from __future__ import annotations

import json
import math
import re
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


# ---------------------------------------------------------------------------
# Diffusers → original key mappings (for models where ComfyUI expects
# the original author's naming, not diffusers naming)
# ---------------------------------------------------------------------------

# Wan 2.1 / 2.2: diffusers (WanTransformer3DModel) → original (WanModel)
_WAN_KEY_MAP: list[tuple[str, str]] = [
    # Attention: attn1 (self-attention) → self_attn
    (r"blocks\.(\d+)\.attn1\.to_q\b", r"blocks.\1.self_attn.q"),
    (r"blocks\.(\d+)\.attn1\.to_k\b", r"blocks.\1.self_attn.k"),
    (r"blocks\.(\d+)\.attn1\.to_v\b", r"blocks.\1.self_attn.v"),
    (r"blocks\.(\d+)\.attn1\.to_out\.0\b", r"blocks.\1.self_attn.o"),
    (r"blocks\.(\d+)\.attn1\.norm_q\b", r"blocks.\1.self_attn.norm_q"),
    (r"blocks\.(\d+)\.attn1\.norm_k\b", r"blocks.\1.self_attn.norm_k"),
    # Attention: attn2 (cross-attention) → cross_attn
    (r"blocks\.(\d+)\.attn2\.to_q\b", r"blocks.\1.cross_attn.q"),
    (r"blocks\.(\d+)\.attn2\.to_k\b", r"blocks.\1.cross_attn.k"),
    (r"blocks\.(\d+)\.attn2\.to_v\b", r"blocks.\1.cross_attn.v"),
    (r"blocks\.(\d+)\.attn2\.to_out\.0\b", r"blocks.\1.cross_attn.o"),
    (r"blocks\.(\d+)\.attn2\.norm_q\b", r"blocks.\1.cross_attn.norm_q"),
    (r"blocks\.(\d+)\.attn2\.norm_k\b", r"blocks.\1.cross_attn.norm_k"),
    # FFN: ffn.net.{0.proj,2} → ffn.{0,2}
    (r"blocks\.(\d+)\.ffn\.net\.0\.proj\b", r"blocks.\1.ffn.0"),
    (r"blocks\.(\d+)\.ffn\.net\.2\b", r"blocks.\1.ffn.2"),
    # Block norm and modulation
    (r"blocks\.(\d+)\.norm2\b", r"blocks.\1.norm3"),
    (r"blocks\.(\d+)\.scale_shift_table\b", r"blocks.\1.modulation"),
    # Condition embedder → separate embeddings
    (r"condition_embedder\.text_embedder\.linear_1\b", "text_embedding.0"),
    (r"condition_embedder\.text_embedder\.linear_2\b", "text_embedding.2"),
    (r"condition_embedder\.time_embedder\.linear_1\b", "time_embedding.0"),
    (r"condition_embedder\.time_embedder\.linear_2\b", "time_embedding.2"),
    (r"condition_embedder\.time_proj\b", "time_projection.1"),
    # Head
    (r"^proj_out\b", "head.head"),
    (r"^scale_shift_table$", "head.modulation"),
    (r"^norm_out\b", "head.norm"),
    # Image embedding (i2v models)
    (r"condition_embedder\.image_embedder\.proj\.0\b", "img_emb.proj.0"),
    (r"condition_embedder\.image_embedder\.proj\.2\b", "img_emb.proj.2"),
    (r"condition_embedder\.image_embedder\.norm\b", "img_emb.norm"),
]

_WAN_KEY_MAP_COMPILED = [(re.compile(p), r) for p, r in _WAN_KEY_MAP]


_KEY_MAPS: dict[str, list[tuple[re.Pattern, str]]] = {
    "wan": _WAN_KEY_MAP_COMPILED,
}


def _remap_key(key: str, key_map: list[tuple[re.Pattern, str]] | None) -> str:
    """Remap a single key from diffusers naming to original naming."""
    if key_map is not None:
        for pattern, replacement in key_map:
            new_key = pattern.sub(replacement, key)
            if new_key != key:
                return new_key
    return key


def convert_to_comfyui(input_path: Path, output_path: Path, remap_keys: str | None = None) -> Path:
    """Convert a modelopt NVFP4 safetensors file to ComfyUI format.

    Performs:
    1. Swap nibble order in packed FP4 weight bytes
    2. Convert block scales from plain to cuBLAS tiled layout
    3. Rename modelopt quantizer keys to ComfyUI conventions
    4. Remap diffusers layer names to original names (if remap_keys specified)
    5. Add per-layer .comfy_quant metadata tensors

    Args:
        remap_keys: Key remapping profile (e.g. "wan"). None to skip remapping.
    """
    state_dict = load_file(str(input_path))

    key_map = _KEY_MAPS.get(remap_keys) if remap_keys else None

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
        out_prefix = _remap_key(prefix, key_map)

        # Packed FP4 weight: swap nibbles
        weight_key = f"{prefix}.weight"
        if weight_key in state_dict:
            b = state_dict[weight_key]
            output[f"{out_prefix}.weight"] = ((b & 0xF) << 4) | (b >> 4)
            processed_keys.add(weight_key)

        # Block scale: apply tiled layout, rename
        scale_key = f"{prefix}.weight_quantizer._scale"
        if scale_key in state_dict:
            scale = state_dict[scale_key]
            output[f"{out_prefix}.weight_scale"] = _to_blocked(scale)
            processed_keys.add(scale_key)

        # Tensor scale (double scale): rename
        ds_key = f"{prefix}.weight_quantizer._double_scale"
        if ds_key in state_dict:
            output[f"{out_prefix}.weight_scale_2"] = state_dict[ds_key]
            processed_keys.add(ds_key)

        # Input scale: rename
        input_key = f"{prefix}.input_quantizer._amax"
        if input_key in state_dict:
            output[f"{out_prefix}.input_scale"] = state_dict[input_key]
            processed_keys.add(input_key)

        # Weight amax: drop (not needed by ComfyUI)
        amax_key = f"{prefix}.weight_quantizer._amax"
        if amax_key in state_dict:
            processed_keys.add(amax_key)

        # Add comfy_quant metadata
        output[f"{out_prefix}.comfy_quant"] = comfy_quant_value.clone()

    # Copy non-quantized tensors (with key remapping)
    for key, tensor in state_dict.items():
        if key not in processed_keys:
            out_key = _remap_key(key, key_map)
            output[out_key] = tensor

    # Drop stray quantizer keys on non-quantized layers (input_quantizer._amax, weight_quantizer._amax)
    output = {k: v for k, v in output.items()
              if not k.endswith((".input_quantizer._amax", ".weight_quantizer._amax"))}

    save_file(output, str(output_path))
    return output_path
