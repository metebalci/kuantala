"""Mixed quantization: detect important layers and assign higher precision."""

from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np
from safetensors import safe_open

from kuantala.config import QuantConfig
from kuantala.utils import get_logger

log = get_logger(__name__)

# Layer name patterns known to be sensitive in diffusion models
_HEURISTIC_PATTERNS = {
    # Normalization layers (tiny, nearly free to keep at high precision)
    "*norm*": "F16",
    "*ln_*": "F16",
    "*layernorm*": "F16",
    "*groupnorm*": "F16",

    # Attention QKV projections
    "*attn*.to_q*": "F16",
    "*attn*.to_k*": "F16",
    "*attn*.to_v*": "F16",
    "*self_attn*.q_proj*": "F16",
    "*self_attn*.k_proj*": "F16",
    "*self_attn*.v_proj*": "F16",

    # Time/timestep embeddings (critical for diffusion scheduling)
    "*time_embed*": "F16",
    "*timestep*": "F16",
    "*t_embedder*": "F16",

    # First and last layers
    "*proj_in*": "F16",
    "*proj_out*": "F16",
    "*input_blocks.0.*": "F16",
    "*out.0.*": "F16",
}


def compute_heuristic_overrides(tensor_names: list[str]) -> dict[str, str]:
    """Return layer overrides based on known-sensitive layer patterns."""
    overrides: dict[str, str] = {}
    for name in tensor_names:
        for pattern, dtype in _HEURISTIC_PATTERNS.items():
            if fnmatch.fnmatch(name.lower(), pattern.lower()):
                overrides[name] = dtype
                break
    if overrides:
        log.info("Heuristics: preserving %d/%d layers at higher precision",
                 len(overrides), len(tensor_names))
    return overrides


def compute_statistics_overrides(
    safetensor_files: list[Path],
    percentage: int = 10,
) -> dict[str, str]:
    """Analyze weight statistics to find the most sensitive layers.

    Layers with high outlier ratios or large dynamic ranges are most harmed
    by quantization. The top `percentage`% are preserved at F16.
    """
    scores: dict[str, float] = {}

    for sf_path in safetensor_files:
        with safe_open(str(sf_path), framework="numpy") as f:
            for name in f.keys():
                data = f.get_tensor(name).astype(np.float32).flatten()
                if len(data) < 32:
                    continue

                std = np.std(data)
                if std == 0:
                    scores[name] = 0.0
                    continue

                mean = np.mean(data)
                abs_max = np.abs(data).max()

                # Outlier ratio: fraction of values > 3 standard deviations
                outlier_mask = np.abs(data - mean) > 3 * std
                outlier_ratio = outlier_mask.sum() / len(data)

                # Dynamic range relative to std
                dynamic_range = abs_max / std if std > 0 else 0

                # Combined sensitivity score
                scores[name] = outlier_ratio * 100 + dynamic_range * 0.1

    if not scores:
        return {}

    # Find threshold for top N%
    sorted_scores = sorted(scores.values(), reverse=True)
    cutoff_idx = max(1, int(len(sorted_scores) * percentage / 100))
    threshold = sorted_scores[min(cutoff_idx, len(sorted_scores) - 1)]

    overrides = {name: "F16" for name, score in scores.items() if score >= threshold}
    log.info(
        "Statistics: preserving %d/%d layers (top %d%%) at F16",
        len(overrides), len(scores), percentage,
    )
    return overrides


def parse_keep_rules(keep_specs: list[str]) -> dict[str, str]:
    """Parse --keep "pattern:dtype" specifications into a dict."""
    rules: dict[str, str] = {}
    for spec in keep_specs:
        if ":" not in spec:
            raise ValueError(
                f"Invalid --keep spec {spec!r}. Expected format: 'pattern:dtype'"
            )
        pattern, dtype = spec.rsplit(":", 1)
        rules[pattern] = dtype
    return rules


def expand_keep_rules(rules: dict[str, str], tensor_names: list[str]) -> dict[str, str]:
    """Expand glob patterns in keep rules against actual tensor names."""
    overrides: dict[str, str] = {}
    for name in tensor_names:
        for pattern, dtype in rules.items():
            if fnmatch.fnmatch(name, pattern):
                overrides[name] = dtype
                break
    if overrides:
        log.info("Manual rules: overriding %d layers", len(overrides))
    return overrides


def compute_layer_overrides(
    config: QuantConfig,
    safetensor_files: list[Path],
) -> dict[str, str]:
    """Compute combined layer overrides from all mixed quantization methods.

    Priority (highest to lowest):
    1. Manual --keep rules
    2. Calibration (not computed here, NVIDIA backend only)
    3. Statistics-based
    4. Heuristics-based
    """
    # Collect all tensor names
    tensor_names: list[str] = []
    for sf_path in safetensor_files:
        with safe_open(str(sf_path), framework="numpy") as f:
            tensor_names.extend(f.keys())

    overrides: dict[str, str] = {}

    # Layer 1: Heuristics (lowest priority)
    if config.mixed_heuristics:
        overrides.update(compute_heuristic_overrides(tensor_names))

    # Layer 2: Statistics
    if config.mixed_statistics is not None:
        stat_overrides = compute_statistics_overrides(
            safetensor_files, config.mixed_statistics
        )
        overrides.update(stat_overrides)

    # Layer 3: Manual keep rules (highest priority)
    if config.keep:
        keep_rules = parse_keep_rules(config.keep)
        manual_overrides = expand_keep_rules(keep_rules, tensor_names)
        overrides.update(manual_overrides)

    if overrides:
        log.info("Total layer overrides: %d/%d layers", len(overrides), len(tensor_names))

    return overrides
