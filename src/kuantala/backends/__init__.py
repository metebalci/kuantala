"""Quantization backend interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from kuantala.config import QuantConfig


class QuantBackend(ABC):
    """Abstract base class for quantization backends."""

    @abstractmethod
    def quantize_component(
        self,
        component_path: Path,
        output_path: Path,
        dtype: str,
        config: QuantConfig,
        layer_overrides: dict[str, str] | None = None,
    ) -> Path:
        """Quantize a single model component.

        Args:
            component_path: Directory containing safetensors files.
            output_path: Where to write the quantized output.
            dtype: Target quantization type (e.g. "Q4_K", "MXFP8").
            config: Full quantization config.
            layer_overrides: Per-layer dtype overrides {layer_name: dtype}.

        Returns:
            Path to the output file.
        """

    @abstractmethod
    def supported_dtypes(self) -> list[str]:
        """Return list of supported dtype strings."""

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return backend info for display."""


def get_backend(name: str) -> QuantBackend:
    """Get a quantization backend by name."""
    if name == "gguf":
        from kuantala.backends.gguf_backend import GGUFBackend
        return GGUFBackend()
    elif name == "nvidia":
        from kuantala.backends.nvidia_backend import NvidiaBackend
        return NvidiaBackend()
    else:
        raise ValueError(f"Unknown backend: {name!r}. Choose 'gguf' or 'nvidia'.")
