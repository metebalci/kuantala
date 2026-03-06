"""Kuantala - Diffusion model quantization tool."""


def __getattr__(name: str):
    if name == "QuantConfig":
        from kuantala.config import QuantConfig
        return QuantConfig
    if name == "quantize":
        from kuantala.core import quantize
        return quantize
    raise AttributeError(f"module 'kuantala' has no attribute {name!r}")


__all__ = ["QuantConfig", "quantize"]
