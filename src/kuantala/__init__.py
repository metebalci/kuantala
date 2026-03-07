"""Kuantala - Generative model quantization tool."""


def __getattr__(name: str):
    if name == "QuantConfig":
        from kuantala.config import QuantConfig
        return QuantConfig
    if name == "quantize":
        from kuantala.core import quantize
        return quantize
    if name == "convert_to_comfyui":
        from kuantala.convert import convert_to_comfyui
        return convert_to_comfyui
    raise AttributeError(f"module 'kuantala' has no attribute {name!r}")


__all__ = ["QuantConfig", "quantize", "convert_to_comfyui"]
