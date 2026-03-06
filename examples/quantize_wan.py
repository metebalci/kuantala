"""Example: Quantize Wan2.1-I2V-14B to Q4_K_M with mixed heuristics."""

from pathlib import Path

from kuantala import QuantConfig, quantize

config = QuantConfig(
    model_source="Wan-AI/Wan2.1-I2V-14B",
    dtype="Q4_K_M",
    vae_dtype="skip",
    output_dir=Path("./wan-q4km"),
    mixed_heuristics=True,
    mixed_statistics=10,
    keep=["*norm*:F16"],
)

output_files = quantize(config)
print(f"Quantized files: {output_files}")
