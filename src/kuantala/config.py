"""Configuration dataclass for quantization jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


GGUF_TYPES = [
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0",
]

NVIDIA_TYPES = ["MXFP8", "NVFP4"]

ALL_DTYPES = GGUF_TYPES + NVIDIA_TYPES

# Types allowed as per-component overrides (including skip and passthrough)
COMPONENT_DTYPES = ALL_DTYPES + ["F16", "F32", "BF16", "skip"]


def is_gguf_dtype(dtype: str) -> bool:
    return dtype in GGUF_TYPES


def is_nvidia_dtype(dtype: str) -> bool:
    return dtype in NVIDIA_TYPES


@dataclass
class QuantConfig:
    """Configuration for a quantization job."""

    model_source: str
    dtype: str
    output_dir: Path = Path("./output")

    # Per-component overrides ("skip" = don't touch, None = use main dtype)
    vae_dtype: str | None = "skip"
    te_dtype: str | None = None

    # Mixed quantization
    mixed_heuristics: bool = False
    mixed_statistics: int | None = None  # percentage, e.g. 10 = top 10%
    mixed_calibration: bool = False
    calibration_data: Path | None = None

    # Manual layer overrides: ["pattern:dtype", ...]
    keep: list[str] = field(default_factory=list)

    # HuggingFace auth token
    hf_token: str | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.dtype not in ALL_DTYPES:
            raise ValueError(
                f"Unknown dtype {self.dtype!r}. "
                f"Choose from: {', '.join(ALL_DTYPES)}"
            )
        for field_name, value in [("vae_dtype", self.vae_dtype), ("te_dtype", self.te_dtype)]:
            if value is not None and value not in COMPONENT_DTYPES:
                raise ValueError(
                    f"Unknown {field_name} {value!r}. "
                    f"Choose from: {', '.join(COMPONENT_DTYPES)}"
                )

    @property
    def backend_name(self) -> str:
        return "nvidia" if is_nvidia_dtype(self.dtype) else "gguf"
