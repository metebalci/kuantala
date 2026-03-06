"""Configuration dataclass for quantization jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


QUANT_DTYPES = ["FP8", "NVFP4"]

PASSTHROUGH_DTYPES = ["FP16", "BF16"]

ALL_DTYPES = QUANT_DTYPES + PASSTHROUGH_DTYPES

# Types allowed as per-component overrides (including skip)
COMPONENT_DTYPES = ALL_DTYPES + ["skip"]


def is_quant_dtype(dtype: str) -> bool:
    return dtype in QUANT_DTYPES


def is_passthrough_dtype(dtype: str) -> bool:
    return dtype in PASSTHROUGH_DTYPES


@dataclass
class QuantConfig:
    """Configuration for a quantization job."""

    model_source: str
    dtype: str
    output_dir: Path = Path("./output")

    # Per-component overrides ("skip" = don't touch, None = use main dtype)
    vae_dtype: str | None = "skip"
    te_dtype: str | None = None
    ie_dtype: str | None = None

    # Calibration (random forward passes to determine optimal scale factors)
    calibration: bool = True
    calib_size: int = 4  # number of calibration batches

    # Manual layer overrides: disable quantization on matched layer names
    keep: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.dtype not in ALL_DTYPES:
            raise ValueError(
                f"Unknown dtype {self.dtype!r}. "
                f"Choose from: {', '.join(ALL_DTYPES)}"
            )
        for field_name, value in [("vae_dtype", self.vae_dtype), ("te_dtype", self.te_dtype), ("ie_dtype", self.ie_dtype)]:
            if value is not None and value not in COMPONENT_DTYPES:
                raise ValueError(
                    f"Unknown {field_name} {value!r}. "
                    f"Choose from: {', '.join(COMPONENT_DTYPES)}"
                )
