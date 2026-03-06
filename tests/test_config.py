"""Tests for config module."""

import pytest

from kuantala.config import QuantConfig


def test_valid_config():
    config = QuantConfig(model_source="test", dtype="FP8")
    assert config.dtype == "FP8"
    assert config.vae_dtype == "skip"
    assert config.te_dtype == "skip"
    assert config.calibration is True


def test_valid_nvfp4():
    config = QuantConfig(model_source="test", dtype="NVFP4")
    assert config.dtype == "NVFP4"


def test_passthrough_fp16():
    config = QuantConfig(model_source="test", dtype="FP16")
    assert config.dtype == "FP16"


def test_invalid_dtype():
    with pytest.raises(ValueError, match="Unknown dtype"):
        QuantConfig(model_source="test", dtype="Q4_K")


def test_invalid_component_dtype():
    with pytest.raises(ValueError, match="Unknown vae_dtype"):
        QuantConfig(model_source="test", dtype="FP8", vae_dtype="Q8_0")


def test_keep_patterns():
    config = QuantConfig(model_source="test", dtype="FP8", keep=["norm_*", "attn_*"])
    assert len(config.keep) == 2
