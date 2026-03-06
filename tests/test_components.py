"""Tests for components module."""

import json
from pathlib import Path

import pytest

from kuantala.components import detect_components


def _make_fake_safetensor(path: Path) -> None:
    """Create a minimal fake safetensors file (just the header)."""
    # Minimal valid safetensors: 8-byte header length + empty JSON header
    header = b'{"__metadata__":{}}'
    header_len = len(header).to_bytes(8, byteorder="little")
    path.write_bytes(header_len + header)


def test_detect_from_model_index(tmp_path):
    """Should parse model_index.json and detect components."""
    # Create model_index.json
    index = {
        "_class_name": "StableDiffusionPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "scheduler": ["diffusers", "PNDMScheduler"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
    }
    (tmp_path / "model_index.json").write_text(json.dumps(index))

    # Create component directories with safetensors
    for name in ["unet", "vae", "text_encoder"]:
        comp_dir = tmp_path / name
        comp_dir.mkdir()
        _make_fake_safetensor(comp_dir / "model.safetensors")

    # Scheduler dir exists but has no safetensors
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler" / "config.json").write_text("{}")

    # Tokenizer dir
    (tmp_path / "tokenizer").mkdir()

    info = detect_components(tmp_path)
    assert info.model_type == "StableDiffusionPipeline"
    assert len(info.components) == 3

    unet = info.get("unet")
    assert unet is not None
    assert unet.name == "unet"

    vae = info.get("vae")
    assert vae is not None

    te = info.get("text_encoder")
    assert te is not None


def test_detect_fallback_scan(tmp_path):
    """Should fall back to directory scanning when no model_index.json."""
    comp_dir = tmp_path / "transformer"
    comp_dir.mkdir()
    _make_fake_safetensor(comp_dir / "model.safetensors")

    info = detect_components(tmp_path)
    assert len(info.components) == 1
    assert info.components[0].component_type == "transformer"


def test_detect_single_component(tmp_path):
    """Should detect single-component model (safetensors in root)."""
    _make_fake_safetensor(tmp_path / "model.safetensors")

    info = detect_components(tmp_path)
    assert len(info.components) == 1
    assert info.components[0].name == "model"
