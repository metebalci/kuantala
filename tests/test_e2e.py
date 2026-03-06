"""End-to-end tests with a real model from HuggingFace."""

import pytest
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    HAS_HUB = True
except ImportError:
    HAS_HUB = False

requires_hub = pytest.mark.skipif(
    not HAS_HUB,
    reason="Requires huggingface-hub (pip install kuantala[hub])",
)

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


@pytest.fixture(scope="module")
def model_dir():
    """Download SD 1.5 (cached after first run)."""
    return Path(snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    ))


@requires_hub
def test_info(model_dir):
    """Verify component detection on a real model."""
    from kuantala.components import detect_components

    info = detect_components(model_dir)
    assert info.model_type == "StableDiffusionPipeline"

    unet = info.get("unet")
    assert unet is not None

    vae = info.get("vae")
    assert vae is not None

    te = info.get("text_encoder")
    assert te is not None


@requires_hub
def test_quantize_q8_0(model_dir, tmp_path):
    """End-to-end GGUF Q8_0 quantization."""
    from kuantala import QuantConfig, quantize

    config = QuantConfig(
        model_source=str(model_dir),
        dtype="Q8_0",
        output_dir=tmp_path / "q8_0",
        vae_dtype="skip",
    )
    output_files = quantize(config)

    assert len(output_files) >= 2  # unet + text_encoder (vae skipped)
    for f in output_files:
        assert f.exists()
        assert f.suffix == ".gguf"
        assert f.stat().st_size > 0


@requires_hub
def test_quantize_q4_k_mixed(model_dir, tmp_path):
    """End-to-end GGUF Q4_K with mixed heuristics."""
    from kuantala import QuantConfig, quantize

    config = QuantConfig(
        model_source=str(model_dir),
        dtype="Q4_K",
        output_dir=tmp_path / "q4km",
        vae_dtype="skip",
        te_dtype="Q8_0",
        mixed_heuristics=True,
    )
    output_files = quantize(config)

    assert len(output_files) >= 2
    for f in output_files:
        assert f.exists()
        assert f.suffix == ".gguf"
        assert f.stat().st_size > 0
