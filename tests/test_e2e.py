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
    reason="Requires huggingface-hub",
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
def test_components(model_dir):
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
def test_components_cli(model_dir):
    """Verify components CLI command on a real model."""
    from click.testing import CliRunner
    from kuantala.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["components", str(model_dir)])
    assert result.exit_code == 0
    assert "StableDiffusionPipeline" in result.output
    assert "unet" in result.output
    assert "vae" in result.output
    assert "text_encoder" in result.output


@requires_hub
def test_config_cli(model_dir):
    """Verify config CLI command on a real model."""
    pytest.importorskip("torch")
    from click.testing import CliRunner
    from kuantala.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["config", str(model_dir)])
    assert result.exit_code == 0
    assert "StableDiffusionPipeline" in result.output
    assert "unet" in result.output


@requires_hub
def test_tensors_safetensors(model_dir):
    """Verify tensors CLI command on a safetensors file."""
    from click.testing import CliRunner
    from kuantala.cli import cli

    sf_files = list(model_dir.glob("**/*.safetensors"))
    assert len(sf_files) > 0

    runner = CliRunner()
    result = runner.invoke(cli, ["tensors", str(sf_files[0])])
    assert result.exit_code == 0
    assert "Dtype Summary" in result.output
    assert "Tensors" in result.output
