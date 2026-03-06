"""Tests for CLI interface."""

from click.testing import CliRunner

from kuantala.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Kuantala" in result.output


def test_formats():
    runner = CliRunner()
    result = runner.invoke(cli, ["formats"])
    assert result.exit_code == 0
    assert "Q4_K" in result.output
    assert "MXFP8" in result.output


def test_quantize_missing_dtype():
    runner = CliRunner()
    result = runner.invoke(cli, ["quantize", "some-model"])
    assert result.exit_code != 0


def test_components_nonexistent_model():
    runner = CliRunner()
    result = runner.invoke(cli, ["components", "/nonexistent/path"])
    assert result.exit_code != 0


def test_layers_nonexistent_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["layers", "/nonexistent/file.gguf"])
    assert result.exit_code != 0


def test_layers_unsupported_format(tmp_path):
    dummy = tmp_path / "model.bin"
    dummy.write_bytes(b"dummy")
    runner = CliRunner()
    result = runner.invoke(cli, ["layers", str(dummy)])
    assert result.exit_code != 0
    assert "Unsupported file format" in result.output
