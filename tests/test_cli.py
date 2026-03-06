"""Tests for CLI interface."""

from click.testing import CliRunner

from kuantala.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Kuantala" in result.output


def test_list_formats():
    runner = CliRunner()
    result = runner.invoke(cli, ["list-formats"])
    assert result.exit_code == 0
    assert "Q4_K" in result.output
    assert "MXFP8" in result.output


def test_quantize_missing_dtype():
    runner = CliRunner()
    result = runner.invoke(cli, ["quantize", "some-model"])
    assert result.exit_code != 0


def test_info_nonexistent_model():
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "/nonexistent/path"])
    assert result.exit_code != 0
