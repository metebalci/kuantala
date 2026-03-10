"""Tests for CLI interface."""

from click.testing import CliRunner

from kuantala.cli import cli, _parse_prompts_file


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Kuantala" in result.output


def test_info():
    runner = CliRunner()
    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "FP8" in result.output
    assert "NVFP4" in result.output


def test_quantize_missing_dtype():
    runner = CliRunner()
    result = runner.invoke(cli, ["quantize", "some-model"])
    assert result.exit_code != 0


def test_components_nonexistent_model():
    runner = CliRunner()
    result = runner.invoke(cli, ["components", "/nonexistent/path"])
    assert result.exit_code != 0


def test_config_nonexistent_model():
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "/nonexistent/path"])
    assert result.exit_code != 0


def test_tensors_nonexistent_file():
    runner = CliRunner()
    result = runner.invoke(cli, ["tensors", "/nonexistent/file.safetensors"])
    assert result.exit_code != 0


def test_tensors_unsupported_format(tmp_path):
    dummy = tmp_path / "model.bin"
    dummy.write_bytes(b"dummy")
    runner = CliRunner()
    result = runner.invoke(cli, ["tensors", str(dummy)])
    assert result.exit_code != 0
    assert "Unsupported file format" in result.output


def test_parse_prompts_file_text_only(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("a cat on a table\na dog in a park\n\n")
    prompts, images = _parse_prompts_file(f)
    assert prompts == ["a cat on a table", "a dog in a park"]
    assert images is None


def test_parse_prompts_file_with_images(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("a cat image:/data/cat.jpg\na dog in a park\nsunset image:/data/sunset.png\n")
    prompts, images = _parse_prompts_file(f)
    assert prompts == ["a cat", "a dog in a park", "sunset"]
    assert images == ["/data/cat.jpg", None, "/data/sunset.png"]


def test_parse_prompts_file_all_images(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("a cat image:/data/cat.jpg\na dog image:/data/dog.png\n")
    prompts, images = _parse_prompts_file(f)
    assert prompts == ["a cat", "a dog"]
    assert images == ["/data/cat.jpg", "/data/dog.png"]


def test_parse_prompts_file_image_in_prompt_text(tmp_path):
    """Ensure 'image:' only splits on the last occurrence preceded by a space."""
    f = tmp_path / "prompts.txt"
    f.write_text("generate an image: a cat sitting image:/data/cat.jpg\n")
    prompts, images = _parse_prompts_file(f)
    assert prompts == ["generate an image: a cat sitting"]
    assert images == ["/data/cat.jpg"]


def test_parse_prompts_file_empty(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("\n\n\n")
    prompts, images = _parse_prompts_file(f)
    assert prompts == []
    assert images is None
