"""Click CLI for kuantala."""

from __future__ import annotations

from pathlib import Path

import click
from rich.table import Table

from kuantala.config import ALL_DTYPES, COMPONENT_DTYPES, QuantConfig
from kuantala.utils import console, setup_logging


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Kuantala - Quantize diffusion models to GGUF, MXFP8, NVFP4."""
    setup_logging(verbose=verbose)


@cli.command()
@click.argument("model", metavar="MODEL_ID_OR_PATH")
@click.option("--dtype", "-d", required=True, type=click.Choice(ALL_DTYPES, case_sensitive=False),
              help="Target quantization type.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=Path("./output"),
              help="Output directory.")
@click.option("--vae-dtype", type=click.Choice(COMPONENT_DTYPES, case_sensitive=False),
              default="skip", help="VAE quantization dtype (default: skip).")
@click.option("--te-dtype", type=click.Choice(COMPONENT_DTYPES, case_sensitive=False),
              default=None, help="Text encoder quantization dtype (default: same as --dtype).")
@click.option("--mixed-heuristics", is_flag=True,
              help="Preserve known-sensitive layers at higher precision.")
@click.option("--mixed-statistics", type=int, default=None, metavar="N",
              help="Preserve top N%% most sensitive layers (by weight statistics).")
@click.option("--mixed-calibration", is_flag=True,
              help="Use calibration forward passes to find sensitive layers (NVIDIA only).")
@click.option("--calibration-data", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to calibration data directory.")
@click.option("--keep", multiple=True,
              help="Manual layer override: 'pattern:dtype' (repeatable).")
@click.option("--hf-token", envvar="HF_TOKEN", default=None,
              help="HuggingFace auth token (optional, also uses token from `hf auth login`).")
def quantize(
    model: str,
    dtype: str,
    output: Path,
    vae_dtype: str,
    te_dtype: str | None,
    mixed_heuristics: bool,
    mixed_statistics: int | None,
    mixed_calibration: bool,
    calibration_data: Path | None,
    keep: tuple[str, ...],
    hf_token: str | None,
) -> None:
    """Quantize a diffusion model.

    MODEL is a HuggingFace diffusers model ID (e.g. Wan-AI/Wan2.1-I2V-14B-Diffusers)
    or a local directory path in diffusers format (with model_index.json).
    """
    from kuantala.core import quantize as run_quantize

    # Normalize case (Click case_sensitive=False passes through original case)
    dtype = dtype.upper()
    if vae_dtype and vae_dtype.lower() != "skip":
        vae_dtype = vae_dtype.upper()
    if te_dtype:
        te_dtype = te_dtype.upper()

    config = QuantConfig(
        model_source=model,
        dtype=dtype,
        output_dir=output,
        vae_dtype=vae_dtype,
        te_dtype=te_dtype,
        mixed_heuristics=mixed_heuristics,
        mixed_statistics=mixed_statistics,
        mixed_calibration=mixed_calibration,
        calibration_data=calibration_data,
        keep=list(keep),
        hf_token=hf_token,
    )

    output_files = run_quantize(config)

    console.print()
    if output_files:
        console.print(f"[bold green]Quantization complete![/]")
        for f in output_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            console.print(f"  {f} ({size_mb:.1f} MB)")
    else:
        console.print("[yellow]No files were produced.[/]")


@cli.command()
@click.argument("model", metavar="MODEL_ID_OR_PATH")
@click.option("--hf-token", envvar="HF_TOKEN", default=None,
              help="HuggingFace auth token (optional, also uses token from `hf auth login`).")
def info(model: str, hf_token: str | None) -> None:
    """Inspect a diffusion model's components.

    MODEL is a HuggingFace diffusers model ID (e.g. Wan-AI/Wan2.1-I2V-14B-Diffusers)
    or a local directory path in diffusers format (with model_index.json).
    """
    from kuantala.components import detect_components
    from kuantala.model_loader import resolve_model_path

    model_dir = resolve_model_path(model, hf_token)
    model_info = detect_components(model_dir)

    console.print(f"\n[bold]Model:[/] {model}")
    console.print(f"[bold]Path:[/] {model_dir}")
    if model_info.model_type:
        console.print(f"[bold]Pipeline:[/] {model_info.model_type}")

    table = Table(title="Components")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Path")
    table.add_column("Safetensors Files", justify="right")

    for comp in model_info.components:
        n_files = len(list(comp.path.glob("*.safetensors")))
        table.add_row(comp.name, comp.component_type, str(comp.path), str(n_files))

    console.print(table)


@cli.command("list-formats")
def list_formats() -> None:
    """List available quantization formats."""
    from kuantala.config import GGUF_TYPES, NVIDIA_TYPES

    table = Table(title="Available Quantization Formats")
    table.add_column("Format", style="cyan")
    table.add_column("Backend", style="green")
    table.add_column("Description")

    descriptions = {
        "Q2_K": "2-bit K-quant (very aggressive, quality loss)",
        "Q3_K": "3-bit K-quant",
        "Q4_0": "4-bit basic quantization",
        "Q4_K": "4-bit K-quant (recommended)",
        "Q5_0": "5-bit basic quantization",
        "Q5_K": "5-bit K-quant",
        "Q6_K": "6-bit K-quant (high quality)",
        "Q8_0": "8-bit quantization (near lossless)",
        "MXFP8": "Microscaling FP8 (NVIDIA, requires Hopper+)",
        "NVFP4": "NVIDIA FP4 (requires Blackwell)",
    }

    for dtype in GGUF_TYPES:
        table.add_row(dtype, "GGUF", descriptions.get(dtype, ""))
    for dtype in NVIDIA_TYPES:
        table.add_row(dtype, "NVIDIA", descriptions.get(dtype, ""))

    console.print(table)
