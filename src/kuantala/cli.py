"""Click CLI for kuantala."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.table import Table

from kuantala.config import ALL_DTYPES, COMPONENT_DTYPES
from kuantala.utils import console, setup_logging

# Component types that can be quantized
_QUANTIZABLE_TYPES = {"transformer", "unet", "vae", "text_encoder", "image_encoder"}


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Kuantala - Quantize diffusion models to FP8 and NVFP4."""
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
              default="skip", help="Text encoder quantization dtype (default: skip).")
@click.option("--ie-dtype", type=click.Choice(COMPONENT_DTYPES, case_sensitive=False),
              default="skip", help="Image encoder quantization dtype (default: skip).")
@click.option("--keep", multiple=True,
              help="Disable quantization on layers matching this glob pattern (repeatable).")
def quantize(
    model: str,
    dtype: str,
    output: Path,
    vae_dtype: str,
    te_dtype: str,
    ie_dtype: str,
    keep: tuple[str, ...],
) -> None:
    """Quantize a diffusion model.

    MODEL is a HuggingFace diffusers model ID (e.g. Wan-AI/Wan2.1-I2V-14B-Diffusers)
    or a local directory path in diffusers format (with model_index.json).
    """
    from kuantala.config import QuantConfig
    from kuantala.core import quantize as run_quantize

    # Normalize case
    dtype = dtype.upper()
    if vae_dtype.lower() != "skip":
        vae_dtype = vae_dtype.upper()
    if te_dtype.lower() != "skip":
        te_dtype = te_dtype.upper()
    if ie_dtype.lower() != "skip":
        ie_dtype = ie_dtype.upper()

    config = QuantConfig(
        model_source=model,
        dtype=dtype,
        output_dir=output,
        vae_dtype=vae_dtype,
        te_dtype=te_dtype,
        ie_dtype=ie_dtype,
        keep=list(keep),
    )

    output_files = run_quantize(config)

    console.print()
    if output_files:
        console.print("[bold green]Quantization complete![/]")
        for f in output_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            console.print(f"  {f} ({size_mb:.1f} MB)")
    else:
        console.print("[yellow]No files were produced.[/]")


@cli.command()
@click.argument("model", metavar="MODEL_ID_OR_PATH")
@click.option("--show-all", is_flag=True, help="Show all components, including non-quantizable ones.")
def components(model: str, show_all: bool) -> None:
    """Show components of a diffusion model.

    MODEL is a HuggingFace diffusers model ID (e.g. Wan-AI/Wan2.1-I2V-14B-Diffusers)
    or a local directory path in diffusers format (with model_index.json).
    """
    model_dir = _resolve_model_dir_cached(model)
    index = _load_model_index(model_dir)
    model_type = index.get("_class_name")

    console.print(f"\n[bold]Model:[/] {model}")
    console.print(f"[bold]Path:[/] {model_dir}")
    if model_type:
        console.print(f"[bold]Pipeline:[/] {model_type}")

    table = Table(title="Components", title_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Class")
    table.add_column("Params", justify="right")
    table.add_column("Dtype", justify="right")
    table.add_column("Size", justify="right")

    from kuantala.components import _classify_component

    for key, value in index.items():
        if key.startswith("_") or value is None:
            continue
        if not isinstance(value, list):
            continue
        library = value[0] if len(value) >= 1 else None
        class_name = value[1] if len(value) >= 2 else None
        if library is None and class_name is None:
            continue
        comp_type = _classify_component(key, class_name, library)
        if not show_all and comp_type not in _QUANTIZABLE_TYPES:
            continue

        class_label = f"{library}.{class_name}" if library and class_name else ""

        # Read safetensors headers for params/dtype/size
        total_params = 0
        total_size = 0
        dtypes: set[str] = set()
        comp_dir = model_dir / key
        if comp_dir.is_dir():
            for sf_path in sorted(comp_dir.glob("*.safetensors")):
                header = _read_local_safetensors_header(sf_path)
                total_size += sf_path.stat().st_size
                total_params += _count_params_from_header(header)
                for tname, meta in header.items():
                    if tname != "__metadata__":
                        dtypes.add(meta.get("dtype", "unknown"))

        params_str = _format_params(total_params) if total_params > 0 else ""
        dtype_str = ", ".join(sorted(dtypes))

        if total_size > 0:
            gb = total_size / (1024 ** 3)
            size_str = f"{gb:.1f} GB" if gb >= 1.0 else f"{total_size / (1024 ** 2):.0f} MB"
        else:
            size_str = ""

        table.add_row(key, comp_type, class_label, params_str, dtype_str, size_str)

    console.print(table)


def _load_model_index(model_dir: Path) -> dict:
    """Load and return model_index.json from a model directory."""
    index_path = model_dir / "model_index.json"
    if not index_path.exists():
        raise click.ClickException(
            f"No model_index.json found in {model_dir}. "
            "The model directory must follow the HuggingFace diffusers layout."
        )
    with open(index_path) as f:
        return json.load(f)


def _resolve_model_dir_cached(model: str) -> Path:
    """Resolve a model to a local directory, using HF cache if available.

    For remote models, checks the cache first. If not cached, prompts
    the user before downloading.
    """
    local = Path(model)
    if local.is_dir():
        return local

    try:
        from huggingface_hub import scan_cache_dir, snapshot_download
    except ImportError:
        raise click.ClickException(
            f"'{model}' is not a local directory and huggingface-hub is not installed. "
            "Install with: pip install huggingface-hub"
        )

    # Check if model is already in the HF cache
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model and repo.repo_type == "model":
                # Find the latest revision snapshot
                for rev in sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True):
                    snapshot_path = rev.snapshot_path
                    if (snapshot_path / "model_index.json").exists():
                        return snapshot_path
    except Exception:
        pass

    # Not cached — ask user
    if not click.confirm(
        f"Model '{model}' is not cached locally. Download it?",
        default=True,
    ):
        raise click.Abort()

    cache_dir = snapshot_download(
        repo_id=model,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    )
    return Path(cache_dir)


def _read_local_safetensors_header(sf_path: Path) -> dict:
    """Read the JSON header from a local safetensors file."""
    with open(sf_path, "rb") as fh:
        header_size = int.from_bytes(fh.read(8), "little")
        return json.loads(fh.read(header_size))


def _count_params_from_header(header: dict) -> int:
    """Count total parameters from a safetensors header."""
    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for dim in meta.get("shape", []):
            count *= dim
        total += count
    return total


@cli.command("formats")
def list_formats() -> None:
    """List available quantization formats."""
    table = Table(title="Available Quantization Formats", title_style="bold")
    table.add_column("Format", style="cyan")
    table.add_column("Description")

    descriptions = {
        "FP8": "8-bit floating point (E4M3), ~50% size reduction. Requires Hopper+ GPU.",
        "NVFP4": "NVIDIA 4-bit floating point, ~75% size reduction. Requires Blackwell GPU.",
        "FP16": "16-bit floating point (passthrough/conversion).",
        "BF16": "Brain floating point 16 (passthrough/conversion).",
    }

    for dtype in ALL_DTYPES:
        table.add_row(dtype, descriptions.get(dtype, ""))

    console.print(table)


# Approximate bits per parameter for size estimation
_BITS_PER_PARAM = {
    "FP8": 8.5,
    "NVFP4": 4.5,
    "FP16": 16.0,
    "BF16": 16.0,
    "F32": 32.0,
}


@cli.command()
@click.argument("model", metavar="MODEL_ID_OR_PATH")
def estimate(model: str) -> None:
    """Estimate output sizes for each quantization format.

    Estimates are computed from parameter counts — no actual quantization
    is performed. VAE is skipped (default behavior).

    MODEL is a HuggingFace diffusers model ID or local directory path.
    """
    from kuantala.components import detect_components
    from kuantala.model_loader import resolve_model_path

    model_dir = resolve_model_path(model)
    model_info = detect_components(model_dir)

    total_params = 0
    for comp in model_info.components:
        if comp.component_type not in _QUANTIZABLE_TYPES - {"vae"}:
            continue

        sf_files = sorted(comp.path.glob("*.safetensors"))
        for sf_path in sf_files:
            header = _read_local_safetensors_header(sf_path)
            total_params += _count_params_from_header(header)

    if not total_params:
        console.print("[yellow]No quantizable components found.[/]")
        return

    console.print(f"\n[bold]Model:[/] {model}")
    console.print(f"[bold]Pipeline:[/] {model_info.model_type or 'unknown'}")
    console.print(f"[bold]Total parameters (excl. VAE):[/] {_format_params(total_params)}")

    def _fmt(b: float) -> str:
        gb = b / (1024 ** 3)
        if gb >= 1.0:
            return f"{gb:.1f} GB"
        return f"{b / (1024 ** 2):.0f} MB"

    table = Table(title="Estimated Output Sizes", title_style="bold")
    table.add_column("Format", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("vs FP16", justify="right")

    fp16_bytes = total_params * _BITS_PER_PARAM["FP16"] / 8
    for dtype in ALL_DTYPES:
        bpp = _BITS_PER_PARAM.get(dtype, 16.0)
        size_bytes = total_params * bpp / 8
        pct = size_bytes / fp16_bytes * 100
        table.add_row(dtype, _fmt(size_bytes), f"{pct:.0f}%")

    console.print(table)


@cli.command()
@click.argument("model", metavar="MODEL_ID_OR_PATH")
def config(model: str) -> None:
    """Show the architecture of a diffusion model from its config.

    Loads model configs (no weights) to show the full module hierarchy.
    Requires torch and diffusers/transformers to be installed.

    MODEL is a HuggingFace diffusers model ID (e.g. Wan-AI/Wan2.1-I2V-14B-Diffusers)
    or a local directory path in diffusers format (with model_index.json).
    """
    import importlib

    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "The config command requires torch and diffusers."
        )

    model_dir = _resolve_model_dir_cached(model)
    index = _load_model_index(model_dir)

    model_type = index.get("_class_name")
    console.print(f"\n[bold]Model:[/] {model}")
    if model_type:
        console.print(f"[bold]Pipeline:[/] {model_type}")

    from kuantala.components import _classify_component

    for key, value in index.items():
        if key.startswith("_") or value is None or not isinstance(value, list):
            continue
        library = value[0] if len(value) >= 1 else None
        class_name = value[1] if len(value) >= 2 else None
        if library is None and class_name is None:
            continue
        comp_type = _classify_component(key, class_name, library)
        if comp_type not in _QUANTIZABLE_TYPES:
            continue

        config_path = model_dir / key / "config.json"
        if not config_path.exists():
            continue

        full_class = f"{library}.{class_name}"
        try:
            import torch
            lib = importlib.import_module(library)
            cls = getattr(lib, class_name)
            with torch.device("meta"):
                if library == "transformers":
                    from transformers import AutoConfig
                    cfg = AutoConfig.from_pretrained(str(config_path.parent))
                    model_instance = cls(cfg)
                else:
                    model_instance = cls.from_config(str(config_path.parent))
        except Exception as e:
            console.print(f"\n[yellow]Could not load {full_class} for '{key}': {e}[/]")
            continue

        total_params = sum(p.numel() for p in model_instance.parameters())

        console.print(f"\n[bold cyan]{key}[/] [dim]({full_class}, {_format_params(total_params)} params)[/]")
        _print_module_tree(model_instance, prefix="")


def _print_module_tree(module: object, prefix: str) -> None:
    """Print a module hierarchy as a tree."""
    import torch.nn as nn

    children = list(module.named_children()) if isinstance(module, nn.Module) else []
    if not children:
        return

    for i, (name, child) in enumerate(children):
        is_last = i == len(children) - 1
        connector = "\u2514\u2500 " if is_last else "\u251c\u2500 "
        child_prefix = prefix + ("   " if is_last else "\u2502  ")

        child_type = type(child).__name__
        param_count = sum(p.numel() for p in child.parameters())
        param_str = f" ({_format_params(param_count)})" if param_count > 0 else ""

        extra = ""
        if isinstance(child, nn.Linear):
            extra = f" in={child.in_features}, out={child.out_features}"
        elif isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            extra = f" in={child.in_channels}, out={child.out_channels}, k={child.kernel_size}"
        elif isinstance(child, (nn.LayerNorm, nn.GroupNorm)):
            extra = f" {list(child.normalized_shape)}" if hasattr(child, 'normalized_shape') else ""

        console.print(f"{prefix}{connector}[cyan]{name}[/] [dim]{child_type}{extra}{param_str}[/]")
        _print_module_tree(child, child_prefix)


@cli.command()
@click.argument("input_file", metavar="INPUT", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Output file path (default: input stem + '-comfyui.safetensors').")
def convert(input_file: Path, output: Path | None) -> None:
    """Convert a kuantala NVFP4 safetensors file to ComfyUI format."""
    if input_file.suffix != ".safetensors":
        raise click.ClickException(f"Input must be a .safetensors file, got: {input_file.suffix}")

    if output is None:
        output = input_file.parent / f"{input_file.stem}-comfyui.safetensors"

    from kuantala.convert import convert_to_comfyui

    console.print(f"[bold]Input:[/]  {input_file}")
    console.print(f"[bold]Output:[/] {output}")

    # Count quantized layers for summary
    from safetensors.torch import load_file
    state_dict = load_file(str(input_file))
    n_layers = sum(1 for k in state_dict if k.endswith(".weight_quantizer._scale"))

    if n_layers == 0:
        raise click.ClickException("No NVFP4-quantized layers found in input file.")

    convert_to_comfyui(input_file, output)

    input_mb = input_file.stat().st_size / (1024 * 1024)
    output_mb = output.stat().st_size / (1024 * 1024)
    console.print(f"\n[bold green]Converted {n_layers} layers[/]")
    console.print(f"  Input:  {input_mb:.1f} MB")
    console.print(f"  Output: {output_mb:.1f} MB")


@cli.command()
@click.argument("file", metavar="FILE_PATH", type=click.Path(exists=True, path_type=Path))
def tensors(file: Path) -> None:
    """Show tensors in a safetensors file.

    Shows per-tensor name, dtype, shape, and parameter count.
    """
    if file.suffix == ".safetensors":
        _inspect_safetensors(file)
    else:
        raise click.ClickException(f"Unsupported file format: {file.suffix}. Use .safetensors")


def _format_params(count: int) -> str:
    """Format parameter count as human-readable string."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def _inspect_safetensors(file: Path) -> None:
    """Inspect a safetensors file."""
    header = _read_local_safetensors_header(file)

    file_size_mb = file.stat().st_size / (1024 * 1024)
    console.print(f"\n[bold]File:[/] {file}")
    console.print(f"[bold]Format:[/] safetensors")
    console.print(f"[bold]Size:[/] {file_size_mb:.1f} MB")

    dtype_counts: dict[str, int] = {}
    dtype_params: dict[str, int] = {}
    total_params = 0
    tensor_list: list[tuple[str, str, list[int], int]] = []

    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta.get("dtype", "unknown")
        shape = meta.get("shape", [])
        param_count = 1
        for dim in shape:
            param_count *= dim
        total_params += param_count
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        dtype_params[dtype] = dtype_params.get(dtype, 0) + param_count
        tensor_list.append((name, dtype, shape, param_count))

    _print_layers_and_summary(tensor_list, dtype_counts, dtype_params, total_params)


def _natural_sort_key(name: str) -> list:
    """Sort key that handles numeric parts naturally (block.2 < block.10)."""
    import re
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def _print_layers_and_summary(
    tensors: list[tuple[str, str, list[int], int]],
    dtype_counts: dict[str, int],
    dtype_params: dict[str, int],
    total_params: int,
) -> None:
    """Print tensor detail table followed by dtype summary."""
    tensors = sorted(tensors, key=lambda t: _natural_sort_key(t[0]))

    table = Table(title="Tensors", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Name")
    table.add_column("Dtype", style="cyan")
    table.add_column("Shape")
    table.add_column("Parameters", justify="right")

    for i, (name, dtype, shape, param_count) in enumerate(tensors):
        shape_str = "\u00d7".join(str(d) for d in shape)
        table.add_row(str(i + 1), name, dtype, shape_str, _format_params(param_count))

    console.print(table)

    summary = Table(title="Dtype Summary", title_style="bold")
    summary.add_column("Dtype", style="cyan")
    summary.add_column("Tensors", justify="right")
    summary.add_column("Parameters", justify="right")
    summary.add_column("% of Total", justify="right")
    for dtype in sorted(dtype_counts.keys()):
        pct = dtype_params[dtype] / total_params * 100 if total_params else 0
        summary.add_row(dtype, str(dtype_counts[dtype]), _format_params(dtype_params[dtype]), f"{pct:.1f}%")
    summary.add_section()
    summary.add_row("[bold]Total[/]", f"[bold]{len(tensors)}[/]", f"[bold]{_format_params(total_params)}[/]", "[bold]100%[/]")
    console.print(summary)
