"""Click CLI for kuantala."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.table import Table

from kuantala.config import DTYPES, CALIB_ALGORITHMS, COMPONENT_DTYPES, DEFAULT_KEEPS, DEFAULT_KEEPS_NAMES, PROMPT_SOURCES
from kuantala.utils import console, setup_logging

# Component types that can be quantized
_QUANTIZABLE_TYPES = {"transformer", "unet", "vae", "text_encoder", "image_encoder"}


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Kuantala - Quantize generative models."""
    setup_logging(verbose=verbose)


@cli.command()
@click.argument("model", metavar="MODEL_ID_OR_PATH")
@click.option("--dtype", "-d", type=click.Choice(DTYPES, case_sensitive=False), default="NVFP4",
              help="Target quantization type (default: NVFP4).")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Output directory (default: output-<MODEL_ID>).")
@click.option("--vae-dtype", type=click.Choice(COMPONENT_DTYPES, case_sensitive=False),
              default="skip", help="VAE quantization dtype (default: skip).")
@click.option("--te-dtype", type=click.Choice(COMPONENT_DTYPES, case_sensitive=False),
              default="skip", help="Text encoder quantization dtype (default: skip).")
@click.option("--ie-dtype", type=click.Choice(COMPONENT_DTYPES, case_sensitive=False),
              default="skip", help="Image encoder quantization dtype (default: skip).")
@click.option("--keep", multiple=True,
              help="Disable quantization on layers matching this glob pattern (repeatable).")
@click.option("--use-default-keeps", type=click.Choice(DEFAULT_KEEPS_NAMES, case_sensitive=False),
              default=None, help="Apply preset keep patterns (auto-detected for known HF model IDs).")
@click.option("--no-default-keeps", is_flag=True, help="Disable auto-detected default keep patterns.")
@click.option("--algorithm", type=click.Choice(CALIB_ALGORITHMS, case_sensitive=False),
              default="max", help="Calibration algorithm (default: max).")
@click.option("--prompts", type=click.Path(exists=True, path_type=Path), default=None,
              help="File with calibration prompts, one per line (default: HF dataset).")
@click.option("--nprompts", type=int, default=32,
              help="Number of calibration prompts to use (default: 32).")
@click.option("--nsteps", type=int, default=None,
              help="Number of inference steps per calibration prompt (default: auto or 30).")
@click.option("--resolution", type=str, default=None,
              help="Calibration resolution: 480p, 540p, 720p, 1080p, 4k, or HEIGHTxWIDTH (default: auto or 480p).")
@click.option("--psrc", type=click.Choice(PROMPT_SOURCES, case_sensitive=False), default=None,
              help="Prompt source: t2i, t2v, i2v (auto-detected for known HF model IDs).")
@click.option("--offload", type=click.Choice(["model", "layers"], case_sensitive=False), default=None,
              help="CPU offload mode: 'model' (component-level) or 'layers' (layer-level, slower but less VRAM).")
def quantize(
    model: str,
    dtype: str,
    output: Path,
    vae_dtype: str,
    te_dtype: str,
    ie_dtype: str,
    keep: tuple[str, ...],
    use_default_keeps: str | None,
    no_default_keeps: bool,
    algorithm: str,
    prompts: Path | None,
    nprompts: int,
    nsteps: int | None,
    resolution: str | None,
    psrc: str | None,
    offload: str | None,
) -> None:
    """Quantize a generative model.

    MODEL is a HuggingFace diffusers model ID (e.g. Wan-AI/Wan2.2-I2V-A14B-Diffusers)
    or a local directory path in diffusers format (with model_index.json).
    """
    from kuantala.config import QuantConfig, get_model_defaults
    from kuantala.core import quantize as run_quantize

    defaults = get_model_defaults(model)

    # Default output directory based on model name
    if output is None:
        safe_name = model.replace("/", "-").strip("-")
        output = Path(f"output-{safe_name}")

    if nsteps is None:
        nsteps = defaults.get("steps", 30)
    if resolution is None:
        resolution = defaults.get("resolution")
    calib_resolution = _parse_resolution(resolution) if isinstance(resolution, str) else resolution or (480, 848)

    # Normalize case
    dtype = dtype.upper()
    if vae_dtype.lower() != "skip":
        vae_dtype = vae_dtype.upper()
    if te_dtype.lower() != "skip":
        te_dtype = te_dtype.upper()
    if ie_dtype.lower() != "skip":
        ie_dtype = ie_dtype.upper()

    # Load custom calibration prompts from file if provided
    prompt_list = None
    if prompts is not None:
        prompt_list = [line.strip() for line in prompts.read_text().splitlines() if line.strip()]

    config = QuantConfig(
        model_source=model,
        dtype=dtype,
        output_dir=output,
        vae_dtype=vae_dtype,
        te_dtype=te_dtype,
        ie_dtype=ie_dtype,
        algorithm=algorithm,
        default_keeps=use_default_keeps,
        no_default_keeps=no_default_keeps,
        calib_size=nprompts,
        calib_steps=nsteps,
        calib_resolution=calib_resolution,
        calib_prompts=prompt_list,
        num_frames=defaults.get("num_frames"),
        offload=offload,
        prompt_source=psrc,
        keep=list(keep),
    )

    output_files = run_quantize(config)

    console.print()
    if output_files:
        console.print("[bold green]Quantization complete![/]")
        for f in output_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            console.print(f"  {f} ({size_mb:.1f} MB)")

        # Write quantize.md summary
        md_path = output / "quantize.md"
        md_path.write_text(_generate_quantize_markdown(config, output_files))
        console.print(f"\n[dim]Saved quantization summary to {md_path}[/]")
    else:
        console.print("[yellow]No files were produced.[/]")


_RESOLUTION_PRESETS = {"480p": (480, 848), "540p": (540, 960), "720p": (720, 1280), "1080p": (1080, 1920), "4k": (2160, 3840)}


def _parse_resolution(resolution: str) -> tuple[int, int]:
    """Parse a resolution string into (height, width) tuple."""
    if resolution.lower() in _RESOLUTION_PRESETS:
        return _RESOLUTION_PRESETS[resolution.lower()]
    try:
        h, w = resolution.lower().split("x")
        return (int(h), int(w))
    except (ValueError, AttributeError):
        raise click.BadParameter(
            f"Invalid resolution: {resolution!r}. Use 480p, 540p, 720p, 1080p, 4k, or HEIGHTxWIDTH."
        )


@cli.command()
@click.argument("model", metavar="MODEL_ID_OR_PATH")
@click.option("--dtypes", "-d", multiple=True, default=("NVFP4",),
              type=click.Choice(DTYPES, case_sensitive=False),
              help="Quantization formats to consider (repeatable, default: NVFP4).")
@click.option("--effective-bits", type=float, default=4.8,
              help="Target average bits per parameter (default: 4.8).")
@click.option("--nprompts", type=int, default=8,
              help="Number of pipeline runs to capture inputs (default: 8).")
@click.option("--nsteps", type=int, default=None,
              help="Inference steps per pipeline run (default: auto or 10).")
@click.option("--resolution", type=str, default=None,
              help="Calibration resolution: 480p, 540p, 720p, 1080p, 4k, or HEIGHTxWIDTH (default: auto or 480p).")
@click.option("--offload", type=click.Choice(["model", "layers"], case_sensitive=False), default=None,
              help="CPU offload mode: 'model' (component-level) or 'layers' (layer-level, slower but less VRAM).")
def analyze(
    model: str,
    dtypes: tuple[str, ...],
    effective_bits: float,
    nprompts: int,
    nsteps: int | None,
    resolution: str | None,
    offload: str | None,
) -> None:
    """Analyze optimal per-layer quantization format selection.

    Uses modelopt's auto_quantize to find the best quantization format for each
    layer given an effective bits constraint. No output files are saved.

    MODEL is a HuggingFace diffusers model ID or local directory path.
    """
    from kuantala.config import get_model_defaults
    from kuantala.core import analyze as run_analyze

    defaults = get_model_defaults(model)
    if nsteps is None:
        nsteps = defaults.get("steps", 10)
    if resolution is None:
        resolution = defaults.get("resolution")
    calib_resolution = _parse_resolution(resolution) if isinstance(resolution, str) else resolution or (480, 848)
    dtype_list = [d.upper() for d in dtypes]

    # Validate effective_bits against selected formats
    _MIN_BITS = {"NVFP4": 4.5, "FP8": 8.0}
    min_bits = min(_MIN_BITS.get(d, 16.0) for d in dtype_list)
    if effective_bits < min_bits or effective_bits > 16.0:
        raise click.BadParameter(
            f"effective-bits must be between {min_bits} and 16.0 for formats {', '.join(dtype_list)}."
        )

    console.print(f"\n[bold]Model:[/] {model}")
    console.print(f"[bold]Formats:[/] {', '.join(dtype_list)}")
    console.print(f"[bold]Target effective bits:[/] {effective_bits}")
    res_str = f"{calib_resolution[0]}x{calib_resolution[1]}"
    console.print(f"[bold]Pipeline runs:[/] {nprompts} × {nsteps} steps @ {res_str}")
    console.print()

    state_dict = run_analyze(
        model_source=model,
        dtypes=dtype_list,
        effective_bits=effective_bits,
        num_prompts=nprompts,
        num_steps=nsteps,
        resolution=calib_resolution,
        num_frames=defaults.get("num_frames"),
        offload=offload,
    )

    # Display results
    best = state_dict.get("best", {})
    recipe = best.get("recipe", {})
    constraints = best.get("constraints", {})
    is_satisfied = best.get("is_satisfied", False)
    total_score = best.get("score", 0)
    candidate_stats = state_dict.get("candidate_stats", {})

    # Per-layer table
    table = Table(title="Per-Layer Format Selection", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Layer")
    table.add_column("Selected Format", style="cyan")
    table.add_column("Score", justify="right")

    format_counts: dict[str, int] = {}
    for i, (layer_name, selected) in enumerate(sorted(recipe.items())):
        fmt_str = str(selected)
        format_counts[fmt_str] = format_counts.get(fmt_str, 0) + 1

        # Get score for this layer's selected format
        layer_stats = candidate_stats.get(layer_name, {})
        scores = layer_stats.get("scores", [])
        formats = layer_stats.get("formats", [])
        score_str = ""
        for fmt, score in zip(formats, scores):
            if str(fmt) == fmt_str:
                score_str = f"{score:.4f}"
                break

        table.add_row(str(i + 1), layer_name, fmt_str, score_str)

    console.print(table)

    # Summary
    summary = Table(title="Summary", title_style="bold")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value")

    actual_bits = constraints.get("effective_bits", "?")
    summary.add_row("Target effective bits", f"{effective_bits}")
    summary.add_row("Actual effective bits", f"{actual_bits}")
    summary.add_row("Constraint satisfied", "[green]Yes[/]" if is_satisfied else "[red]No[/]")
    summary.add_row("Total sensitivity score", f"{total_score:.4f}")

    for fmt, count in sorted(format_counts.items()):
        summary.add_row(f"Layers → {fmt}", str(count))

    console.print(summary)


@cli.command()
@click.argument("model", metavar="MODEL_ID_OR_PATH")
@click.option("--show-all", is_flag=True, help="Show all components, including non-quantizable ones.")
def components(model: str, show_all: bool) -> None:
    """Show components of a generative model.

    MODEL is a HuggingFace diffusers model ID (e.g. Wan-AI/Wan2.2-I2V-A14B-Diffusers)
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


@cli.command("info")
def info() -> None:
    """Show supported formats and default keep presets."""
    table = Table(title="Quantization Formats", title_style="bold")
    table.add_column("Format", style="cyan")
    table.add_column("Description")

    descriptions = {
        "FP8": "8-bit floating point (E4M3). Requires Hopper+ GPU.",
        "NVFP4": "NVIDIA 4-bit floating point. Requires Blackwell+ GPU.",
    }

    for dtype in DTYPES:
        table.add_row(dtype, descriptions.get(dtype, ""))

    console.print(table)

    keeps_table = Table(title="Default Keep Presets", title_style="bold")
    keeps_table.add_column("Preset", style="cyan")
    keeps_table.add_column("Patterns")

    for name in DEFAULT_KEEPS_NAMES:
        patterns = ", ".join(DEFAULT_KEEPS[name])
        keeps_table.add_row(name, patterns)

    console.print(keeps_table)

    from kuantala.config import MODEL_DEFAULTS

    models_table = Table(title="Known Models", title_style="bold")
    models_table.add_column("Model ID", style="cyan")
    models_table.add_column("Keep Preset")
    models_table.add_column("Prompt Source")
    models_table.add_column("Resolution")
    models_table.add_column("Steps")

    for model_id in sorted(MODEL_DEFAULTS):
        defaults = MODEL_DEFAULTS[model_id]
        h, w = defaults.get("resolution", (0, 0))
        models_table.add_row(
            model_id,
            defaults.get("keeps", ""),
            defaults.get("psrc", ""),
            f"{h}x{w}" if h else "",
            str(defaults.get("steps", "")),
        )

    console.print(models_table)


# Approximate bits per parameter for size estimation
_BITS_PER_PARAM = {
    "FP8": 8.5,
    "NVFP4": 4.5,
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

    fp16_bytes = total_params * 16.0 / 8
    for dtype in DTYPES:
        bpp = _BITS_PER_PARAM.get(dtype, 16.0)
        size_bytes = total_params * bpp / 8
        pct = size_bytes / fp16_bytes * 100
        table.add_row(dtype, _fmt(size_bytes), f"{pct:.0f}%")

    console.print(table)


@cli.command()
@click.argument("model", metavar="MODEL_ID_OR_PATH")
def config(model: str) -> None:
    """Show the architecture of a generative model from its config.

    Loads model configs (no weights) to show the full module hierarchy.
    Requires torch and diffusers/transformers to be installed.

    MODEL is a HuggingFace diffusers model ID (e.g. Wan-AI/Wan2.2-I2V-A14B-Diffusers)
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
@click.option("--remap-keys", type=str, default=None,
              help="Remap key names: preset name ('wan') or path to a file with 'pattern replacement' lines.")
def convert(input_file: Path, output: Path | None, remap_keys: str | None) -> None:
    """Convert a kuantala NVFP4 safetensors file to ComfyUI format."""
    if input_file.suffix != ".safetensors":
        raise click.ClickException(f"Input must be a .safetensors file, got: {input_file.suffix}")

    if output is None:
        output = input_file.parent / f"{input_file.stem}-comfyui.safetensors"

    from kuantala.convert import convert_to_comfyui

    console.print(f"[bold]Input:[/]  {input_file}")
    console.print(f"[bold]Output:[/] {output}")
    if remap_keys:
        console.print(f"[bold]Remap:[/]  {remap_keys}")

    # Count quantized layers for summary
    from safetensors.torch import load_file
    state_dict = load_file(str(input_file))
    n_layers = sum(1 for k in state_dict if k.endswith(".weight_quantizer._scale"))

    if n_layers == 0:
        raise click.ClickException("No NVFP4-quantized layers found in input file.")

    convert_to_comfyui(input_file, output, remap_keys=remap_keys)

    input_mb = input_file.stat().st_size / (1024 * 1024)
    output_mb = output.stat().st_size / (1024 * 1024)
    console.print(f"\n[bold green]Converted {n_layers} layers[/]")
    console.print(f"  Input:  {input_mb:.1f} MB")
    console.print(f"  Output: {output_mb:.1f} MB")


@cli.command("eval", hidden=False)
@click.argument("model", metavar="MODEL_ID_OR_PATH")
@click.option("-q", "--quantized-dir", required=True, type=click.Path(exists=True, path_type=Path),
              help="Directory with quantized safetensors from 'kuantala quantize'.")
@click.option("--prompts", type=click.Path(exists=True, path_type=Path), default=None,
              help="File with eval prompts, one per line (default: HF dataset test split).")
@click.option("--nprompts", type=int, default=16,
              help="Number of eval prompts (default: 16).")
@click.option("--nsteps", type=int, default=None,
              help="Number of inference steps (default: auto or 30).")
@click.option("--resolution", type=str, default=None,
              help="Resolution: 480p, 540p, 720p, 1080p, 4k, or HEIGHTxWIDTH (default: auto or 480p).")
@click.option("--decode", is_flag=True,
              help="Also compare decoded pixel-space outputs (default: latent only).")
@click.option("--psrc", type=click.Choice(PROMPT_SOURCES, case_sensitive=False), default=None,
              help="Prompt source: t2i, t2v, i2v (auto-detected for known HF model IDs).")
@click.option("--offset", type=int, default=1024,
              help="Dataset offset for eval prompts to avoid overlap with calibration (default: 1024).")
@click.option("--offload", type=click.Choice(["model", "layers"], case_sensitive=False), default=None,
              help="CPU offload mode: 'model' (component-level) or 'layers' (layer-level, slower but less VRAM).")
def eval_cmd(
    model: str,
    quantized_dir: Path,
    prompts: Path | None,
    nprompts: int,
    nsteps: int | None,
    resolution: str | None,
    decode: bool,
    psrc: str | None,
    offset: int,
    offload: str | None,
) -> None:
    """Evaluate quantization quality by comparing original vs quantized outputs.

    Runs the pipeline with fixed seeds on both original and quantized models,
    then computes PSNR and SSIM metrics.

    MODEL is a HuggingFace diffusers model ID or local directory path.
    """
    from kuantala.config import get_model_defaults
    from kuantala.core import evaluate

    defaults = get_model_defaults(model)
    if nsteps is None:
        nsteps = defaults.get("steps", 30)
    if resolution is None:
        resolution = defaults.get("resolution")
    eval_resolution = _parse_resolution(resolution) if isinstance(resolution, str) else resolution or (480, 848)

    prompt_list = None
    if prompts is not None:
        prompt_list = [line.strip() for line in prompts.read_text().splitlines() if line.strip()]

    console.print(f"\n[bold]Model:[/] {model}")
    console.print(f"[bold]Quantized dir:[/] {quantized_dir}")
    console.print(f"[bold]Prompts:[/] {nprompts} ({'custom' if prompts else 'HF dataset test split'})")
    console.print(f"[bold]Steps:[/] {nsteps}")
    console.print(f"[bold]Resolution:[/] {eval_resolution[0]}x{eval_resolution[1]}")
    console.print(f"[bold]Decode:[/] {'yes' if decode else 'no (latent only)'}")
    console.print()

    results = evaluate(
        model_source=model,
        quantized_dir=quantized_dir,
        num_prompts=nprompts,
        num_steps=nsteps,
        resolution=eval_resolution,
        decode=decode,
        custom_prompts=prompt_list,
        prompt_source=psrc,
        num_frames=defaults.get("num_frames"),
        offset=offset,
        offload=offload,
    )

    _display_eval_results(results, decode)

    # Write eval.md
    md_path = quantized_dir / "eval.md"
    md_path.write_text(_generate_eval_markdown(results, decode))
    console.print(f"\n[bold green]Saved eval results to {md_path}[/]")


# Register 'evaluate' as an alias for 'eval'
cli.add_command(eval_cmd, name="evaluate")


def _display_eval_results(results: dict, decode: bool) -> None:
    """Display eval results as Rich tables."""
    for label, comp_data in results.get("components", {}).items():
        comp_metrics = comp_data["metrics"]

        # Per-prompt table
        table = Table(title=f"Eval: {label}", title_style="bold")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Prompt", max_width=40, no_wrap=True)
        table.add_column("Seed", justify="right")
        table.add_column("Latent PSNR", justify="right", style="cyan")
        table.add_column("Latent SSIM", justify="right", style="cyan")
        if decode:
            table.add_column("Pixel PSNR", justify="right", style="green")
            table.add_column("Pixel SSIM", justify="right", style="green")

        latent_psnrs: list[float] = []
        latent_ssims: list[float] = []
        pixel_psnrs: list[float] = []
        pixel_ssims: list[float] = []

        for i, pm in enumerate(comp_metrics):
            prompt_text = pm["prompt"]
            if len(prompt_text) > 37:
                prompt_text = prompt_text[:37] + "..."

            # Aggregate latent frame metrics
            latent_frames = pm.get("latent_frames", [])
            if latent_frames:
                avg_psnr = sum(f["psnr"] for f in latent_frames) / len(latent_frames)
                avg_ssim = sum(f["ssim"] for f in latent_frames) / len(latent_frames)
                latent_psnrs.append(avg_psnr)
                latent_ssims.append(avg_ssim)
                psnr_str = f"{avg_psnr:.2f} dB"
                ssim_str = f"{avg_ssim:.4f}"
            else:
                psnr_str = "-"
                ssim_str = "-"

            row = [str(i + 1), prompt_text, str(pm["seed"]), psnr_str, ssim_str]

            if decode:
                decoded_frames = pm.get("decoded_frames", [])
                if decoded_frames:
                    avg_p = sum(f["psnr"] for f in decoded_frames) / len(decoded_frames)
                    avg_s = sum(f["ssim"] for f in decoded_frames) / len(decoded_frames)
                    pixel_psnrs.append(avg_p)
                    pixel_ssims.append(avg_s)
                    row.extend([f"{avg_p:.2f} dB", f"{avg_s:.4f}"])
                else:
                    row.extend(["-", "-"])

            table.add_row(*row)

            # Show per-frame breakdown for video (multiple frames)
            if latent_frames and len(latent_frames) > 1:
                for fi, frame in enumerate(latent_frames):
                    frame_row = ["", f"  frame {fi}", "", f"{frame['psnr']:.2f}", f"{frame['ssim']:.4f}"]
                    if decode:
                        decoded_frames = pm.get("decoded_frames", [])
                        if fi < len(decoded_frames):
                            frame_row.extend([f"{decoded_frames[fi]['psnr']:.2f}", f"{decoded_frames[fi]['ssim']:.4f}"])
                        else:
                            frame_row.extend(["", ""])
                    table.add_row(*frame_row)

        console.print(table)

        # Summary table
        summary = Table(title=f"Summary: {label}", title_style="bold")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value")

        summary.add_row("Component", comp_data["component"])
        summary.add_row("Dtype", comp_data["dtype"])
        summary.add_row("Prompts", str(len(comp_metrics)))

        if latent_psnrs:
            summary.add_row("Latent PSNR (avg)", f"{sum(latent_psnrs) / len(latent_psnrs):.2f} dB")
        if latent_ssims:
            summary.add_row("Latent SSIM (avg)", f"{sum(latent_ssims) / len(latent_ssims):.4f}")
        if decode and pixel_psnrs:
            summary.add_row("Pixel PSNR (avg)", f"{sum(pixel_psnrs) / len(pixel_psnrs):.2f} dB")
        if decode and pixel_ssims:
            summary.add_row("Pixel SSIM (avg)", f"{sum(pixel_ssims) / len(pixel_ssims):.4f}")

        console.print(summary)


def _resolve_prompt_source(config: object) -> str | None:
    """Resolve the prompt source for a QuantConfig, or None for custom prompts."""
    from kuantala.config import QuantConfig, detect_prompt_source
    assert isinstance(config, QuantConfig)
    if config.calib_prompts is not None:
        return None
    return config.prompt_source or detect_prompt_source(config.model_source) or "t2i"


def _prompt_source_markdown(source: str | None) -> str:
    """Return a markdown-formatted prompt source string."""
    from kuantala.core import _PROMPT_DATASETS
    if source is None:
        return "custom file"
    ds = _PROMPT_DATASETS.get(source)
    if ds is None:
        return source
    name = ds["name"]
    return f"[{name}](https://huggingface.co/datasets/{name})"


def _generate_quantize_markdown(config: object, output_files: list[Path]) -> str:
    """Generate a markdown summary of the quantization run."""
    from kuantala.config import QuantConfig
    assert isinstance(config, QuantConfig)

    h, w = config.calib_resolution
    lines = [
        "## Quantization Details",
        "",
        f"Quantized with [kuantala](https://github.com/kuantala/kuantala) using [NVIDIA Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer).",
        "",
        "### Configuration",
        "",
        "| | |",
        "|---|---|",
        f"| Original model | `{config.model_source}` |",
        f"| Quantization dtype | {config.dtype} |",
        f"| Algorithm | {config.algorithm} |",
        f"| Calibration prompts | {config.calib_size} from {_prompt_source_markdown(_resolve_prompt_source(config))} |",
        f"| Calibration steps | {config.calib_steps} |",
        f"| Calibration resolution | {h}x{w} |",
    ]

    if config.vae_dtype != "skip":
        lines.append(f"| VAE dtype | {config.vae_dtype} |")
    if config.te_dtype != "skip":
        lines.append(f"| Text encoder dtype | {config.te_dtype} |")
    if config.ie_dtype != "skip":
        lines.append(f"| Image encoder dtype | {config.ie_dtype} |")
    if config.default_keeps:
        lines.append(f"| Default keeps | {config.default_keeps} |")
    if config.keep:
        lines.append(f"| Custom keeps | {', '.join(f'`{k}`' for k in config.keep)} |")

    lines.append("")
    lines.append("### Output Files")
    lines.append("")
    lines.append("| File | Size |")
    lines.append("|------|-----:|")
    for f in output_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        if size_mb >= 1024:
            size_str = f"{size_mb / 1024:.1f} GB"
        else:
            size_str = f"{size_mb:.1f} MB"
        lines.append(f"| `{f.name}` | {size_str} |")

    lines.append("")
    return "\n".join(lines)


def _generate_eval_markdown(results: dict, decode: bool) -> str:
    """Generate a markdown snippet with eval results for model cards."""
    cfg = results.get("config", {})
    h, w = cfg.get("resolution", (0, 0))
    lines = [
        "## Evaluation Results",
        "",
        f"Quantization quality evaluated against the original model using [kuantala](https://github.com/kuantala/kuantala).",
        "",
        "### Setup",
        "",
        f"| | |",
        f"|---|---|",
        f"| Original model | `{cfg.get('model_source', '')}` |",
        f"| Eval prompts | {cfg.get('num_prompts', '')} from {_prompt_source_markdown(cfg.get('prompt_source'))} |",
        f"| Inference steps | {cfg.get('num_steps', '')} |",
        f"| Resolution | {h}x{w} |",
        f"| Comparison | {'latent + pixel' if decode else 'latent space'} |",
        "",
    ]

    for label, comp_data in results.get("components", {}).items():
        comp_metrics = comp_data["metrics"]
        lines.append(f"### {label}")
        lines.append("")

        # Build per-prompt table
        header = "| # | Prompt | Seed | Latent PSNR | Latent SSIM |"
        sep = "|--:|--------|-----:|------------:|------------:|"
        if decode:
            header += " Pixel PSNR | Pixel SSIM |"
            sep += "------------:|------------:|"
        lines.append(header)
        lines.append(sep)

        latent_psnrs: list[float] = []
        latent_ssims: list[float] = []
        pixel_psnrs: list[float] = []
        pixel_ssims: list[float] = []

        for i, pm in enumerate(comp_metrics):
            prompt_text = pm["prompt"]
            if len(prompt_text) > 40:
                prompt_text = prompt_text[:40] + "..."

            latent_frames = pm.get("latent_frames", [])
            if latent_frames:
                avg_psnr = sum(f["psnr"] for f in latent_frames) / len(latent_frames)
                avg_ssim = sum(f["ssim"] for f in latent_frames) / len(latent_frames)
                latent_psnrs.append(avg_psnr)
                latent_ssims.append(avg_ssim)
                row = f"| {i+1} | {prompt_text} | {pm['seed']} | {avg_psnr:.2f} dB | {avg_ssim:.4f} |"
            else:
                row = f"| {i+1} | {prompt_text} | {pm['seed']} | - | - |"

            if decode:
                decoded_frames = pm.get("decoded_frames", [])
                if decoded_frames:
                    avg_p = sum(f["psnr"] for f in decoded_frames) / len(decoded_frames)
                    avg_s = sum(f["ssim"] for f in decoded_frames) / len(decoded_frames)
                    pixel_psnrs.append(avg_p)
                    pixel_ssims.append(avg_s)
                    row += f" {avg_p:.2f} dB | {avg_s:.4f} |"
                else:
                    row += " - | - |"

            lines.append(row)

        # Summary
        lines.append("")
        lines.append("**Summary**")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")

        if latent_psnrs:
            lines.append(f"| Latent PSNR (avg) | {sum(latent_psnrs) / len(latent_psnrs):.2f} dB |")
        if latent_ssims:
            lines.append(f"| Latent SSIM (avg) | {sum(latent_ssims) / len(latent_ssims):.4f} |")
        if decode and pixel_psnrs:
            lines.append(f"| Pixel PSNR (avg) | {sum(pixel_psnrs) / len(pixel_psnrs):.2f} dB |")
        if decode and pixel_ssims:
            lines.append(f"| Pixel SSIM (avg) | {sum(pixel_ssims) / len(pixel_ssims):.4f} |")

        lines.append("")

    return "\n".join(lines)


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

    metadata = header.pop("__metadata__", {})
    if metadata:
        for k, v in metadata.items():
            console.print(f"[bold]{k}:[/] {v}")

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
