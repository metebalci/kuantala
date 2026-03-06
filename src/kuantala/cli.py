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
@click.option("--ie-dtype", type=click.Choice(COMPONENT_DTYPES, case_sensitive=False),
              default=None, help="Image encoder quantization dtype (default: same as --dtype).")
@click.option("--no-heuristics", is_flag=True,
              help="Disable heuristic-based mixed precision (on by default).")
@click.option("--statistics", type=click.Choice(["low", "medium", "high"], case_sensitive=False),
              default=None, help="Preserve statistically sensitive layers (low/medium/high).")
@click.option("--no-calibration", is_flag=True,
              help="Disable calibration forward passes (on by default for NVIDIA backend).")
@click.option("--calibration-data", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to calibration data directory.")
@click.option("--keep", multiple=True,
              help="Manual layer override: 'pattern:dtype' (repeatable).")
def quantize(
    model: str,
    dtype: str,
    output: Path,
    vae_dtype: str,
    te_dtype: str | None,
    ie_dtype: str,
    no_heuristics: bool,
    statistics: str | None,
    no_calibration: bool,
    calibration_data: Path | None,
    keep: tuple[str, ...],
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
    if ie_dtype:
        ie_dtype = ie_dtype.upper() if ie_dtype.lower() != "skip" else ie_dtype

    config = QuantConfig(
        model_source=model,
        dtype=dtype,
        output_dir=output,
        vae_dtype=vae_dtype,
        te_dtype=te_dtype,
        ie_dtype=ie_dtype,
        heuristics=not no_heuristics,
        statistics=statistics.lower() if statistics else None,
        calibration=not no_calibration,
        calibration_data=calibration_data,
        keep=list(keep),
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
@click.option("--show-all", is_flag=True, help="Show all components, including non-quantizable ones.")
def components(model: str, show_all: bool) -> None:
    """Show components of a diffusion model.

    MODEL is a HuggingFace diffusers model ID (e.g. Wan-AI/Wan2.1-I2V-14B-Diffusers)
    or a local directory path in diffusers format (with model_index.json).
    """
    import json as _json
    from pathlib import Path

    local = Path(model)
    is_local = local.is_dir()

    if is_local:
        model_dir = local
        index_path = model_dir / "model_index.json"
        if not index_path.exists():
            raise click.ClickException(
                f"No model_index.json found in {model_dir}. "
                "The model directory must follow the HuggingFace diffusers layout."
            )
    else:
        # Remote: download only model_index.json
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise click.ClickException(
                f"'{model}' is not a local directory and huggingface-hub is not installed. "
                "Install with: pip install kuantala[hub]"
            )
        try:
            index_path = Path(hf_hub_download(repo_id=model, filename="model_index.json", token=None))
        except Exception:
            raise click.ClickException(
                f"'{model}' does not contain a model_index.json on HuggingFace Hub. "
                "Kuantala requires a diffusers-format model."
            )
        model_dir = None

    with open(index_path) as f:
        index = _json.load(f)

    model_type = index.get("_class_name")

    console.print(f"\n[bold]Model:[/] {model}")
    if model_dir:
        console.print(f"[bold]Path:[/] {model_dir}")
    if model_type:
        console.print(f"[bold]Pipeline:[/] {model_type}")

    table = Table(title="Components", title_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Class")
    table.add_column("Parameters", justify="right")
    table.add_column("Dtype", justify="right")

    from kuantala.components import _classify_component

    # For remote repos, list files to find safetensors per component
    repo_files: list[str] = []
    if not is_local:
        try:
            from huggingface_hub import HfApi, RepoFile
            api = HfApi()
            repo_files = [
                f.rfilename for f in api.list_repo_tree(model, token=None, recursive=True)
                if isinstance(f, RepoFile)
            ]
        except Exception:
            pass

    _quantizable_types = {"transformer", "unet", "vae", "text_encoder", "image_encoder"}

    # Collect component info, with progress for remote fetches
    rows: list[tuple[str, str, str, int, set]] = []

    # Count total safetensors files for progress
    components_to_show: list[tuple[str, str | None, str | None, str]] = []
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
        if not show_all and comp_type not in _quantizable_types:
            continue
        components_to_show.append((key, library, class_name, comp_type))

    with console.status("Fetching model info...") as status:
        for key, library, class_name, comp_type in components_to_show:
            class_label = f"{library}.{class_name}" if library and class_name else ""

            # Collect safetensors headers (local or remote)
            headers_list: list[dict] = []
            if is_local:
                comp_dir = model_dir / key
                if comp_dir.is_dir():
                    for sf_path in sorted(comp_dir.glob("*.safetensors")):
                        headers_list.append(_read_local_safetensors_header(sf_path))
            else:
                comp_sf_files = [f for f in repo_files if f.startswith(f"{key}/") and f.endswith(".safetensors")]
                for sf_file in comp_sf_files:
                    status.update(f"Fetching {sf_file}...")
                    header = _fetch_remote_safetensors_header(model, sf_file)
                    if header:
                        headers_list.append(header)

            # Compute params and dtypes from headers
            total_params = 0
            dtypes: set[str] = set()
            for header in headers_list:
                for tname, meta in header.items():
                    if tname == "__metadata__":
                        continue
                    dtypes.add(meta.get("dtype", "unknown"))
                    param_count = 1
                    for dim in meta.get("shape", []):
                        param_count *= dim
                    total_params += param_count

            rows.append((key, comp_type, class_label, total_params, dtypes))

    for key, comp_type, class_label, total_params, dtypes in rows:
        if total_params >= 1_000_000_000:
            params_str = f"{total_params / 1_000_000_000:.1f}B"
        elif total_params >= 1_000_000:
            params_str = f"{total_params / 1_000_000:.0f}M"
        elif total_params > 0:
            params_str = f"{total_params / 1_000:.0f}K"
        else:
            params_str = ""

        dtype_str = ", ".join(sorted(dtypes))
        table.add_row(key, comp_type, class_label, params_str, dtype_str)

    console.print(table)


def _read_local_safetensors_header(sf_path: Path) -> dict:
    """Read the JSON header from a local safetensors file."""
    import json as _json

    with open(sf_path, "rb") as fh:
        header_size = int.from_bytes(fh.read(8), "little")
        return _json.loads(fh.read(header_size))


def _fetch_remote_safetensors_header(repo_id: str, filename: str) -> dict | None:
    """Fetch safetensors header from HF Hub using HTTP range requests.

    Authentication is handled automatically by huggingface-hub
    (via ``hf auth login`` or the ``HF_TOKEN`` environment variable).
    """
    import json as _json
    import struct

    try:
        from huggingface_hub import hf_hub_url, get_session, utils
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        session = get_session()
        headers = utils.build_hf_headers()
        # Read header size (first 8 bytes)
        headers["Range"] = "bytes=0-7"
        resp = session.get(url, headers=headers)
        resp.raise_for_status()
        header_size = struct.unpack("<Q", resp.content[:8])[0]
        # Read the JSON header
        headers["Range"] = f"bytes=8-{8 + header_size - 1}"
        resp = session.get(url, headers=headers)
        resp.raise_for_status()
        return _json.loads(resp.content)
    except Exception as e:
        from kuantala.utils import get_logger
        get_logger(__name__).debug("Failed to fetch header for %s: %s", filename, e)
        return None


@cli.command("formats")
def list_formats() -> None:
    """List available quantization formats."""
    from kuantala.config import GGUF_TYPES, NVIDIA_TYPES

    table = Table(title="Available Quantization Formats", title_style="bold")
    table.add_column("Format", style="cyan")
    table.add_column("Backend", style="green")
    table.add_column("Description")

    descriptions = {
        "Q2_K": "2-bit K-means quantization",
        "Q3_K": "3-bit K-means quantization",
        "Q4_0": "4-bit basic quantization",
        "Q4_K": "4-bit K-means quantization",
        "Q5_0": "5-bit basic quantization",
        "Q5_K": "5-bit K-means quantization",
        "Q6_K": "6-bit K-means quantization",
        "Q8_0": "8-bit basic quantization",
        "MXFP8": "Microscaling FP8 (requires Hopper+)",
        "NVFP4": "NVIDIA FP4 (requires Blackwell)",
    }

    for dtype in GGUF_TYPES:
        table.add_row(dtype, "GGUF", descriptions.get(dtype, ""))
    for dtype in NVIDIA_TYPES:
        table.add_row(dtype, "NVIDIA", descriptions.get(dtype, ""))

    console.print(table)


# Approximate bits per parameter for each quantization type (including overhead)
_BITS_PER_PARAM = {
    "Q2_K": 2.6,
    "Q3_K": 3.4,
    "Q4_0": 4.5,
    "Q4_K": 4.5,
    "Q5_0": 5.5,
    "Q5_K": 5.5,
    "Q6_K": 6.6,
    "Q8_0": 8.5,
    "MXFP8": 8.0,
    "NVFP4": 4.0,
    "F16": 16.0,
    "BF16": 16.0,
    "F32": 32.0,
}


@cli.command()
@click.argument("model", metavar="MODEL_ID_OR_PATH")
def estimate(model: str) -> None:
    """Estimate output sizes for common quantization formats.

    Shows a table of estimated file sizes for key GGUF types (Q4_K, Q5_K,
    Q6_K, Q8_0) and NVIDIA types (MXFP8, NVFP4) if torch + modelopt are
    installed. Heuristics are always assumed on (default). VAE is skipped.

    MODEL is a HuggingFace diffusers model ID or local directory path.
    """
    from kuantala.components import detect_components
    from kuantala.config import NVIDIA_TYPES
    from kuantala.mixed import _HEURISTIC_PATTERNS, compute_statistics_overrides
    from kuantala.model_loader import resolve_model_path

    import fnmatch

    # Show key GGUF types + NVIDIA types (if available)
    estimate_dtypes = ["Q4_K", "Q5_K", "Q6_K", "Q8_0"]
    try:
        import torch  # noqa: F401
        import modelopt  # noqa: F401
        estimate_dtypes += NVIDIA_TYPES
    except ImportError:
        pass

    model_dir = resolve_model_path(model)
    model_info = detect_components(model_dir)

    _quantizable_types = {"transformer", "unet", "text_encoder", "image_encoder"}

    # Gather per-component param counts and tensor names
    components_data: list[dict] = []
    for comp in model_info.components:
        if comp.component_type not in _quantizable_types:
            continue

        sf_files = sorted(comp.path.glob("*.safetensors"))
        if not sf_files:
            continue

        # Read headers to get param counts and tensor names
        total_params = 0
        tensor_info: dict[str, int] = {}  # name -> param_count
        for sf_path in sf_files:
            header = _read_local_safetensors_header(sf_path)
            for tname, meta in header.items():
                if tname == "__metadata__":
                    continue
                param_count = 1
                for dim in meta.get("shape", []):
                    param_count *= dim
                tensor_info[tname] = param_count
                total_params += param_count

        # Count heuristic-preserved params
        heuristic_params = 0
        for tname, pc in tensor_info.items():
            for pattern in _HEURISTIC_PATTERNS:
                if fnmatch.fnmatch(tname.lower(), pattern.lower()):
                    heuristic_params += pc
                    break

        # Run actual statistics analysis for each level
        stats_params = {}
        for level in ("low", "medium", "high"):
            overrides = compute_statistics_overrides(sf_files, level)
            stats_params[level] = sum(tensor_info.get(t, 0) for t in overrides)

        components_data.append({
            "name": comp.name,
            "type": comp.component_type,
            "total_params": total_params,
            "heuristic_params": heuristic_params,
            "stats_params": stats_params,
        })

    if not components_data:
        console.print("[yellow]No quantizable components found.[/]")
        return

    # Print component summary
    total_all = sum(c["total_params"] for c in components_data)
    console.print(f"\n[bold]Model:[/] {model}")
    console.print(f"[bold]Pipeline:[/] {model_info.model_type or 'unknown'}")
    console.print(f"[bold]Total parameters (excl. VAE):[/] {_format_params(total_all)}")
    for c in components_data:
        console.print(f"  {c['name']}: {_format_params(c['total_params'])}")

    def _estimate_size(dtype: str, comp: dict, use_stats: str | None) -> float:
        """Estimate size in bytes for a component at a given dtype."""
        total = comp["total_params"]
        preserved = comp["heuristic_params"]
        if use_stats:
            preserved = max(preserved, preserved + comp["stats_params"].get(use_stats, 0))

        quant_params = max(0, total - preserved)
        bpp_quant = _BITS_PER_PARAM.get(dtype, 8.0)
        bpp_f16 = _BITS_PER_PARAM["F16"]
        return (quant_params * bpp_quant + preserved * bpp_f16) / 8

    def _fmt(b: float) -> str:
        gb = b / (1024 ** 3)
        if gb >= 1.0:
            return f"{gb:.1f} GB"
        return f"{b / (1024 ** 2):.0f} MB"

    # Build size estimate table
    table = Table(title="Estimated Output Sizes", title_style="bold")
    table.add_column("Format", style="cyan")
    table.add_column("Heuristics", justify="right")
    table.add_column("+ Stats Low", justify="right")
    table.add_column("+ Stats Medium", justify="right")
    table.add_column("+ Stats High", justify="right")

    for dtype in estimate_dtypes:
        sizes = {}
        for stats_level in (None, "low", "medium", "high"):
            total_bytes = sum(
                _estimate_size(dtype, comp, stats_level) for comp in components_data
            )
            sizes[stats_level] = total_bytes

        table.add_row(
            dtype,
            _fmt(sizes[None]),
            _fmt(sizes["low"]),
            _fmt(sizes["medium"]),
            _fmt(sizes["high"]),
        )

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
    import json as _json

    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "The config command requires torch and diffusers."
        )

    local = Path(model)
    is_local = local.is_dir()

    if is_local:
        model_dir = local
        index_path = model_dir / "model_index.json"
        if not index_path.exists():
            raise click.ClickException(
                f"No model_index.json found in {model_dir}. "
                "The model directory must follow the HuggingFace diffusers layout."
            )
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise click.ClickException(
                f"'{model}' is not a local directory and huggingface-hub is not installed. "
                "Install with: pip install kuantala[hub]"
            )
        try:
            index_path = Path(hf_hub_download(repo_id=model, filename="model_index.json", token=None))
        except Exception:
            raise click.ClickException(
                f"'{model}' does not contain a model_index.json on HuggingFace Hub. "
                "Kuantala requires a diffusers-format model."
            )
        model_dir = index_path.parent

    with open(index_path) as f:
        index = _json.load(f)

    model_type = index.get("_class_name")
    console.print(f"\n[bold]Model:[/] {model}")
    if model_type:
        console.print(f"[bold]Pipeline:[/] {model_type}")

    from kuantala.components import _classify_component

    _quantizable_types = {"transformer", "unet", "vae", "text_encoder", "image_encoder"}

    for key, value in index.items():
        if key.startswith("_") or value is None or not isinstance(value, list):
            continue
        library = value[0] if len(value) >= 1 else None
        class_name = value[1] if len(value) >= 2 else None
        if library is None and class_name is None:
            continue
        comp_type = _classify_component(key, class_name, library)
        if comp_type not in _quantizable_types:
            continue

        # Download config.json for this component if remote
        config_path = model_dir / key / "config.json"
        if not config_path.exists() and not is_local:
            try:
                from huggingface_hub import hf_hub_download
                config_path = Path(hf_hub_download(
                    repo_id=model, filename=f"{key}/config.json"                ))
            except Exception:
                console.print(f"\n[yellow]Could not fetch config for {key}[/]")
                continue

        if not config_path.exists():
            continue

        # Instantiate model from config on meta device (no memory allocation)
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

        # Count parameters
        total_params = sum(p.numel() for p in model_instance.parameters())

        console.print(f"\n[bold cyan]{key}[/] [dim]({full_class}, {_format_params(total_params)} params)[/]")

        # Print module tree
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

        # Summarize this module
        child_type = type(child).__name__
        param_count = sum(p.numel() for p in child.parameters())
        param_str = f" ({_format_params(param_count)})" if param_count > 0 else ""

        # For leaf-like modules (Linear, Conv, Norm), show shape info
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
@click.argument("file", metavar="FILE_PATH", type=click.Path(exists=True, path_type=Path))
def tensors(file: Path) -> None:
    """Show tensors in a safetensors or GGUF file.

    Shows per-tensor name, dtype, shape, and parameter count.
    """
    if file.suffix == ".gguf":
        _inspect_gguf(file)
    elif file.suffix == ".safetensors":
        _inspect_safetensors(file)
    else:
        raise click.ClickException(f"Unsupported file format: {file.suffix}. Use .safetensors or .gguf")


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
    import json as _json

    with open(file, "rb") as fh:
        header_size = int.from_bytes(fh.read(8), "little")
        header = _json.loads(fh.read(header_size))

    file_size_mb = file.stat().st_size / (1024 * 1024)
    console.print(f"\n[bold]File:[/] {file}")
    console.print(f"[bold]Format:[/] safetensors")
    console.print(f"[bold]Size:[/] {file_size_mb:.1f} MB")

    # Collect tensor info
    dtype_counts: dict[str, int] = {}
    dtype_params: dict[str, int] = {}
    total_params = 0
    tensors: list[tuple[str, str, list[int], int]] = []

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
        tensors.append((name, dtype, shape, param_count))

    _print_layers_and_summary(tensors, dtype_counts, dtype_params, total_params)


def _inspect_gguf(file: Path) -> None:
    """Inspect a GGUF file."""
    try:
        from gguf import GGUFReader
    except ImportError:
        raise click.ClickException("gguf package not installed. Install with: pip install kuantala[gguf]")

    reader = GGUFReader(str(file))

    file_size_mb = file.stat().st_size / (1024 * 1024)
    console.print(f"\n[bold]File:[/] {file}")
    console.print(f"[bold]Format:[/] GGUF")
    console.print(f"[bold]Size:[/] {file_size_mb:.1f} MB")

    # Collect tensor info
    dtype_counts: dict[str, int] = {}
    dtype_params: dict[str, int] = {}
    total_params = 0
    tensors: list[tuple[str, str, list[int], int]] = []

    for tensor in reader.tensors:
        name = tensor.name
        dtype = str(tensor.tensor_type).split(".")[-1]
        shape = list(tensor.shape)
        param_count = 1
        for dim in shape:
            param_count *= dim
        total_params += param_count
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        dtype_params[dtype] = dtype_params.get(dtype, 0) + param_count
        tensors.append((name, dtype, shape, param_count))

    _print_layers_and_summary(tensors, dtype_counts, dtype_params, total_params)


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
    # Sort tensors naturally (numeric-aware) for consistent ordering
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

    # Dtype summary
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
