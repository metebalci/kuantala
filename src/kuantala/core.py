"""Main quantization orchestrator."""

from __future__ import annotations

import gc
import importlib
import inspect
import json as _json
from pathlib import Path
from typing import Any

import torch
import modelopt.torch.quantization as mtq
from safetensors.torch import save_file

from kuantala.components import ModelComponent, detect_components
from kuantala.config import DEFAULT_KEEPS, QuantConfig, detect_default_keeps, detect_prompt_source
from kuantala.model_loader import resolve_model_path
from kuantala.utils import get_logger

log = get_logger(__name__)

# modelopt quantization configs for each dtype
_MODELOPT_CONFIGS = {
    "FP8": mtq.FP8_DEFAULT_CFG,
    "NVFP4": mtq.NVFP4_DEFAULT_CFG,
}

# Configs for auto_quantize (AWQ_LITE recommended for NVFP4)
_AUTO_QUANTIZE_CONFIGS = {
    "FP8": mtq.FP8_DEFAULT_CFG,
    "NVFP4": mtq.NVFP4_AWQ_LITE_CFG,
}

# Prompt datasets per source type.
# Each entry: (hf_id, config_name, calib_split, eval_split, prompt_column, image_column)
# First 10240 entries are used for calibration, second 10240 for eval.
_PROMPT_DATASETS: dict[str, dict[str, Any]] = {
    "t2i": {
        "name": "poloclub/diffusiondb",
        "license": "CC0-1.0",
        "prompt_column": "prompt",
        "image_column": None,
    },
    "t2v": {
        "name": "WenhaoWang/VidProM",
        "license": "CC-BY-NC-4.0",
        "config": "VidProM_unique",
        "prompt_column": "prompt",
        "image_column": None,
    },
    "i2v": {
        "name": "WenhaoWang/TIP-I2V",
        "license": "CC-BY-NC-4.0",
        "prompt_column": "Text_Prompt",
        "image_column": "Image_Prompt",
        "streaming": True,
        "calib_split": "Subset",
        "eval_split": "Eval",
    },
    "ti2i": {
        "name": "UCSC-VLAA/HQ-Edit",
        "license": "CC-BY-NC-4.0",
        "prompt_column": "edit",
        "image_column": "input_image",
        "streaming": True,
    },
}

# First 10240 entries = calibration pool, second 10240 = eval pool
_POOL_SIZE = 10240


def _load_prompts(
    num_prompts: int, source: str, for_eval: bool = False
) -> tuple[list[str], list[Any] | None]:
    """Load prompts (and optionally images) from an HF dataset.

    Returns (prompts, images) where images is None for t2i/t2v.
    Uses first 10240 entries for calibration, second 10240 for eval.
    """
    ds_cfg = _PROMPT_DATASETS[source]
    dataset_name = ds_cfg["name"]
    # For streaming datasets, use a small offset to avoid slow iteration
    pool_size = num_prompts if ds_cfg.get("streaming") else _POOL_SIZE
    offset = pool_size if for_eval else 0
    end = offset + num_prompts
    purpose = "eval" if for_eval else "calibration"

    if source == "t2i":
        # DiffusionDB: load metadata.parquet directly via pyarrow (dataset script is broken)
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        log.info("Loading %s prompts from %s [%d:%d]...", purpose, dataset_name, offset, end)
        path = hf_hub_download(dataset_name, "metadata.parquet", repo_type="dataset")
        table = pq.read_table(path, columns=[ds_cfg["prompt_column"]])
        all_prompts = table.column(ds_cfg["prompt_column"]).to_pylist()
        prompts = all_prompts[offset:end]
        images = None
    elif ds_cfg.get("streaming"):
        # Streaming datasets (e.g. TIP-I2V, HQ-Edit): iterate to collect prompts + images
        from datasets import load_dataset

        if for_eval and "eval_split" in ds_cfg:
            split = ds_cfg["eval_split"]
            stream_offset = 0
        elif not for_eval and "calib_split" in ds_cfg:
            split = ds_cfg["calib_split"]
            stream_offset = offset
        else:
            split = "train"
            stream_offset = offset

        log.info("Loading %s prompts from %s (streaming, split=%s, skip=%d, take=%d)...",
                 purpose, dataset_name, split, stream_offset, num_prompts)
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        prompt_col = ds_cfg["prompt_column"]
        img_col = ds_cfg["image_column"]
        prompts = []
        images = [] if img_col else None
        for i, row in enumerate(dataset):
            if i < stream_offset:
                continue
            if len(prompts) >= num_prompts:
                break
            prompts.append(row[prompt_col])
            if img_col:
                images.append(row[img_col])
    else:
        # VidProM: use datasets library with split slicing
        from datasets import load_dataset

        split = f"train[{offset}:{end}]"
        load_kwargs: dict[str, Any] = {"name": ds_cfg.get("config")}

        log.info("Loading %s prompts from %s (split=%s)...", purpose, dataset_name, split)
        dataset = load_dataset(dataset_name, split=split, **load_kwargs)
        prompts = list(dataset[ds_cfg["prompt_column"]][:num_prompts])

        img_col = ds_cfg["image_column"]
        images = list(dataset[img_col][:num_prompts]) if img_col else None

    log.info("Loaded %d %s prompts%s from %s",
             len(prompts), purpose,
             f" with {len(images)} images" if images else "",
             dataset_name)
    return prompts, images


def _resolve_component_dtype(
    component: ModelComponent, config: QuantConfig
) -> str | None:
    """Determine what dtype to use for a component. Returns None if skip."""
    if component.component_type == "vae":
        dtype = config.vae_dtype
    elif component.component_type == "text_encoder":
        dtype = config.te_dtype
    elif component.component_type == "image_encoder":
        dtype = config.ie_dtype
    else:
        dtype = config.dtype

    if dtype is None:
        dtype = config.dtype

    if dtype == "skip":
        return None

    return dtype


def _load_model(component: ModelComponent) -> torch.nn.Module:
    """Load a model component via diffusers/transformers from_pretrained."""
    if not component.library or not component.class_name:
        raise RuntimeError(
            f"Cannot load component '{component.name}': missing library/class info"
        )

    lib = importlib.import_module(component.library)
    cls = getattr(lib, component.class_name)

    # diffusers uses torch_dtype, transformers uses dtype
    dtype_kwarg = "dtype" if component.library == "transformers" else "torch_dtype"
    model = cls.from_pretrained(str(component.path), **{dtype_kwarg: torch.float16})
    model = model.cuda()
    log.info("Loaded %s.%s on CUDA", component.library, component.class_name)
    return model


def _load_pipeline(model_dir: Path) -> Any:
    """Load the full diffusers pipeline for calibration."""
    from diffusers import DiffusionPipeline

    log.info("Loading pipeline from %s...", model_dir)
    pipe = DiffusionPipeline.from_pretrained(str(model_dir), torch_dtype=torch.float16)
    pipe.to("cuda")
    log.info("Pipeline loaded on CUDA")
    return pipe


def _build_pipeline_kwargs(
    pipe: Any, num_inference_steps: int = 30, resolution: tuple[int, int] = (256, 256),
    has_dataset_images: bool = False,
) -> dict[str, Any]:
    """Build kwargs for pipeline calibration calls based on its signature."""
    params = inspect.signature(pipe.__call__).parameters
    height, width = resolution

    kwargs: dict[str, Any] = {"num_inference_steps": num_inference_steps}

    # Skip VAE decoding for speed
    if "output_type" in params:
        kwargs["output_type"] = "latent"

    # Video models need frame/dimension info
    if "height" in params:
        kwargs["height"] = height
    if "width" in params:
        kwargs["width"] = width
    if "num_frames" in params:
        kwargs["num_frames"] = 9  # satisfies (n-1)%4==0 for Wan

    # I2V models need a conditioning image — use random if no dataset images
    if "image" in params and not has_dataset_images:
        param = params["image"]
        if param.default is inspect.Parameter.empty:
            from PIL import Image
            import numpy as np
            kwargs["image"] = Image.fromarray(
                np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            )

    return kwargs


def _make_pipeline_calibration_fn(
    pipe: Any, prompts: list[str], num_inference_steps: int = 30,
    resolution: tuple[int, int] = (256, 256), images: list[Any] | None = None,
) -> Any:
    """Create a calibration forward_loop that runs the full pipeline."""
    kwargs = _build_pipeline_kwargs(pipe, num_inference_steps, resolution, has_dataset_images=images is not None)

    def forward_loop(model: Any) -> None:
        log.info("Running pipeline calibration with %d prompts...", len(prompts))
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                log.info("  Calibration prompt %d/%d", i + 1, len(prompts))
                try:
                    call_kwargs = dict(kwargs)
                    if images is not None:
                        call_kwargs["image"] = images[i % len(images)]
                    pipe(prompt=prompt, **call_kwargs)
                except Exception as e:
                    log.warning("Pipeline calibration failed for prompt %d: %s", i + 1, e)

    return forward_loop


def _make_random_calibration_fn(model: Any, component: ModelComponent, num_batches: int = 4) -> Any:
    """Create a calibration forward_loop that feeds random data through the model.

    Used for non-transformer components (VAE, text encoder, image encoder) which
    are rarely quantized and don't need the full pipeline.
    """
    import json

    cfg = {}
    cfg_file = component.path / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = json.load(f)

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    def forward_loop(m: Any) -> None:
        log.info("Running calibration with random data (%d batches)...", num_batches)
        with torch.no_grad(), torch.autocast("cuda", dtype=model_dtype):
            for _ in range(num_batches):
                _run_random_forward(m, component.component_type, cfg, device, model_dtype)

    return forward_loop


def _run_random_forward(model: Any, component_type: str, cfg: dict, device: Any, model_dtype: Any) -> None:
    """Run a single forward pass with random inputs appropriate for the component type.

    Used for non-transformer components only.
    """
    if component_type == "text_encoder":
        vocab_size = cfg.get("vocab_size", 32128)
        max_length = cfg.get("max_position_embeddings", 77)
        seq_len = min(max_length, 64)
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
        model(input_ids=input_ids)

    elif component_type == "vae":
        in_channels = cfg.get("in_channels", 3)
        sample_size = cfg.get("sample_size", 256)
        if isinstance(sample_size, list):
            sample_size = sample_size[0]
        s = min(sample_size, 64)

        class_name = cfg.get("_class_name", "")
        if "Wan" in class_name or "Video" in class_name:
            x = torch.randn(1, in_channels, 4, s, s, device=device, dtype=model_dtype)
        else:
            x = torch.randn(1, in_channels, s, s, device=device, dtype=model_dtype)

        try:
            model.encode(x)
        except Exception:
            try:
                model(x)
            except Exception:
                log.warning("Forward pass failed for %s, falling back to parameter activation", component_type)
                for p in model.parameters():
                    _ = p * 1.0

    elif component_type == "image_encoder":
        image_size = cfg.get("image_size", 224)
        num_channels = cfg.get("num_channels", 3)
        pixel_values = torch.randn(1, num_channels, image_size, image_size, device=device, dtype=model_dtype)

        try:
            model(pixel_values=pixel_values)
        except TypeError:
            try:
                model(pixel_values)
            except Exception:
                log.warning("Forward pass failed for %s, falling back to parameter activation", component_type)
                for p in model.parameters():
                    _ = p * 1.0

    else:
        for p in model.parameters():
            _ = p * 1.0


def _disable_kv_cache_plugins():
    """Disable modelopt KV cache quantization plugins.

    Diffusion model components don't use KV cache, and the auto-patching
    fails on some attention classes (e.g. UMT5Attention).
    """
    try:
        from modelopt.torch.quantization.plugins import custom as _custom_plugins
        saved = set(_custom_plugins.CUSTOM_MODEL_PLUGINS)
        _custom_plugins.CUSTOM_MODEL_PLUGINS.clear()
        return saved
    except Exception:
        return None


def _restore_kv_cache_plugins(saved):
    """Restore previously saved KV cache plugins."""
    if saved is not None:
        try:
            from modelopt.torch.quantization.plugins import custom as _custom_plugins
            _custom_plugins.CUSTOM_MODEL_PLUGINS.update(saved)
        except Exception:
            pass


def _disable_quantizers_by_pattern(model: torch.nn.Module, patterns: list[str]) -> None:
    """Disable quantizers on layers matching any of the given glob patterns."""
    import fnmatch

    if not patterns:
        return

    def filter_fn(name: str) -> bool:
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    mtq.disable_quantizer(model, filter_fn)
    log.info("Disabled quantizers on layers matching: %s", patterns)


def _build_metadata(component: ModelComponent, dtype: str, config: QuantConfig) -> dict[str, str]:
    """Build safetensors metadata recording how the file was produced."""
    import json
    meta: dict[str, str] = {
        "quantizer": "kuantala",
        "dtype": dtype,
        "model_source": config.model_source,
        "component": component.name,
    }
    if component.class_name:
        meta["class_name"] = component.class_name
    if component.library:
        meta["library"] = component.library
    if config.default_keeps:
        meta["default_keeps"] = config.default_keeps
    if config.keep:
        meta["keep"] = json.dumps(config.keep)
    return meta


def _resolve_keeps(config: QuantConfig) -> list[str]:
    """Resolve keep patterns: default preset + user overrides."""
    all_keeps = list(config.keep)
    preset = config.default_keeps
    if preset is None and not config.no_default_keeps:
        preset = detect_default_keeps(config.model_source)
    if preset and preset in DEFAULT_KEEPS:
        all_keeps = DEFAULT_KEEPS[preset] + all_keeps
        log.info("Applied default keeps for '%s': %s", preset, DEFAULT_KEEPS[preset])
    return all_keeps


def _quantize_and_save(
    model: torch.nn.Module,
    component: ModelComponent,
    dtype: str,
    config: QuantConfig,
    output_path: Path,
    forward_loop: Any,
) -> Path:
    """Quantize an already-loaded model and save to safetensors."""
    import warnings

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = output_path.with_suffix(".safetensors")
    metadata = _build_metadata(component, dtype, config)

    quant_cfg = {**_MODELOPT_CONFIGS[dtype], "algorithm": config.algorithm}

    saved_plugins = _disable_kv_cache_plugins()
    try:
        log.info("Applying %s quantization (algorithm=%s) via modelopt...", dtype, config.algorithm)
        mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

        all_keeps = _resolve_keeps(config)
        if all_keeps:
            _disable_quantizers_by_pattern(model, all_keeps)

        log.info("Compressing weights to real %s...", dtype)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Real quantization has been applied")
            mtq.compress(model)
    finally:
        _restore_kv_cache_plugins(saved_plugins)

    # Save compressed state dict
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, str(output_file), metadata=metadata)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    log.info("Written %s (%.1f MB)", output_file, file_size_mb)
    return output_file


def quantize(config: QuantConfig) -> list[Path]:
    """Run quantization according to the given config.

    Returns list of output file paths.
    """
    model_dir = resolve_model_path(config.model_source)
    model_info = detect_components(model_dir)
    if not model_info.components:
        raise RuntimeError(f"No quantizable components found in {model_dir}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []

    # Categorize components
    pipeline_components: list[tuple[ModelComponent, str]] = []
    individual_components: list[tuple[ModelComponent, str]] = []

    for component in model_info.components:
        dtype = _resolve_component_dtype(component, config)
        if dtype is None:
            log.info("Skipping %s (dtype=skip)", component.name)
            continue
        if component.component_type in ("tokenizer", "scheduler", "other"):
            log.debug("Skipping non-quantizable component: %s", component.name)
            continue

        if component.component_type in ("transformer", "unet"):
            pipeline_components.append((component, dtype))
        else:
            individual_components.append((component, dtype))

    # Quantize transformer/unet with pipeline-based calibration
    if pipeline_components:
        pipe = _load_pipeline(model_dir)

        if config.calib_prompts:
            prompts = config.calib_prompts[:config.calib_size]
            images = None
        else:
            source = config.prompt_source or detect_prompt_source(config.model_source) or "t2i"
            prompts, images = _load_prompts(config.calib_size, source)

        if len(prompts) < config.calib_size:
            log.warning("Only %d prompts available (requested %d)", len(prompts), config.calib_size)
        forward_loop = _make_pipeline_calibration_fn(
            pipe, prompts, config.calib_steps, config.calib_resolution, images=images
        )

        for component, dtype in pipeline_components:
            log.info("Processing component: %s (%s -> %s)", component.name, component.component_type, dtype)
            model = getattr(pipe, component.name)
            output_path = config.output_dir / f"{component.name}-{dtype}"
            output_file = _quantize_and_save(model, component, dtype, config, output_path, forward_loop)
            output_files.append(output_file)

        del pipe
        torch.cuda.empty_cache()
        gc.collect()

    # Quantize other components individually (VAE, text encoder, etc.)
    for component, dtype in individual_components:
        log.info("Processing component: %s (%s -> %s)", component.name, component.component_type, dtype)
        model = _load_model(component)

        forward_loop = _make_random_calibration_fn(model, component, config.calib_size)

        output_path = config.output_dir / f"{component.name}-{dtype}"
        output_file = _quantize_and_save(model, component, dtype, config, output_path, forward_loop)
        output_files.append(output_file)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    if not output_files:
        log.warning("No components were quantized.")

    return output_files


def analyze(
    model_source: str,
    dtypes: list[str],
    effective_bits: float,
    num_prompts: int,
    num_steps: int,
    resolution: tuple[int, int],
) -> dict:
    """Run auto_quantize to find optimal per-layer format selection.

    Loads the pipeline, captures transformer inputs, then runs modelopt's
    auto_quantize to determine the best format per layer given an effective
    bits constraint. Returns the auto_quantize state_dict.
    """
    model_dir = resolve_model_path(model_source)
    pipe = _load_pipeline(model_dir)

    # Find transformer/unet
    transformer = None
    comp_name = None
    for name in ["transformer", "unet"]:
        if hasattr(pipe, name) and getattr(pipe, name) is not None:
            transformer = getattr(pipe, name)
            comp_name = name
            break
    if transformer is None:
        raise RuntimeError("No transformer or unet found in pipeline")

    log.info("Analyzing component: %s", comp_name)

    # Capture transformer inputs from pipeline runs (stored on CPU to save GPU memory)
    captured_inputs: list[tuple[tuple, dict]] = []

    def capture_hook(module: Any, args: tuple, kwargs: dict) -> None:
        cpu_args = tuple(
            a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in args
        )
        cpu_kwargs = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        captured_inputs.append((cpu_args, cpu_kwargs))

    handle = transformer.register_forward_pre_hook(capture_hook, with_kwargs=True)

    prompts, _ = _load_prompts(num_prompts, "t2i")
    pipe_kwargs = _build_pipeline_kwargs(pipe, num_steps, resolution)

    log.info("Capturing transformer inputs from %d pipeline runs (%d steps each)...", len(prompts), num_steps)
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            log.info("  Pipeline run %d/%d", i + 1, len(prompts))
            pipe(prompt=prompt, **pipe_kwargs)

    handle.remove()
    num_captured = len(captured_inputs)
    log.info("Captured %d transformer forward passes", num_captured)

    # Build format list for auto_quantize
    quant_formats = [_AUTO_QUANTIZE_CONFIGS[d] for d in dtypes if d in _AUTO_QUANTIZE_CONFIGS]
    if not quant_formats:
        raise RuntimeError(f"No valid quantization formats for: {dtypes}")

    device = next(transformer.parameters()).device

    def forward_step(model: Any, batch: Any) -> Any:
        args, kwargs = batch
        gpu_args = tuple(
            a.to(device) if isinstance(a, torch.Tensor) else a for a in args
        )
        gpu_kwargs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return model(*gpu_args, **gpu_kwargs)

    def loss_func(output: Any, batch: Any) -> torch.Tensor:
        if hasattr(output, "sample"):
            out = output.sample
        elif isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        return out.float().pow(2).mean()

    saved_plugins = _disable_kv_cache_plugins()
    try:
        log.info(
            "Running auto_quantize (formats=%s, effective_bits=%.1f, calib=%d, score=%d)...",
            dtypes, effective_bits,
            min(64, num_captured), min(32, num_captured),
        )
        _, state_dict = mtq.auto_quantize(
            transformer,
            constraints={"effective_bits": effective_bits},
            quantization_formats=quant_formats,
            data_loader=captured_inputs,
            forward_step=forward_step,
            loss_func=loss_func,
            num_calib_steps=min(64, num_captured),
            num_score_steps=min(32, num_captured),
        )
    finally:
        _restore_kv_cache_plugins(saved_plugins)

    del pipe, transformer, captured_inputs
    torch.cuda.empty_cache()
    gc.collect()

    return state_dict


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------


def _discover_quantized_files(quantized_dir: Path) -> list[dict[str, Any]]:
    """Scan a directory for quantized safetensors files.

    Returns list of dicts with keys: path, component, dtype, class_name, library.
    Reads safetensors metadata first, falls back to filename parsing.
    """
    results = []
    for sf_path in sorted(quantized_dir.glob("*.safetensors")):
        info: dict[str, Any] = {"path": sf_path}

        # Try reading metadata from the safetensors header
        try:
            with open(sf_path, "rb") as fh:
                header_size = int.from_bytes(fh.read(8), "little")
                header = _json.loads(fh.read(header_size))
            meta = header.get("__metadata__", {})
        except Exception:
            meta = {}

        if meta.get("quantizer") == "kuantala":
            info["component"] = meta.get("component", sf_path.stem)
            info["dtype"] = meta.get("dtype", "unknown")
            info["class_name"] = meta.get("class_name")
            info["library"] = meta.get("library")
        else:
            # Fallback: parse filename like "transformer-FP8.safetensors"
            stem = sf_path.stem
            parts = stem.rsplit("-", 1)
            if len(parts) == 2:
                info["component"] = parts[0]
                info["dtype"] = parts[1].upper()
            else:
                info["component"] = stem
                info["dtype"] = "unknown"
            info["class_name"] = None
            info["library"] = None

        results.append(info)
        log.info("Discovered quantized file: %s (component=%s, dtype=%s)",
                 sf_path.name, info["component"], info["dtype"])

    return results


def _load_quantized_component(
    component_path: Path,
    quantized_file: Path,
    dtype: str,
    library: str | None,
    class_name: str | None,
) -> torch.nn.Module:
    """Load a quantized component by reconstructing the modelopt structure and loading weights.

    For passthrough dtypes (FP16, BF16): loads state_dict directly.
    For quantized dtypes (FP8, NVFP4): uses mtq.quantize + mtq.compress + load_state_dict.
    """
    import warnings
    from safetensors.torch import load_file

    if not library or not class_name:
        raise RuntimeError(
            f"Cannot load quantized component from {quantized_file}: "
            "missing library/class_name in metadata"
        )

    lib = importlib.import_module(library)
    cls = getattr(lib, class_name)

    dtype_kwarg = "dtype" if library == "transformers" else "torch_dtype"
    model = cls.from_pretrained(str(component_path), **{dtype_kwarg: torch.float16})
    model = model.cuda()

    saved_sd = load_file(str(quantized_file))

    quant_cfg = _MODELOPT_CONFIGS[dtype]
    saved_plugins = _disable_kv_cache_plugins()
    try:
        mtq.quantize(model, quant_cfg, forward_loop=None)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Real quantization has been applied")
            mtq.compress(model)
    finally:
        _restore_kv_cache_plugins(saved_plugins)
    model.load_state_dict(saved_sd, strict=False)

    log.info("Loaded quantized %s (%s) from %s", class_name, dtype, quantized_file.name)
    return model


def evaluate(
    model_source: str,
    quantized_dir: Path,
    num_prompts: int = 16,
    num_steps: int = 30,
    resolution: tuple[int, int] = (480, 848),
    decode: bool = False,
    custom_prompts: list[str] | None = None,
    prompt_source: str | None = None,
) -> dict[str, Any]:
    """Compare original vs quantized pipeline outputs.

    Returns structured results dict with per-component metrics.
    """
    from kuantala.metrics import compute_metrics_per_frame

    model_dir = resolve_model_path(model_source)

    # Load prompts
    if custom_prompts:
        prompts = custom_prompts[:num_prompts]
        dataset_images = None
    else:
        source = prompt_source or detect_prompt_source(model_source) or "t2i"
        prompts, dataset_images = _load_prompts(num_prompts, source, for_eval=True)
    if len(prompts) < num_prompts:
        log.warning("Only %d prompts available (requested %d)", len(prompts), num_prompts)
    prompts = prompts[:num_prompts]

    # Discover quantized files
    quant_files = _discover_quantized_files(quantized_dir)
    if not quant_files:
        raise RuntimeError(f"No quantized safetensors files found in {quantized_dir}")

    # Load original pipeline
    log.info("Loading original pipeline for reference outputs...")
    pipe = _load_pipeline(model_dir)
    pipe_kwargs = _build_pipeline_kwargs(
        pipe, num_steps, resolution, has_dataset_images=dataset_images is not None
    )

    # If not decoding, force latent output
    if not decode:
        pipe_kwargs["output_type"] = "latent"

    def _call_kwargs(i: int) -> dict[str, Any]:
        """Build per-prompt kwargs, injecting dataset image if available."""
        kw = dict(pipe_kwargs)
        if dataset_images is not None:
            kw["image"] = dataset_images[i % len(dataset_images)]
        return kw

    # Generate reference outputs with fixed seeds
    log.info("Generating %d reference outputs...", len(prompts))
    ref_latents: list[torch.Tensor] = []
    ref_decoded: list[torch.Tensor] = []

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            gen = torch.Generator(device="cuda").manual_seed(i)
            log.info("  Reference prompt %d/%d", i + 1, len(prompts))

            if decode:
                # First get latent
                latent_kw = {**_call_kwargs(i), "output_type": "latent"}
                result = pipe(prompt=prompt, generator=gen, **latent_kw)
                latent = result.images if hasattr(result, "images") else result[0]
                if isinstance(latent, torch.Tensor):
                    ref_latents.append(latent.cpu())

                # Then get decoded
                gen = torch.Generator(device="cuda").manual_seed(i)
                result = pipe(prompt=prompt, generator=gen, **_call_kwargs(i))
                decoded = result.images if hasattr(result, "images") else result[0]
                if isinstance(decoded, torch.Tensor):
                    ref_decoded.append(decoded.cpu())
                elif isinstance(decoded, list):
                    import numpy as np
                    frames = [torch.from_numpy(np.array(img)).float() / 255.0 for img in decoded]
                    ref_decoded.append(torch.stack(frames).permute(0, 3, 1, 2).cpu())
            else:
                result = pipe(prompt=prompt, generator=gen, **_call_kwargs(i))
                latent = result.images if hasattr(result, "images") else result[0]
                if isinstance(latent, torch.Tensor):
                    ref_latents.append(latent.cpu())

    # Determine data range for latents (they can be outside [0,1])
    if ref_latents:
        all_ref = torch.cat([r.flatten() for r in ref_latents])
        latent_data_range = (all_ref.max() - all_ref.min()).item()
        if latent_data_range == 0:
            latent_data_range = 1.0
    else:
        latent_data_range = 1.0

    # Process each quantized file
    resolved_source = None if custom_prompts else (prompt_source or detect_prompt_source(model_source) or "t2i")
    all_results: dict[str, Any] = {
        "components": {},
        "prompts": prompts,
        "config": {
            "model_source": model_source,
            "num_prompts": len(prompts),
            "num_steps": num_steps,
            "resolution": resolution,
            "decode": decode,
            "prompt_source": resolved_source,
        },
    }

    for qf in quant_files:
        component_name = qf["component"]
        dtype = qf["dtype"]
        label = f"{component_name} ({dtype})"
        log.info("Evaluating quantized component: %s", label)

        # Find the matching component in the pipeline
        comp_attr = component_name
        if not hasattr(pipe, comp_attr):
            log.warning("Component '%s' not found in pipeline, skipping", comp_attr)
            continue

        # Resolve original component path
        component_path = model_dir / component_name
        if not component_path.is_dir():
            log.warning("Original component path '%s' not found, skipping", component_path)
            continue

        # Load quantized weights and swap into pipeline
        original_component = getattr(pipe, comp_attr)
        try:
            quantized_model = _load_quantized_component(
                component_path, qf["path"], dtype, qf["library"], qf["class_name"]
            )
            setattr(pipe, comp_attr, quantized_model)
        except Exception as e:
            log.error("Failed to load quantized component '%s': %s", label, e)
            continue

        # Generate quantized outputs with same seeds
        log.info("Generating %d quantized outputs for %s...", len(prompts), label)
        comp_metrics: list[dict[str, Any]] = []

        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                gen = torch.Generator(device="cuda").manual_seed(i)
                log.info("  Quantized prompt %d/%d", i + 1, len(prompts))

                prompt_metrics: dict[str, Any] = {"prompt": prompt, "seed": i}

                if decode and ref_decoded:
                    # Latent comparison
                    latent_kw = {**_call_kwargs(i), "output_type": "latent"}
                    result = pipe(prompt=prompt, generator=gen, **latent_kw)
                    latent = result.images if hasattr(result, "images") else result[0]
                    if isinstance(latent, torch.Tensor):
                        frames = compute_metrics_per_frame(
                            ref_latents[i].unsqueeze(0) if ref_latents[i].dim() < 5 else ref_latents[i],
                            latent.cpu().unsqueeze(0) if latent.dim() < 5 else latent.cpu(),
                            latent_data_range,
                        )
                        prompt_metrics["latent_frames"] = frames

                    # Decoded comparison
                    gen = torch.Generator(device="cuda").manual_seed(i)
                    result = pipe(prompt=prompt, generator=gen, **_call_kwargs(i))
                    decoded = result.images if hasattr(result, "images") else result[0]
                    if isinstance(decoded, torch.Tensor):
                        dec_tensor = decoded.cpu()
                    elif isinstance(decoded, list):
                        import numpy as np
                        dec_frames = [torch.from_numpy(np.array(img)).float() / 255.0 for img in decoded]
                        dec_tensor = torch.stack(dec_frames).permute(0, 3, 1, 2).cpu()
                    else:
                        dec_tensor = None

                    if dec_tensor is not None:
                        decoded_frames = compute_metrics_per_frame(
                            ref_decoded[i].unsqueeze(0) if ref_decoded[i].dim() < 4 else ref_decoded[i],
                            dec_tensor.unsqueeze(0) if dec_tensor.dim() < 4 else dec_tensor,
                            1.0,
                        )
                        prompt_metrics["decoded_frames"] = decoded_frames
                else:
                    result = pipe(prompt=prompt, generator=gen, **_call_kwargs(i))
                    latent = result.images if hasattr(result, "images") else result[0]
                    if isinstance(latent, torch.Tensor):
                        frames = compute_metrics_per_frame(
                            ref_latents[i].unsqueeze(0) if ref_latents[i].dim() < 5 else ref_latents[i],
                            latent.cpu().unsqueeze(0) if latent.dim() < 5 else latent.cpu(),
                            latent_data_range,
                        )
                        prompt_metrics["latent_frames"] = frames

                comp_metrics.append(prompt_metrics)

        all_results["components"][label] = {
            "component": component_name,
            "dtype": dtype,
            "metrics": comp_metrics,
        }

        # Restore original component
        setattr(pipe, comp_attr, original_component)
        del quantized_model
        torch.cuda.empty_cache()
        gc.collect()

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    return all_results
