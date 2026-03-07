"""Model loading: HuggingFace Hub download + local path resolution."""

from __future__ import annotations

from pathlib import Path

from kuantala.utils import get_logger

log = get_logger(__name__)


def resolve_model_path(source: str) -> Path:
    """Resolve a model source to a local directory path.

    Args:
        source: Either a local path or a HuggingFace Hub model ID (e.g. "user/model").

    Returns:
        Path to local directory containing model files.

    Authentication is handled by huggingface-hub automatically (via ``hf auth login``
    or the ``HF_TOKEN`` environment variable).
    """
    local = Path(source)
    if local.is_dir():
        log.info("Using local model at %s", local)
        return local

    # Treat as HuggingFace Hub model ID
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            f"'{source}' is not a local directory and huggingface-hub is not installed. "
            "Install with: pip install kuantala[hub]"
        )

    # Check for model_index.json before downloading the full model
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(repo_id=source, filename="model_index.json")
    except Exception as e:
        raise FileNotFoundError(
            f"'{source}' does not contain a model_index.json on HuggingFace Hub. "
            "Kuantala requires a diffusers-format model (with model_index.json). "
            "Look for a '-Diffusers' variant of the model, e.g. 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'."
        ) from e

    log.info("Downloading %s from HuggingFace Hub...", source)
    cache_dir = snapshot_download(
        repo_id=source,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    )
    log.info("Model cached at %s", cache_dir)
    return Path(cache_dir)
