"""Model loading: HuggingFace Hub download + local path resolution."""

from __future__ import annotations

from pathlib import Path

from kuantala.utils import get_logger

log = get_logger(__name__)


def resolve_model_path(source: str, token: str | None = None) -> Path:
    """Resolve a model source to a local directory path.

    Args:
        source: Either a local path or a HuggingFace Hub model ID (e.g. "user/model").
        token: Optional HuggingFace auth token for gated models.

    Returns:
        Path to local directory containing model files.
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

    log.info("Downloading %s from HuggingFace Hub...", source)
    cache_dir = snapshot_download(
        repo_id=source,
        token=token,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    )
    log.info("Model cached at %s", cache_dir)
    return Path(cache_dir)
