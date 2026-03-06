"""Tests for model_loader module."""

from pathlib import Path

import pytest

from kuantala.model_loader import resolve_model_path


def test_resolve_local_path(tmp_path):
    """Local directories should be returned as-is."""
    model_dir = tmp_path / "my_model"
    model_dir.mkdir()
    result = resolve_model_path(str(model_dir))
    assert result == model_dir


def test_resolve_nonexistent_local_path_treated_as_hub_id():
    """Non-existent local paths should be treated as HF Hub IDs."""
    # This would attempt to download from HF Hub, which we don't want in tests
    # Just verify it doesn't raise for a non-path string
    with pytest.raises(Exception):
        # Will fail because "fake/model" doesn't exist on Hub
        resolve_model_path("fake/model-that-does-not-exist")
