"""Tests for GGUF backend quantization routines."""

import numpy as np
import pytest


def test_quantize_tensor_q8_0():
    from kuantala.backends.gguf_backend import _quantize_tensor_q8_0
    data = np.random.randn(256).astype(np.float32)
    quantized, qtype = _quantize_tensor_q8_0(data)
    assert isinstance(quantized, np.ndarray)
    # Q8_0: 256 values = 8 blocks of 32, each block = 2 (scale) + 32 (values) = 34 bytes
    assert len(quantized) == 8 * 34


def test_quantize_tensor_q4_0():
    from kuantala.backends.gguf_backend import _quantize_tensor_q4_0
    data = np.random.randn(256).astype(np.float32)
    quantized, qtype = _quantize_tensor_q4_0(data)
    assert isinstance(quantized, np.ndarray)
    # Q4_0: 256 values = 8 blocks of 32, each block = 2 (scale) + 16 (packed) = 18 bytes
    assert len(quantized) == 8 * 18


def test_quantize_tensor_q5_0():
    from kuantala.backends.gguf_backend import _quantize_tensor_q5_0
    data = np.random.randn(256).astype(np.float32)
    quantized, qtype = _quantize_tensor_q5_0(data)
    assert isinstance(quantized, np.ndarray)
    # Q5_0: 256 values = 8 blocks of 32, each block = 2 (scale) + 4 (high bits) + 16 (low nibbles) = 22 bytes
    assert len(quantized) == 8 * 22


def test_quantize_preserves_shape_info():
    """Quantizing and the shape should be tracked externally."""
    from kuantala.backends.gguf_backend import _quantize_tensor_q8_0
    shape = (4, 64)
    data = np.random.randn(*shape).astype(np.float32)
    quantized, _ = _quantize_tensor_q8_0(data)
    # Data is flattened internally, shape is preserved by the writer
    assert quantized.ndim == 1


def test_zero_tensor():
    """Quantizing a zero tensor should not crash."""
    from kuantala.backends.gguf_backend import _quantize_tensor_q8_0
    data = np.zeros(64, dtype=np.float32)
    quantized, _ = _quantize_tensor_q8_0(data)
    assert len(quantized) == 2 * 34  # 2 blocks of 32
