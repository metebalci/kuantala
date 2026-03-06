"""Tests for NVIDIA backend."""

import pytest

try:
    import torch
    import modelopt
    HAS_NVIDIA_DEPS = torch.cuda.is_available()
    if HAS_NVIDIA_DEPS:
        _cc = torch.cuda.get_device_capability()
        HAS_MXFP8 = _cc >= (9, 0)   # Hopper+
        HAS_NVFP4 = _cc >= (10, 0)  # Blackwell+
    else:
        HAS_MXFP8 = False
        HAS_NVFP4 = False
except ImportError:
    HAS_NVIDIA_DEPS = False
    HAS_MXFP8 = False
    HAS_NVFP4 = False

requires_nvidia = pytest.mark.skipif(
    not HAS_NVIDIA_DEPS,
    reason="Requires torch with CUDA and nvidia-modelopt",
)

requires_mxfp8 = pytest.mark.skipif(
    not HAS_MXFP8,
    reason="Requires Hopper+ GPU (compute capability >= 9.0)",
)

requires_nvfp4 = pytest.mark.skipif(
    not HAS_NVFP4,
    reason="Requires Blackwell+ GPU (compute capability >= 10.0)",
)


def test_check_dependencies_no_torch(monkeypatch):
    """Should raise ImportError with install instructions when torch is missing."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from kuantala.backends.nvidia_backend import _check_dependencies
    with pytest.raises(ImportError, match="PyTorch not installed"):
        _check_dependencies()


@requires_nvidia
def test_get_quant_config_mxfp8():
    from kuantala.backends.nvidia_backend import _get_quant_config
    cfg = _get_quant_config("MXFP8")
    assert cfg is not None


@requires_nvidia
def test_get_quant_config_nvfp4():
    from kuantala.backends.nvidia_backend import _get_quant_config
    cfg = _get_quant_config("NVFP4")
    assert cfg is not None


@requires_nvidia
def test_get_quant_config_invalid():
    from kuantala.backends.nvidia_backend import _get_quant_config
    with pytest.raises(ValueError, match="Unsupported NVIDIA dtype"):
        _get_quant_config("Q4_K")


@requires_nvidia
def test_nvidia_backend_init():
    from kuantala.backends.nvidia_backend import NvidiaBackend
    backend = NvidiaBackend()
    assert "MXFP8" in backend.supported_dtypes()
    assert "NVFP4" in backend.supported_dtypes()


@requires_nvidia
def test_nvidia_backend_info():
    from kuantala.backends.nvidia_backend import NvidiaBackend
    backend = NvidiaBackend()
    info = backend.get_info()
    assert info["name"] == "NVIDIA"
    assert info["requires_torch"] is True


@requires_mxfp8
def test_quantize_small_linear_mxfp8():
    """Quantize a small Linear layer to MXFP8 via modelopt."""
    import modelopt.torch.quantization as mtq

    model = torch.nn.Linear(64, 32).cuda()
    original_weight = model.weight.clone()

    cfg = mtq.FP8_DEFAULT_CFG
    mtq.quantize(model, cfg, forward_loop=lambda m: m(torch.randn(1, 64).cuda()))

    # Weight should still exist and have the same shape
    assert model.weight.shape == original_weight.shape


@requires_nvfp4
def test_quantize_small_linear_nvfp4():
    """Quantize a small Linear layer to NVFP4 via modelopt."""
    import modelopt.torch.quantization as mtq

    model = torch.nn.Linear(64, 32).cuda()
    original_weight = model.weight.clone()

    cfg = mtq.NVFP4_DEFAULT_CFG
    mtq.quantize(model, cfg, forward_loop=lambda m: m(torch.randn(1, 64).cuda()))

    assert model.weight.shape == original_weight.shape
