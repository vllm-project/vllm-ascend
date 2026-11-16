"""Tests for static kernel sym_range selection in _configure_backend."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.compilation.compiler_interface import _configure_backend


def _run_configure_backend(mode, capture_sizes, enable_super_kernel=False):
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=capture_sizes,
        ),
    )
    ascend_compilation_config = SimpleNamespace(
        enable_static_kernel=True,
        enable_super_kernel=enable_super_kernel,
    )
    options = {}

    def process_kwargs_options(config, payload):
        options.update(payload["options"])

    _configure_backend(
        SimpleNamespace(),
        ascend_compilation_config,
        vllm_config,
        process_kwargs_options,
    )
    return options


@pytest.mark.parametrize(
    "mode",
    [
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_sym_range_uses_cudagraph_capture_sizes_under_piecewise(mode):
    """PIECEWISE pieces are captured and replayed exactly at the cudagraph
    capture sizes, so those sizes must form the static kernel sym_range."""
    capture_sizes = [16, 1024, 2048, 4096, 6400, 8192]
    options = _run_configure_backend(mode, capture_sizes)

    assert options["static_kernel_compile"] is True
    assert options["_vllm_aclnn_static_kernel_sym_range"] == capture_sizes


@pytest.mark.parametrize(
    "mode",
    [
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.NONE,
    ],
)
def test_sym_range_uses_decode_batch_sizes_otherwise(mode):
    """Non-piecewise routes keep the original decode batch-size sym_range."""
    with patch(
        "vllm_ascend.compilation.compiler_interface._compute_decode_cudagraph_batch_sizes",
        return_value=[1, 2, 4],
    ) as mock_compute:
        options = _run_configure_backend(mode, [16, 1024])

    assert options["_vllm_aclnn_static_kernel_sym_range"] == [1, 2, 4]
    mock_compute.assert_called_once()


def test_super_kernel_options_enabled():
    options = _run_configure_backend(CUDAGraphMode.PIECEWISE, [16], enable_super_kernel=True)

    assert options["super_kernel_optimize"] is True
    assert options["super_kernel_optimize_options"] == {"dcci_after_kernel_end": [".*"]}
