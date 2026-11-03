# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from unittest.mock import MagicMock

from vllm.model_executor.layers.mamba.ops import causal_conv1d as cc1d

from vllm_ascend.ops.triton.mamba import causal_conv1d as mamba_conv1d
from vllm_ascend.patch.worker import patch_ascend_ops, patch_triton

_REMOVED_KERNEL = "causal_conv1d_update_npu"


def test_mamba_module_exports_no_update_kernel():
    """The symbol the Triton rebind imported is gone, not merely renamed."""
    assert not hasattr(mamba_conv1d, _REMOVED_KERNEL)


def test_patch_triton_does_not_rebind_the_removed_kernel():
    """Guard against re-adding an import that can never resolve."""
    tree = ast.parse(Path(patch_triton.__file__).read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert _REMOVED_KERNEL not in imported


def test_bound_update_is_the_ascend_wrapper():
    """The Ascend wrapper stays the effective implementation for vLLM consumers."""
    assert cc1d.causal_conv1d_update is patch_ascend_ops._npu_causal_conv1d_update


def test_wrapper_reports_the_fallback_once_per_call(monkeypatch):
    """The host-syncing fallback must be announced, and only once per call."""
    mock_logger = MagicMock()
    monkeypatch.setattr(patch_ascend_ops, "logger", mock_logger)
    impl = MagicMock(return_value="conv-out")
    monkeypatch.setattr(patch_ascend_ops, "_npu_causal_conv1d_update_impl", impl)

    # Spec-decode metadata that vLLM passes and the Ascend kernel does not take.
    out = patch_ascend_ops._npu_causal_conv1d_update(
        "x",
        "weight",
        conv_state_indices=None,
        null_block_id=42,
        max_query_len=8,
        metadata=object(),
    )

    assert out == "conv-out"
    impl.assert_called_once_with("x", "weight", conv_state_indices=None)
    mock_logger.warning.assert_not_called()
    mock_logger.warning_once.assert_called_once()
    message = mock_logger.warning_once.call_args.args[0]
    assert "PyTorch Ascend implementation" in message
    assert "VLLM_USE_BREAKABLE_CUDAGRAPH" in message
