# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test explicit fusion dispatch and metadata without requiring an NPU kernel."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def dispatch(monkeypatch):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/patch_deepseek_v2.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"_supports_dense_norm_quant", "_maybe_fused_add_rms_norm_quant"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names

    class DenseMLP:
        gate_up_proj = object()

    class Norm:
        weight = object()
        bias = None
        variance_epsilon = 1e-6

        def __call__(self, hidden, residual):
            return hidden, residual

    kernel = Mock()
    monkeypatch.setattr(torch.ops.npu, "npu_add_rms_norm_dynamic_quant", kernel, raising=False)
    scope = {
        "torch": torch,
        "DeepseekV2MLP": DenseMLP,
        "AscendRMSNorm": Norm,
        "enable_custom_op": lambda: True,
        "HardwareCapability": SimpleNamespace(GRAPH_NORM_QUANT_FUSION=object()),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: True),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(source), "exec"), scope)
    return SimpleNamespace(scope=scope, kernel=kernel, mlp=DenseMLP(), norm=Norm())


def test_selects_only_dense_consumer_with_matching_key(dispatch):
    select = dispatch.scope["_supports_dense_norm_quant"]
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)
    layer = SimpleNamespace(mlp=dispatch.mlp, use_sequence_parallel_moe=False, post_attention_layernorm=dispatch.norm)
    key = object()

    assert select(layer, hidden, lambda _: key, key)
    assert not select(layer, hidden, lambda _: None, key)  # e.g. LoRA needs the original input
    layer.use_sequence_parallel_moe = True
    assert not select(layer, hidden, lambda _: key, key)
    layer.use_sequence_parallel_moe = False
    layer.mlp = object()  # MoE does not enter the dense path
    assert not select(layer, hidden, lambda _: key, key)


def test_fused_outputs_keep_quantization_and_residual_metadata(dispatch):
    fused = dispatch.scope["_maybe_fused_add_rms_norm_quant"]
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)
    residual = torch.zeros_like(hidden)
    data = torch.ones((2, 4), dtype=torch.int8)
    scale = torch.ones(2)
    updated_residual = torch.ones_like(hidden)
    dispatch.kernel.return_value = (data, None, updated_residual, scale)
    key = object()

    output, actual_residual = fused(dispatch.norm, hidden, residual, SimpleNamespace, key)

    assert output.data is data and output.scale is scale
    assert output.orig_dtype == hidden.dtype and output.orig_shape == hidden.shape
    assert output.quant_key is key and actual_residual is updated_residual
    dispatch.kernel.assert_called_once()


def test_noncontiguous_input_falls_back_to_norm(dispatch):
    fused = dispatch.scope["_maybe_fused_add_rms_norm_quant"]
    hidden = torch.ones((4, 2), dtype=torch.bfloat16).t()
    residual = torch.zeros_like(hidden)

    result = fused(dispatch.norm, hidden, residual, SimpleNamespace, object())

    assert result == (hidden, residual)
    dispatch.kernel.assert_not_called()
