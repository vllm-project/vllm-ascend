# SPDX-License-Identifier: Apache-2.0
"""Preserve the selected CustomOp dispatch outside the native prefill contract."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def _load_prefill_entry(native, supported=True, registered=True):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/layernorm.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if getattr(node, "name", None) == "AscendFusedRMSNormGated")
    method = next(node for node in cls.body if getattr(node, "name", None) == "forward_prefill")

    class Norm:
        def forward_oot(self, x, gate, *, out):
            self.original_calls += 1
            return out

        def __call__(self, *args, **kwargs):
            return self._forward_method(*args, **kwargs)

    ops = SimpleNamespace(kda_rms_norm_gated=object()) if registered else SimpleNamespace()
    scope = dict(
        math=math,
        torch=SimpleNamespace(finfo=torch.finfo, float32=torch.float32, ops=SimpleNamespace(_C_ascend=ops)),
        AscendFusedRMSNormGated=Norm,
        supports_kda_rms_norm_gated=lambda *args: supported,
        is_950=lambda: True,
        kda_rms_norm_gated=native,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    Norm.forward_prefill = scope["forward_prefill"]
    layer = Norm()
    layer.original_calls = 0
    layer._forward_method = layer.forward_oot
    layer.bias, layer.eps, layer.activation = None, 1e-6, "sigmoid"
    layer.weight = torch.empty(128, device="meta")
    return layer


@pytest.mark.parametrize("activation", ["sigmoid", "swish", "silu"])
def test_prefill_native_preserves_gate_contract(activation):
    x = torch.empty(1, 1024, 12, 128, device="meta")
    gate, out = torch.empty_like(x), torch.empty_like(x)
    native = Mock(return_value=out)
    layer = _load_prefill_entry(native)
    layer.activation = activation
    assert layer.forward_prefill(x, gate, out=out) is out
    native.assert_called_once_with(x, gate, layer.weight, eps=1e-6, out=out, sigmoid_only=activation == "sigmoid")
    assert layer.original_calls == 0


@pytest.mark.parametrize("reason", ["dispatch", "unsupported", "unregistered", "short", "bias", "epsilon"])
def test_prefill_fallback_preserves_existing_dispatch(reason):
    x = torch.empty(1, 64 if reason == "short" else 16384, 12, 128, device="meta")
    gate, out = torch.empty_like(x), torch.empty_like(x)
    native = Mock(side_effect=AssertionError("unexpected native path"))
    layer = _load_prefill_entry(native, supported=reason != "unsupported", registered=reason != "unregistered")
    chosen = Mock(return_value=out)
    if reason == "dispatch":
        layer._forward_method = chosen
    elif reason == "bias":
        layer.bias = torch.empty(128, device="meta")
    elif reason == "epsilon":
        layer.eps = 1e-100  # positive double but underflows the native FP32 attribute
    assert layer.forward_prefill(x, gate, out=out) is out
    native.assert_not_called()
    if reason == "dispatch":
        chosen.assert_called_once_with(x, gate, out=out)
        assert layer.original_calls == 0
    else:
        assert layer.original_calls == 1
