from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.kimi_k3 import (
    _k3_external_router_logits,
    _prepare_k3_external_router_weight,
)


def test_external_router_keeps_gate_path_when_disabled(monkeypatch):
    logits = torch.ones(2, 3, dtype=torch.float32)
    calls = []

    class Gate:
        def __call__(self, hidden_states):
            calls.append(hidden_states)
            return logits, None

    monkeypatch.setattr(torch, "mm", lambda *args, **kwargs: pytest.fail("BF16 matmul was called"))
    hidden_states = torch.ones(2, 8, dtype=torch.bfloat16)
    assert _k3_external_router_logits(hidden_states, Gate(), False) is logits
    assert calls == [hidden_states]


def test_external_router_uses_bf16_operands_and_fp32_logits(monkeypatch):
    hidden_states = torch.ones(2, 8, dtype=torch.float32)
    weight = torch.ones(3, 8, dtype=torch.bfloat16)
    gate = SimpleNamespace(weight=weight, bias=None)
    output = torch.zeros(2, 3, dtype=torch.float32)
    seen = []

    def fake_mm(lhs, rhs, *, out_dtype):
        seen.append((lhs.dtype, rhs.dtype, out_dtype, lhs.shape, rhs.shape))
        return output

    monkeypatch.setattr(torch, "mm", fake_mm)
    assert _k3_external_router_logits(hidden_states, gate, True) is output
    assert seen == [(torch.bfloat16, torch.bfloat16, torch.float32, (2, 8), (8, 3))]


@pytest.mark.parametrize("dtype,bias", [(torch.float32, None), (torch.bfloat16, torch.ones(3))])
def test_external_router_rejects_incompatible_gate(dtype, bias, monkeypatch):
    gate = SimpleNamespace(weight=torch.ones(3, 8, dtype=dtype), bias=bias)
    monkeypatch.setattr(torch, "mm", lambda *args, **kwargs: pytest.fail("matmul was called"))
    with pytest.raises(ValueError, match="bias-free BF16 gate weight"):
        _k3_external_router_logits(torch.ones(2, 8), gate, True)


def test_prepare_router_weight_converts_loaded_fp32_once():
    gate = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(3, 8)), bias=None)
    _prepare_k3_external_router_weight(gate)
    assert gate.weight.dtype == torch.bfloat16
    storage = gate.weight.data_ptr()
    _prepare_k3_external_router_weight(gate)
    assert gate.weight.data_ptr() == storage


def test_prepare_router_weight_rejects_bias():
    gate = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(3, 8)), bias=torch.ones(3))
    with pytest.raises(ValueError, match="bias-free gate"):
        _prepare_k3_external_router_weight(gate)
