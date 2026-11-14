# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import fla_npu.ops
import pytest
import torch
from torch import nn

from vllm_ascend.ops import gdn


def test_recurrent_gate_bias_materializes_after_meta_weight_load():
    layer = nn.Module()
    layer.dt_bias = nn.Parameter(torch.empty(2, device="meta", dtype=torch.bfloat16))
    layer.register_buffer("_recurrent_dt_bias_fp32", torch.empty(2, device="meta"), persistent=False)
    pending = layer._recurrent_dt_bias_fp32
    gdn._refresh_recurrent_gate_bias(layer)
    assert layer._recurrent_dt_bias_fp32 is pending

    layer.dt_bias = nn.Parameter(torch.tensor([0.3, -1.7], dtype=torch.bfloat16))
    gdn._refresh_recurrent_gate_bias(layer)
    cached = layer._recurrent_dt_bias_fp32
    assert cached.device == layer.dt_bias.device
    assert cached.dtype == torch.float32
    torch.testing.assert_close(cached, layer.dt_bias.float(), rtol=0, atol=0)
    address = cached.data_ptr()
    with torch.no_grad():
        layer.dt_bias.add_(1)
    gdn._refresh_recurrent_gate_bias(layer)
    assert layer._recurrent_dt_bias_fp32 is cached
    assert cached.data_ptr() == address
    torch.testing.assert_close(cached, layer.dt_bias.float(), rtol=0, atol=0)
    assert "_recurrent_dt_bias_fp32" not in layer.state_dict()


@pytest.fixture
def fusion_backend(monkeypatch):
    gdn._supports_fused_recurrent_gdn.cache_clear()
    monkeypatch.setattr(gdn, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: True))
    backend = SimpleNamespace()
    monkeypatch.setattr(fla_npu.ops, "ascendc", backend)
    yield backend
    gdn._supports_fused_recurrent_gdn.cache_clear()


def _extended_op(*, use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, A_log, dt_bias):
    pass


@pytest.mark.parametrize("op", [None, object(), lambda: None])
def test_fusion_probe_rejects_unsupported_resolved_ops(fusion_backend, op):
    fusion_backend._get_direct_op = Mock(return_value=op)
    assert not gdn._supports_fused_recurrent_gdn()


@pytest.mark.parametrize("error", [ImportError, AttributeError, RuntimeError, TypeError, ValueError])
def test_fusion_probe_handles_backend_resolution_errors(fusion_backend, error):
    fusion_backend._get_direct_op = Mock(side_effect=error("backend unavailable"))
    assert not gdn._supports_fused_recurrent_gdn()
    assert not gdn._supports_fused_recurrent_gdn()
    fusion_backend._get_direct_op.assert_called_once_with("npu_recurrent_gated_delta_rule")


def test_fusion_probe_handles_uninspectable_callable(fusion_backend):
    class Uninspectable:
        @property
        def __signature__(self):
            raise ValueError("signature unavailable")

        def __call__(self):
            pass

    fusion_backend._get_direct_op = Mock(return_value=Uninspectable())
    assert not gdn._supports_fused_recurrent_gdn()


def test_fusion_probe_accepts_extended_backend(fusion_backend):
    fusion_backend._get_direct_op = Mock(return_value=_extended_op)
    assert gdn._supports_fused_recurrent_gdn()


@pytest.mark.parametrize("op,expected", [(_extended_op, True), (lambda: None, False)])
def test_fusion_probe_without_resolver(fusion_backend, monkeypatch, op, expected):
    monkeypatch.setattr(gdn, "recurrent_gated_delta_rule", op)
    assert gdn._supports_fused_recurrent_gdn() is expected


def test_fusion_probe_skips_backend_on_non_a5(fusion_backend, monkeypatch):
    monkeypatch.setattr(gdn, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: False))
    fusion_backend._get_direct_op = Mock(side_effect=AssertionError("non-A5 backend probe"))
    assert not gdn._supports_fused_recurrent_gdn()
    fusion_backend._get_direct_op.assert_not_called()


def test_fusion_probe_does_not_hide_unexpected_errors(fusion_backend):
    fusion_backend._get_direct_op = Mock(side_effect=AssertionError("backend bug"))
    with pytest.raises(AssertionError, match="backend bug"):
        gdn._supports_fused_recurrent_gdn()
