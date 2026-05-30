#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

import vllm_ascend._310p.fused_moe.fused_moe as fused_moe_310
from vllm_ascend._310p.fused_moe.fused_moe import (
    AscendFusedMoE310,
    AscendSharedFusedMoE310,
    FusedMoEResult310,
)


class _DummyGate(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        # Keep gate output deterministic: sigmoid(0)=0.5.
        return torch.zeros(
            hidden_states.shape[0],
            1,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ), None


class _DummySharedExperts(torch.nn.Module):
    def __init__(self, with_gate: bool):
        super().__init__()
        self.expert_gate = _DummyGate() if with_gate else None

    def forward(self, hidden_states: torch.Tensor):
        out = hidden_states * 2.0 + 1.0
        if self.expert_gate is not None:
            gate_out, _ = self.expert_gate(hidden_states)
            out = F.sigmoid(gate_out) * out
        return out


class _TupleLinear(torch.nn.Module):
    def __init__(self, scale: float, bias: float = 0.0):
        super().__init__()
        self.scale = scale
        self.bias = bias

    def forward(self, hidden_states: torch.Tensor):
        return hidden_states * self.scale + self.bias, None


class _SplitSharedExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = _TupleLinear(2.0, 1.0)
        self.act_fn = torch.nn.Identity()
        self.down_proj = _TupleLinear(3.0)
        self.expert_gate = None

    def forward(self, hidden_states: torch.Tensor):
        shared_gate_up, _ = self.gate_up_proj(hidden_states)
        shared_out, _ = self.down_proj(self.act_fn(shared_gate_up))
        return shared_out


def _build_layer(shared_experts: torch.nn.Module | None) -> AscendSharedFusedMoE310:
    layer = AscendSharedFusedMoE310.__new__(AscendSharedFusedMoE310)
    # The test bypasses full layer init with __new__, so we must initialize
    # nn.Module internals before assigning child modules.
    torch.nn.Module.__init__(layer)
    layer._shared_experts = shared_experts
    return layer


def test_forward_shared_experts_without_gate_310():
    layer = _build_layer(_DummySharedExperts(with_gate=False))
    hidden_states = torch.randn(4, 8)
    output = layer._forward_shared_experts(hidden_states)
    expected = hidden_states * 2.0 + 1.0
    torch.testing.assert_close(output, expected)


def test_forward_shared_experts_with_gate_310():
    layer = _build_layer(_DummySharedExperts(with_gate=True))
    hidden_states = torch.randn(4, 8)
    output = layer._forward_shared_experts(hidden_states)
    expected = 0.5 * (hidden_states * 2.0 + 1.0)
    torch.testing.assert_close(output, expected)


def test_forward_impl_with_shared_experts_returns_tuple_310():
    layer = _build_layer(_DummySharedExperts(with_gate=True))
    hidden_states = torch.randn(3, 8)
    router_logits = torch.randn(3, 8)
    routed_out = torch.randn(3, 8)

    with patch.object(AscendFusedMoE310, "forward_impl", return_value=routed_out):
        shared_out, routed = layer.shared_forward_impl(hidden_states, router_logits)

    expected_shared = 0.5 * (hidden_states * 2.0 + 1.0)
    torch.testing.assert_close(shared_out, expected_shared)
    torch.testing.assert_close(routed, routed_out)


def test_forward_impl_without_shared_experts_integration_310():
    layer = _build_layer(None)
    hidden_states = torch.randn(3, 8)
    assert layer._forward_shared_experts(hidden_states) is None


def test_forward_impl_without_shared_experts_returns_routed_only_310():
    layer = _build_layer(None)
    hidden_states = torch.randn(3, 8)
    router_logits = torch.randn(3, 8)
    routed_out = torch.randn(3, 8)

    with patch.object(AscendFusedMoE310, "forward_impl", return_value=routed_out):
        output = layer.shared_forward_impl(hidden_states, router_logits)

    torch.testing.assert_close(output, routed_out)


def test_forward_impl_with_multistream_shared_experts_uses_event_result_310():
    layer = _build_layer(_SplitSharedExperts())
    layer.multistream_overlap_shared_expert = True
    layer.shared_expert_stream = MagicMock()
    hidden_states = torch.randn(3, 8)
    router_logits = torch.randn(3, 8)
    routed_out = torch.randn(3, 8)
    before_dispatch_evt = MagicMock()
    before_combine_evt = MagicMock()
    current_stream = MagicMock()
    before_routed_evt = MagicMock()
    current_stream.record_event.return_value = before_routed_evt
    call_order = []

    original_part1 = layer._shared_experts_part1
    original_part2 = layer._shared_experts_part2

    def tracked_part1(hidden_states):
        call_order.append("shared_part1")
        return original_part1(hidden_states)

    def tracked_part2(hidden_states, shared_gate_up):
        call_order.append("shared_part2")
        return original_part2(hidden_states, shared_gate_up)

    def routed_forward(*args, **kwargs):
        call_order.append("routed")
        return FusedMoEResult310(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_combine_evt=before_combine_evt,
        )

    with (
        patch.object(
            AscendFusedMoE310,
            "forward_impl",
            side_effect=routed_forward,
        ) as routed_forward,
        patch.object(layer, "_shared_experts_part1", side_effect=tracked_part1),
        patch.object(layer, "_shared_experts_part2", side_effect=tracked_part2),
        patch.object(fused_moe_310.torch_npu.npu, "current_stream", return_value=current_stream),
        patch.object(layer, "_shared_stream_context", return_value=nullcontext()),
    ):
        shared_out, routed = layer.forward_impl(hidden_states, router_logits)

    routed_forward.assert_called_once_with(
        layer,
        hidden_states=hidden_states,
        router_logits=router_logits,
        return_with_event=True,
    )
    assert call_order == ["shared_part1", "shared_part2", "routed"]
    layer.shared_expert_stream.wait_event.assert_any_call(before_routed_evt)
    current_stream.wait_stream.assert_called_once_with(layer.shared_expert_stream)
    torch.testing.assert_close(shared_out, layer._shared_experts(hidden_states))
    torch.testing.assert_close(routed, routed_out)


def test_forward_impl_with_multistream_shared_experts_requires_split_modules_310():
    layer = _build_layer(_DummySharedExperts(with_gate=True))
    layer.multistream_overlap_shared_expert = True
    hidden_states = torch.randn(3, 8)
    router_logits = torch.randn(3, 8)

    with pytest.raises(RuntimeError, match="requires shared_experts to expose"):
        layer.forward_impl(hidden_states, router_logits)


def test_is_internal_router_is_false_310():
    layer = _build_layer(_DummySharedExperts(with_gate=True))
    assert layer.is_internal_router is False
