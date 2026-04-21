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

from unittest.mock import patch

import torch
import torch.nn.functional as F

from vllm_ascend._310p.fused_moe.fused_moe import (
    AscendFusedMoE310,
    AscendSharedFusedMoE310,
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
        shared_out, routed = layer.forward_impl(hidden_states, router_logits)

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
        output = layer.forward_impl(hidden_states, router_logits)

    torch.testing.assert_close(output, routed_out)


def test_is_internal_router_is_false_310():
    layer = _build_layer(_DummySharedExperts(with_gate=True))
    assert layer.is_internal_router is False
