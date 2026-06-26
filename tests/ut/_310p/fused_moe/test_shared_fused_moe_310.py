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

from vllm_ascend._310p.fused_moe import fused_moe as fused_moe_310_module


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


def _build_routed_experts(shared_experts: torch.nn.Module | None):
    # vLLM PR #41184 moved the shared-expert owner from the FusedMoE layer to
    # RoutedExperts. Build only the minimal owner state needed by
    # these helper tests.
    routed_experts = torch.nn.Module.__new__(fused_moe_310_module.AscendRoutedExperts310)
    torch.nn.Module.__init__(routed_experts)
    routed_experts._shared_experts = shared_experts
    return routed_experts


def test_routed_experts_forward_shared_experts_without_gate_310():
    routed_experts = _build_routed_experts(_DummySharedExperts(with_gate=False))
    hidden_states = torch.randn(4, 8)
    output = routed_experts._forward_shared_experts(hidden_states)
    expected = hidden_states * 2.0 + 1.0
    torch.testing.assert_close(output, expected)


def test_routed_experts_forward_shared_experts_with_gate_310():
    routed_experts = _build_routed_experts(_DummySharedExperts(with_gate=True))
    hidden_states = torch.randn(4, 8)
    output = routed_experts._forward_shared_experts(hidden_states)
    expected = 0.5 * (hidden_states * 2.0 + 1.0)
    torch.testing.assert_close(output, expected)


def test_routed_experts_shared_forward_uses_separate_shared_input_310():
    routed_experts = _build_routed_experts(_DummySharedExperts(with_gate=True))
    hidden_states = torch.randn(3, 8)
    shared_input = torch.randn(3, 8)
    router_logits = torch.randn(3, 8)
    routed_out = torch.randn(3, 8)

    # vLLM PR #41184 makes MoERunner pass a separate shared_experts_input to
    # RoutedExperts.shared_forward_impl. 310P should consume that tensor only
    # for the shared path and keep routed MoE inputs unchanged.
    with patch.object(
        fused_moe_310_module.AscendRoutedExperts310,
        "forward_impl",
        return_value=routed_out,
    ) as mock_forward_impl:
        shared_out, routed = routed_experts.shared_forward_impl(
            hidden_states,
            router_logits,
            shared_input,
        )

    mock_forward_impl.assert_called_once_with(
        hidden_states=hidden_states,
        router_logits=router_logits,
    )
    expected_shared = 0.5 * (shared_input * 2.0 + 1.0)
    torch.testing.assert_close(shared_out, expected_shared)
    torch.testing.assert_close(routed, routed_out)


def test_routed_experts_without_shared_experts_integration_310():
    routed_experts = _build_routed_experts(None)
    hidden_states = torch.randn(3, 8)
    assert routed_experts._forward_shared_experts(hidden_states) is None


def test_routed_experts_without_shared_experts_returns_routed_only_310():
    routed_experts = _build_routed_experts(None)
    hidden_states = torch.randn(3, 8)
    router_logits = torch.randn(3, 8)
    routed_out = torch.randn(3, 8)

    with patch.object(
        fused_moe_310_module.AscendRoutedExperts310,
        "forward_impl",
        return_value=routed_out,
    ):
        output = routed_experts.shared_forward_impl(hidden_states, router_logits)

    torch.testing.assert_close(output, routed_out)


def test_routed_experts_is_internal_router_is_false_310():
    routed_experts = _build_routed_experts(_DummySharedExperts(with_gate=True))
    assert routed_experts.is_internal_router is False
