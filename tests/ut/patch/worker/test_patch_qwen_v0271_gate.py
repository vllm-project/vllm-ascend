# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.patch.worker import patch_qwen_v0271_gate as control


@pytest.mark.parametrize("marked,cached,expected", [(True, False, False), (True, True, True), (False, False, True)])
def test_v0271_policy_is_scoped_to_marked_gates(marked, cached, expected):
    gate = SimpleNamespace(ascend_v0271_gate_control=marked)
    if cached:
        gate.weight_fp32 = torch.ones(3, 4)
    assert AscendMoERunner.is_internal_router.fget(SimpleNamespace(gate=gate)) is expected


def test_constructor_disables_precast_before_weight_loading():
    def original_init(block):
        block.gate = SimpleNamespace(precast_fp32_weight=True)

    block = SimpleNamespace()
    control._with_v0271_gate_policy(original_init)(block)
    assert block.gate.precast_fp32_weight is False
    assert block.gate.ascend_v0271_gate_control is True
    assert not AscendMoERunner.is_internal_router.fget(SimpleNamespace(gate=block.gate))


@pytest.mark.parametrize("forward", [control.qwen3_moe_forward, control.qwen3_next_forward])
@pytest.mark.parametrize("internal", [False, True])
def test_gate_runs_once_outside_experts_only_for_external_routing(forward, internal):
    hidden = torch.randn(4, 8, dtype=torch.bfloat16)
    logits = torch.randn(4, 3, dtype=torch.bfloat16)
    gate = Mock(return_value=(logits, None))
    experts = Mock(return_value=hidden)
    experts.is_internal_router = internal
    block = SimpleNamespace(is_sequence_parallel=False, gate=gate, experts=experts)

    result = forward(block, hidden)

    assert torch.equal(result, hidden)
    forwarded = experts.call_args.kwargs
    assert forwarded["router_logits"] is (forwarded["hidden_states"] if internal else logits)
    assert forwarded["hidden_states"].dtype == torch.bfloat16
    assert gate.call_count == (0 if internal else 1)


@pytest.mark.parametrize("already_parallel", [False, True])
def test_next_sequence_parallel_order_and_replicated_shared_output(monkeypatch, already_parallel):
    hidden = torch.ones(4, 8, dtype=torch.bfloat16)
    chunk = Mock(side_effect=lambda x: x[:2])
    gather = Mock(side_effect=lambda x, dim: torch.cat([x, x], dim=dim))
    monkeypatch.setattr(control, "sequence_parallel_chunk", chunk)
    monkeypatch.setattr(control, "tensor_model_parallel_all_gather", gather)
    gate = Mock(side_effect=lambda x: (x[:, :3], None))
    experts = Mock(side_effect=lambda **kwargs: kwargs["hidden_states"].clone())
    experts.is_internal_router = False
    shared = Mock(side_effect=lambda x: torch.full_like(x, 2))
    block = SimpleNamespace(
        is_sequence_parallel=True,
        gate=gate,
        experts=experts,
        replicate_shared_expert=True,
        shared_expert=shared,
    )

    result = control.qwen3_next_forward(block, hidden, already_sequence_parallel=already_parallel)

    assert torch.equal(result, torch.full_like(hidden, 3))
    assert chunk.call_count == gather.call_count == (0 if already_parallel else 1)
    assert gate.call_args.args[0].shape[0] == (4 if already_parallel else 2)
    assert shared.call_count == 1
