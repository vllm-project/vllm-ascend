from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import experts_selector as experts_selector_module


@pytest.mark.parametrize(
    "moe_comm_type",
    [
        MoECommType.ALLGATHER,
        MoECommType.MC2,
        MoECommType.ALLTOALL,
        MoECommType.FUSED_MC2,
    ],
)
def test_hash_routing_uses_comm_aligned_input_ids(monkeypatch, moe_comm_type):
    raw_input_ids = torch.tensor([7, 8, 9, 10], dtype=torch.int64)
    aligned_input_ids = torch.tensor([9, -1, 7], dtype=torch.int64)
    tid2eid = torch.arange(64, dtype=torch.int64).view(16, 4)
    captured = {}

    def fake_hash_gate(**kwargs):
        captured.update(kwargs)
        return (
            torch.ones(3, 2, dtype=torch.float32),
            torch.zeros(3, 2, dtype=torch.int32),
            torch.arange(3, dtype=torch.int32),
        )

    monkeypatch.setattr(torch.ops._C_ascend, "moe_gating_top_k_hash", fake_hash_gate, raising=False)

    if moe_comm_type == MoECommType.ALLGATHER:
        prepare_finalize = SimpleNamespace(all_gather_input_id_with_dp_group=MagicMock(return_value=aligned_input_ids))
        moe_comm_method = SimpleNamespace(prepare_finalize=prepare_finalize)
    else:
        prepare_finalize = None
        moe_comm_method = SimpleNamespace(pad_and_split_input_ids=MagicMock(return_value=aligned_input_ids))

    forward_context = SimpleNamespace(
        input_ids=raw_input_ids,
        moe_comm_type=moe_comm_type,
        moe_comm_method=moe_comm_method,
        flash_comm_v1_enabled=False,
    )
    monkeypatch.setattr(experts_selector_module, "get_forward_context", lambda: forward_context)

    experts_selector_module._select_experts_with_fusion_ops(
        hidden_states=torch.randn(3, 32),
        router_logits=torch.randn(3, 16),
        top_k=2,
        use_grouped_topk=True,
        renormalize=False,
        e_score_correction_bias=None,
        topk_group=2,
        num_expert_group=4,
        scoring_func="sqrtsoftplus",
        tid2eid=tid2eid,
    )

    if moe_comm_type == MoECommType.ALLGATHER:
        assert prepare_finalize is not None
        prepare_finalize.all_gather_input_id_with_dp_group.assert_called_once_with(raw_input_ids)
    else:
        moe_comm_method.pad_and_split_input_ids.assert_called_once_with(raw_input_ids)

    expected_input_ids = torch.tensor([9, 0, 7], dtype=torch.int64)
    assert torch.equal(captured["input_ids"], expected_input_ids)
    assert torch.equal(captured["tid2eid"], tid2eid.to(torch.int32))


def test_hash_route_recomputes_weights_from_router_logits(monkeypatch):
    raw_input_ids = torch.tensor([7, 8, 9, 10], dtype=torch.int64)
    aligned_input_ids = torch.tensor([9, -1], dtype=torch.int64)
    tid2eid = torch.arange(16, dtype=torch.int64).view(4, 4)
    router_logits = torch.tensor(
        [[-4.0, -1.0, 0.5, 2.0], [3.0, -2.0, 1.0, -0.5]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)

    def fake_hash_gate(**kwargs):
        bad_weights = torch.full((2, 2), float("nan"), dtype=torch.float32)
        return bad_weights, topk_ids, torch.arange(2, dtype=torch.int32)

    monkeypatch.setattr(torch.ops._C_ascend, "moe_gating_top_k_hash", fake_hash_gate, raising=False)

    moe_comm_method = SimpleNamespace(pad_and_split_input_ids=MagicMock(return_value=aligned_input_ids))
    forward_context = SimpleNamespace(
        input_ids=raw_input_ids,
        moe_comm_type=MoECommType.MC2,
        moe_comm_method=moe_comm_method,
        flash_comm_v1_enabled=False,
    )
    monkeypatch.setattr(experts_selector_module, "get_forward_context", lambda: forward_context)

    topk_weights, actual_topk_ids = experts_selector_module._select_experts_with_fusion_ops(
        hidden_states=torch.randn(2, 32),
        router_logits=router_logits,
        top_k=2,
        use_grouped_topk=True,
        renormalize=True,
        e_score_correction_bias=None,
        topk_group=1,
        num_expert_group=1,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=2.5,
        tid2eid=tid2eid,
    )

    expected = torch.nn.functional.softplus(router_logits).sqrt().gather(1, topk_ids.to(torch.int64))
    expected = expected / expected.sum(dim=-1, keepdim=True)
    expected = expected * 2.5
    assert torch.equal(actual_topk_ids, topk_ids)
    assert torch.isfinite(topk_weights).all()
    torch.testing.assert_close(topk_weights, expected)
