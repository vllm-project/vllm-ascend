# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vllm_ascend._310p.fused_moe.grouped_topk_router import AscendGroupedTopKRouter310
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.router.router_factory import create_ascend_fused_moe_router


def _patch_allgather_context(monkeypatch, input_ids: torch.Tensor, prepared_ids: torch.Tensor) -> None:
    prepare_finalize = SimpleNamespace(all_gather_input_id_with_dp_group=lambda _: prepared_ids)
    context = SimpleNamespace(
        input_ids=input_ids,
        moe_comm_type=MoECommType.ALLGATHER,
        moe_comm_method=SimpleNamespace(prepare_finalize=prepare_finalize),
        flash_comm_v1_enabled=False,
    )
    monkeypatch.setattr(
        "vllm_ascend._310p.fused_moe.grouped_topk_router.get_forward_context",
        lambda: context,
    )


def test_hash_router_uses_token_table_and_router_scores(monkeypatch) -> None:
    hidden_states = torch.zeros(3, 4, dtype=torch.float16)
    router_logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 3.0, 2.0, 1.0],
            [-1.0, 0.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    tid2eid = torch.tensor(
        [
            [3, 1],
            [0, 2],
            [1, 3],
            [2, 0],
        ],
        dtype=torch.int32,
    )
    input_ids = torch.tensor([0, 2, 3], dtype=torch.int64)
    _patch_allgather_context(monkeypatch, input_ids, input_ids)
    router = AscendGroupedTopKRouter310(
        top_k=2,
        global_num_experts=4,
        num_expert_group=1,
        topk_group=1,
        use_grouped_topk=True,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        tid2eid=tid2eid,
    )

    weights, ids = router._select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        input_ids=input_ids,
    )

    expected_ids = tid2eid[input_ids]
    expected_weights = F.softplus(router_logits).sqrt().gather(1, expected_ids.long())
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = expected_weights * 1.5

    torch.testing.assert_close(ids, expected_ids)
    torch.testing.assert_close(weights.float(), expected_weights, rtol=2e-3, atol=2e-3)


def test_hash_router_aligns_input_ids_with_allgather_rows(monkeypatch) -> None:
    input_ids = torch.tensor([0, 1], dtype=torch.int64)
    prepared_ids = torch.tensor([0, 2, 3, 1], dtype=torch.int64)
    _patch_allgather_context(monkeypatch, input_ids, prepared_ids)
    tid2eid = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int32)
    router_logits = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    router = AscendGroupedTopKRouter310(
        top_k=2,
        global_num_experts=4,
        num_expert_group=1,
        topk_group=1,
        use_grouped_topk=True,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        tid2eid=tid2eid,
    )

    _, ids = router._select_experts(
        hidden_states=torch.zeros(4, 4, dtype=torch.float16),
        router_logits=router_logits,
        input_ids=input_ids,
    )

    torch.testing.assert_close(ids, tid2eid[prepared_ids])


def test_router_factory_passes_hash_table(monkeypatch) -> None:
    monkeypatch.setattr("vllm_ascend.ops.fused_moe.router.router_factory.is_310p", lambda: True)
    tid2eid = torch.zeros(16, 2, dtype=torch.int32)

    router = create_ascend_fused_moe_router(
        top_k=2,
        global_num_experts=8,
        renormalize=True,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sqrtsoftplus",
        tid2eid=tid2eid,
    )

    assert isinstance(router, AscendGroupedTopKRouter310)
    assert router.tid2eid is tid2eid
