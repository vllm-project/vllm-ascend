# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import routed_experts
from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter


@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
@pytest.mark.parametrize("group_count", [1, 2])
@pytest.mark.parametrize("renorm", [False, True])
def test_router_preserves_cann_logical_ids_and_weights_before_mapping(scoring, group_count, renorm):
    router = AscendFusedTopKRouter(
        top_k=6,
        global_num_experts=16,
        num_expert_group=group_count,
        topk_group=1,
        renormalize=renorm,
        scoring_func=scoring,
    )
    logical_ids = torch.tensor([[0, 2, 4, 6, 8, 10]], dtype=torch.int32)
    physical_ids = logical_ids + 16
    weights = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    routed_to_mapping = []

    def map_ids(ids):
        routed_to_mapping.append(ids)
        return physical_ids

    router._apply_eplb_mapping = map_ids
    with patch(
        "vllm_ascend.ops.fused_moe.router.fused_topk_router.DeviceOperator.moe_gating_top_k",
        return_value=(weights, logical_ids, None),
    ) as cann:
        actual_weights, actual_ids = router._select_experts(
            hidden_states=torch.zeros(1, 16),
            router_logits=torch.zeros(1, 16),
        )

    assert actual_weights is weights
    assert actual_ids is physical_ids
    assert routed_to_mapping == [logical_ids]
    assert cann.call_args.kwargs["group_count"] == group_count
    assert cann.call_args.kwargs["renorm"] == int(renorm)
    assert cann.call_args.kwargs["norm_type"] == (0 if scoring == "softmax" else 1)


@pytest.mark.parametrize(
    "comm,dp,pcp,sequence_parallel,mask,expected",
    [
        (MoECommType.ALLGATHER, 1, 1, False, None, 6),
        (MoECommType.ALLGATHER, 2, 1, False, None, 6),
        (MoECommType.ALLGATHER, 1, 2, False, None, 6),
        (MoECommType.ALLGATHER, 1, 1, True, None, 6),
        (MoECommType.MC2, 1, 1, False, [1, 1, 0, 0], 2),
        (MoECommType.FUSED_MC2, 1, 1, False, [1, 1, 1, 0], 3),
        (MoECommType.FUSED_MC2, 1, 1, False, None, 6),
        (MoECommType.MC2, 1, 1, False, None, 6),
        (MoECommType.ALLTOALL, 1, 1, False, None, 2),
        (MoECommType.ALLTOALL, 1, 1, True, None, 6),
    ],
)
def test_valid_prefix_tracks_prepared_router_rows(monkeypatch, comm, dp, pcp, sequence_parallel, mask, expected):
    state = SimpleNamespace(num_unpadded_tokens_tensors=[torch.tensor(6, dtype=torch.int32)])
    layer = SimpleNamespace(
        router=SimpleNamespace(eplb_state=state),
        moe_config=SimpleNamespace(dp_size=dp, pcp_size=pcp, is_sequence_parallel=sequence_parallel),
    )
    context = SimpleNamespace(
        moe_comm_type=comm,
        moe_comm_method=SimpleNamespace(prepare_finalize=SimpleNamespace(tp_rank=1, tp_size=2, num_tokens=8)),
    )
    monkeypatch.setattr(routed_experts, "_EXTRA_CTX", context)
    monkeypatch.setattr(routed_experts, "dbo_current_ubatch_id", lambda: 0)
    mc2_mask = torch.tensor(mask, dtype=torch.bool) if mask is not None else None

    prepared_rows = 8 if sequence_parallel else 4
    result = routed_experts._mapping_valid_token_prefix(layer, torch.zeros(prepared_rows, 16), mc2_mask)
    assert (None if result is None else int(result)) == expected


def test_alltoall_uneven_tp_split_uses_the_actual_rank_offset(monkeypatch):
    state = SimpleNamespace(num_unpadded_tokens_tensors=[torch.tensor(6, dtype=torch.int32)])
    layer = SimpleNamespace(
        router=SimpleNamespace(eplb_state=state),
        moe_config=SimpleNamespace(dp_size=1, pcp_size=1, is_sequence_parallel=False),
    )
    context = SimpleNamespace(
        moe_comm_type=MoECommType.ALLTOALL,
        moe_comm_method=SimpleNamespace(prepare_finalize=SimpleNamespace(tp_rank=2, tp_size=8, num_tokens=10)),
    )
    monkeypatch.setattr(routed_experts, "_EXTRA_CTX", context)
    monkeypatch.setattr(routed_experts, "dbo_current_ubatch_id", lambda: 0)

    result = routed_experts._mapping_valid_token_prefix(layer, torch.zeros(1, 16), None)
    assert int(result) == 1  # Rank 2 owns row 4; the old rank * local_rows formula gave row 2.
