# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather


def test_dp_padding_follows_gathered_rows():
    layer = PrepareAndFinalizeWithAllGather.__new__(PrepareAndFinalizeWithAllGather)
    layer.moe_config = SimpleNamespace(
        dp_size=2, pcp_size=1, dp_group=SimpleNamespace(all_gather=lambda x, dim: torch.cat([x, x], dim))
    )
    context = SimpleNamespace(
        eplb_heat_collection_status=True, mc2_mask=torch.tensor([True, True, False]), max_tokens_across_dp=4
    )
    with patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX", context):
        result = layer._prepare_with_dp_group(torch.ones(3, 8), torch.ones(3, 4), False)
    assert result.hidden_states.shape[0] == 8
    assert result.mc2_mask.tolist() == [True, True, False, False] * 2


def test_sequence_parallel_shards_mask_before_gather():
    layer = PrepareAndFinalizeWithAllGather.__new__(PrepareAndFinalizeWithAllGather)
    layer.moe_config = SimpleNamespace(is_sequence_parallel=True)
    context = SimpleNamespace(eplb_heat_collection_status=True, mc2_mask=torch.tensor([True, True, True, False]))
    with (
        patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX", context),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_tensor_model_parallel_rank", return_value=1),
        patch("torch.ops.vllm.maybe_all_gather_and_maybe_unpad", side_effect=lambda x: torch.cat([x, x])),
    ):
        result = layer._prepare_with_ep_group(torch.ones(2, 8), torch.ones(2, 4))
    assert result.mc2_mask.tolist() == [True, False, True, False]
    assert result.hidden_states.shape[0] == result.mc2_mask.shape[0]


def test_draft_valid_mask_can_differ_from_target_and_collection_can_be_disabled():
    layer = PrepareAndFinalizeWithAllGather.__new__(PrepareAndFinalizeWithAllGather)
    context = SimpleNamespace(eplb_heat_collection_status=True, mc2_mask=torch.tensor([True] * 5 + [False] * 3))
    with patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX", context):
        assert layer._eplb_valid_rows(8).sum() == 5
        context.mc2_mask.copy_(torch.tensor([True] * 2 + [False] * 6))
        assert layer._eplb_valid_rows(8).sum() == 2
        assert layer._eplb_valid_rows(0).numel() == 0
        context.eplb_heat_collection_status = False
        assert layer._eplb_valid_rows(8) is None
