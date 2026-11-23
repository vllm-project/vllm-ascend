# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import PropertyMock, patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV


def _make_dispatcher() -> TokenDispatcherWithAll2AllV:
    dispatcher = object.__new__(TokenDispatcherWithAll2AllV)
    dispatcher.top_k = 2
    dispatcher.num_experts = 4
    dispatcher.num_local_experts = 2
    dispatcher.lora_context = None
    dispatcher.local_expert_indices = [0, 1]
    dispatcher.expert_ids_per_ep_rank = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="npu")
    dispatcher._input_splits_staging = None
    dispatcher._output_splits_staging = None
    dispatcher._splits_copy_done = None
    return dispatcher


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an available NPU")
def test_all2allv_split_staging_reuses_pinned_buffers_across_zero_receive_dispatch():
    torch.npu.set_device(0)
    dispatcher = _make_dispatcher()
    hidden_states = torch.randn(8, 16, dtype=torch.bfloat16, device="npu")
    topk_ids = torch.zeros((8, 2), dtype=torch.int64, device="npu")

    normal_global_counts = torch.tensor(
        [[1, 1, 0, 0], [1, 1, 0, 0]],
        dtype=torch.int64,
        device="npu",
    )
    zero_global_counts = torch.zeros((2, 4), dtype=torch.int64, device="npu")

    def fake_permute(tokens, indices, num_out_tokens=None):
        return tokens, torch.empty((0,), dtype=torch.int64, device=tokens.device)

    with (
        patch.object(
            TokenDispatcherWithAll2AllV,
            "ep_group",
            new_callable=PropertyMock,
            return_value=object(),
        ),
        patch.object(
            TokenDispatcherWithAll2AllV,
            "ep_size",
            new_callable=PropertyMock,
            return_value=2,
        ),
        patch(
            "vllm_ascend.ops.fused_moe.token_dispatcher.gather_from_sequence_parallel_region",
            side_effect=[normal_global_counts, zero_global_counts],
        ),
        patch(
            "vllm_ascend.ops.fused_moe.token_dispatcher.torch.histc",
            return_value=torch.tensor([2, 2, 2, 2], dtype=torch.int64, device="npu"),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.token_dispatcher.torch_npu.npu_moe_token_permute",
            side_effect=fake_permute,
        ),
    ):
        first = dispatcher._dispatch_preprocess(hidden_states, topk_ids)
        first_input_splits = first[3].copy()
        first_output_splits = first[4].copy()
        first_expert_indices = first[5].clone()
        input_staging = dispatcher._input_splits_staging
        output_staging = dispatcher._output_splits_staging
        split_event = dispatcher._splits_copy_done

        second = dispatcher._dispatch_preprocess(hidden_states, topk_ids)

    assert input_staging is not None and input_staging.is_pinned()
    assert output_staging is not None and output_staging.is_pinned()
    assert dispatcher._input_splits_staging is input_staging
    assert dispatcher._output_splits_staging is output_staging
    assert dispatcher._splits_copy_done is split_event

    assert first_input_splits.tolist() == [4, 4]
    assert first_output_splits.tolist() == [2, 2]
    assert first_expert_indices.numel() == 4
    assert first[3].tolist() == [4, 4]
    assert second[3].tolist() == [4, 4]
    assert second[4].tolist() == [0, 0]
    assert second[5].numel() == 0
