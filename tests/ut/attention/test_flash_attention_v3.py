# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import pytest
import torch
from vllm.config import CUDAGraphMode


def make_decode_batch(fa3_backend, *, is_draft, draft_eager):
    impl = fa3_backend.AscendFlashAttentionImpl(8, 192, 192**-0.5, num_kv_heads=2)
    config = NS(
        speculative_config=NS(method="dflash", num_speculative_tokens=3, enforce_eager=draft_eager),
        scheduler_config=NS(max_num_seqs=2),
        compilation_config=NS(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[8],
            static_forward_context={"attn": NS(impl=impl, is_draft_layer=is_draft)},
        ),
    )
    builder = fa3_backend.AscendFlashAttentionMetadataBuilder(
        NS(dtype=torch.bfloat16, block_size=128), ["attn"], config, torch.device("cpu")
    )
    query_len = 3 if is_draft else 4
    offsets = torch.arange(3, dtype=torch.int32) * query_len
    common = NS(
        num_reqs=2,
        query_start_loc_cpu=offsets,
        query_start_loc=offsets.clone(),
        seq_lens=torch.tensor([21, 31], dtype=torch.int32),
        block_table_tensor=torch.zeros(2, 1, dtype=torch.int32),
        slot_mapping=torch.arange(2 * query_len, dtype=torch.int32),
        num_input_tokens=2 * query_len,
        num_actual_tokens=2 * query_len,
        max_query_len=query_len,
        causal=not is_draft,
    )
    return builder, common


@pytest.mark.parametrize("draft_eager", [False, True])
def test_padded_draft_bucket_outside_target_capture_sizes(fa3_backend, draft_eager):
    builder, common = make_decode_batch(fa3_backend, is_draft=True, draft_eager=draft_eager)
    capture = builder.build(0, common)
    first_tiling = next(iter(capture.scheduler_metadata.values()))

    # The six-token bucket now contains just one three-token draft request.
    common.query_start_loc_cpu[-1] = 3
    common.query_start_loc.copy_(common.query_start_loc_cpu)
    metadata = builder.build(0, common)
    scheduler_args = fa3_backend.get_scheduler_metadata.call_args.kwargs
    assert scheduler_args["batch_size"] == 1
    assert scheduler_args["cache_seqlens"].data_ptr() == common.seq_lens.data_ptr()
    next_tiling = next(iter(metadata.scheduler_metadata.values()))
    assert (next_tiling.data_ptr() == first_tiling.data_ptr()) is not draft_eager


def test_target_reuses_captured_tiling_after_rejection_with_eager_draft(fa3_backend):
    builder, common = make_decode_batch(fa3_backend, is_draft=False, draft_eager=True)
    capture = builder.build_for_graph_capture(common)
    captured_tiling = next(iter(capture.scheduler_metadata.values()))

    # The second row is a graph dummy. The first request has two tokens after rejection.
    common.query_start_loc_cpu.copy_(torch.tensor([0, 2, 4]))
    common.query_start_loc.copy_(common.query_start_loc_cpu)
    common.num_actual_tokens = common.max_query_len = 2
    replay = builder.build(0, common)
    assert fa3_backend.get_scheduler_metadata.call_args.kwargs["batch_size"] == 1
    assert replay.max_query_len == 4
    assert next(iter(replay.scheduler_metadata.values())).data_ptr() == captured_tiling.data_ptr()
