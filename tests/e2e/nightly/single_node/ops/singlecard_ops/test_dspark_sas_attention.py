# SPDX-License-Identifier: Apache-2.0

import math
import os

import pytest
import torch

from vllm_ascend.ops.dspark_attention import (
    dspark_attention_from_standard_cache,
    dspark_attention_from_standard_cache_sas,
)
from vllm_ascend.utils import enable_custom_op


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


@pytest.mark.skipif(not _has_npu(), reason="requires Ascend NPU")
@pytest.mark.skipif(
    os.getenv("VLLM_ASCEND_DSPARK_USE_PTA_REF", "0") == "1",
    reason="DSpark SAS fast path is disabled by PTA reference env",
)
@pytest.mark.parametrize(
    ("window_size", "cache_block_size", "seq_len", "pad_query_rows"),
    [
        (7, 16, 20, 0),
        (128, 128, 133, 0),
        (128, 128, 133, 1),
    ],
)
def test_dspark_standard_cache_sas_matches_pta_reference(window_size, cache_block_size, seq_len, pad_query_rows):
    enable_custom_op()
    torch.npu.set_device(0)
    torch.manual_seed(20260702)

    device = torch.device("npu:0")
    dtype = torch.bfloat16
    query_block_size = 5
    num_q_heads = 4
    num_kv_heads = 1
    head_dim = 512
    num_physical_blocks = 4

    # The small-window case uses non-zero start_pos; the real DSpark window case
    # verifies window_size + query_block_size > 128 with aligned sparse indices.
    active_positions = torch.arange(seq_len - query_block_size, seq_len, dtype=torch.int32, device=device)
    if pad_query_rows:
        positions = torch.cat(
            [
                active_positions,
                torch.zeros(pad_query_rows, dtype=torch.int32, device=device),
            ]
        )
    else:
        positions = active_positions
    query_start_loc = torch.tensor([0, query_block_size], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    block_table = torch.tensor([[2, 0]], dtype=torch.int32, device=device)
    active_slot_mapping = (
        block_table[0, active_positions.to(torch.long) // cache_block_size] * cache_block_size
        + active_positions % cache_block_size
    ).to(torch.int32)
    if pad_query_rows:
        slot_mapping = torch.cat(
            [
                active_slot_mapping,
                torch.full((pad_query_rows,), -1, dtype=torch.int32, device=device),
            ]
        )
        token_to_req_indices = torch.cat(
            [
                torch.zeros(query_block_size, dtype=torch.int32, device=device),
                torch.full((pad_query_rows,), -1, dtype=torch.int32, device=device),
            ]
        )
    else:
        slot_mapping = active_slot_mapping
        token_to_req_indices = None

    q = (torch.randn(query_block_size + pad_query_rows, num_q_heads, head_dim, device=device) * 0.02).to(dtype)
    standard_kv_cache = (
        torch.randn(num_physical_blocks, cache_block_size, num_kv_heads, head_dim, device=device) * 0.02
    ).to(dtype)
    attn_sink = torch.zeros(num_q_heads, dtype=torch.float32, device=device)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    expected = dspark_attention_from_standard_cache(
        q,
        standard_kv_cache,
        block_table,
        positions,
        slot_mapping,
        draft_kv=None,
        attn_sink=attn_sink,
        block_size=query_block_size,
        window_size=window_size,
        cache_block_size=cache_block_size,
        softmax_scale=softmax_scale,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        token_to_req_indices=token_to_req_indices,
    )
    actual = dspark_attention_from_standard_cache_sas(
        q,
        standard_kv_cache,
        block_table,
        positions,
        slot_mapping,
        attn_sink,
        query_block_size,
        window_size,
        cache_block_size,
        softmax_scale,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        token_to_req_indices=token_to_req_indices,
    )

    assert actual is not None
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=3e-2, rtol=3e-2)
    if pad_query_rows:
        torch.testing.assert_close(actual[-pad_query_rows:].cpu(), torch.zeros_like(actual[-pad_query_rows:]).cpu())
