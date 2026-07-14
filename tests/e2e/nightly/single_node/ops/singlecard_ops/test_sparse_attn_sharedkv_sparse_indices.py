# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

HEAD_DIM = 512
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
SPARSE_INDEX_WIDTH = 128
ORI_WIN_LEFT = 127
ORI_WIN_RIGHT = 0


def _reference_attention(
    query: torch.Tensor,
    kv_rows: list[torch.Tensor],
    sinks: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    output = torch.empty_like(query)
    for token_idx, token_kv in enumerate(kv_rows):
        key = token_kv.expand(-1, NUM_Q_HEADS, -1).float()
        scores = torch.einsum("hd,khd->hk", query[token_idx].float(), key) * softmax_scale
        sink = sinks.float().view(NUM_Q_HEADS, 1)
        scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
        exp_scores = torch.exp(scores - scores_max)
        probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max))
        output[token_idx] = torch.einsum("hk,khd->hd", probs, key).to(query.dtype)
    return output


def _make_metadata(
    query: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    ori_win_left: int,
    ori_win_right: int,
) -> torch.Tensor:
    return torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=NUM_Q_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=1,
        max_seqlen_q=query.shape[0],
        max_seqlen_kv=int(seqused_kv.max().item()),
        cmp_ratio=4,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device=str(query.device),
    )


def _run_operator(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    ori_sparse_indices: torch.Tensor | None,
    ori_win_left: int = ORI_WIN_LEFT,
    ori_win_right: int = ORI_WIN_RIGHT,
) -> torch.Tensor:
    metadata = _make_metadata(query, cu_seqlens_q, seqused_kv, ori_win_left, ori_win_right)
    return torch.ops._C_ascend.npu_sparse_attn_sharedkv(
        query,
        ori_kv=kv_cache,
        ori_sparse_indices=ori_sparse_indices,
        ori_block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=1.0 / math.sqrt(HEAD_DIM),
        cmp_ratio=4,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q="TND",
        layout_kv="PA_ND",
    )[0]


@torch.inference_mode()
def test_sparse_original_indices_match_explicit_slot_reference():
    torch.manual_seed(20260712)
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    cache_block_size = 16
    query_tokens = 2
    actual_kv_len = 64

    query = (torch.randn(query_tokens, NUM_Q_HEADS, HEAD_DIM, device=device) * 0.02).to(dtype)
    kv_cache = (torch.randn(6, cache_block_size, NUM_KV_HEADS, HEAD_DIM, device=device) * 0.02).to(dtype)
    block_table = torch.tensor([[4, 1, 3, 0]], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, query_tokens], dtype=torch.int32, device=device)
    seqused_kv = torch.tensor([actual_kv_len], dtype=torch.int32, device=device)
    sinks = torch.linspace(-0.2, 0.2, NUM_Q_HEADS, dtype=torch.float32, device=device)

    slot_rows = (
        (1, 19, 34, 17, 80, 47, 65),
        (63, 2, 48, 33, 79, 16),
    )
    sparse_indices = torch.full(
        (query_tokens, NUM_KV_HEADS, SPARSE_INDEX_WIDTH),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for token_idx, slots in enumerate(slot_rows):
        sparse_indices[token_idx, 0, : len(slots)] = torch.tensor(slots, dtype=torch.int32, device=device)

    actual = _run_operator(
        query,
        kv_cache,
        block_table,
        cu_seqlens_q,
        seqused_kv,
        sinks,
        sparse_indices,
    )
    flat_cache = kv_cache.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    kv_rows = [flat_cache[torch.tensor(slots, dtype=torch.long, device=device)] for slots in slot_rows]
    expected = _reference_attention(query, kv_rows, sinks, 1.0 / math.sqrt(HEAD_DIM))

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_sparse_original_indices_support_multiple_inner_loops():
    torch.manual_seed(20260714)
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    cache_block_size = 16
    actual_kv_len = 256
    index_width = 256
    visible_count = 140

    query = (torch.randn(1, NUM_Q_HEADS, HEAD_DIM, device=device) * 0.02).to(dtype)
    kv_cache = (torch.randn(20, cache_block_size, NUM_KV_HEADS, HEAD_DIM, device=device) * 0.02).to(dtype)
    block_table = torch.arange(16, dtype=torch.int32, device=device).reshape(1, -1)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused_kv = torch.tensor([actual_kv_len], dtype=torch.int32, device=device)
    sinks = torch.zeros(NUM_Q_HEADS, dtype=torch.float32, device=device)
    slots = (torch.arange(visible_count, dtype=torch.int32, device=device) * 37) % (
        kv_cache.shape[0] * cache_block_size
    )
    sparse_indices = torch.full(
        (1, NUM_KV_HEADS, index_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    sparse_indices[0, 0, :visible_count] = slots

    actual = _run_operator(
        query,
        kv_cache,
        block_table,
        cu_seqlens_q,
        seqused_kv,
        sinks,
        sparse_indices,
        ori_win_left=index_width - 1,
    )
    flat_cache = kv_cache.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    expected = _reference_attention(
        query,
        [flat_cache[slots.long()]],
        sinks,
        1.0 / math.sqrt(HEAD_DIM),
    )

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_sparse_original_indices_absent_preserves_swa_behavior():
    torch.manual_seed(20260713)
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    cache_block_size = 16
    actual_kv_len = 32

    query = (torch.randn(1, NUM_Q_HEADS, HEAD_DIM, device=device) * 0.02).to(dtype)
    kv_cache = (torch.randn(4, cache_block_size, NUM_KV_HEADS, HEAD_DIM, device=device) * 0.02).to(dtype)
    block_table = torch.tensor([[2, 0]], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused_kv = torch.tensor([actual_kv_len], dtype=torch.int32, device=device)
    sinks = torch.zeros(NUM_Q_HEADS, dtype=torch.float32, device=device)

    actual = _run_operator(query, kv_cache, block_table, cu_seqlens_q, seqused_kv, sinks, None)
    logical_positions = torch.arange(actual_kv_len, device=device)
    physical_blocks = block_table[0, logical_positions // cache_block_size].long()
    block_offsets = logical_positions % cache_block_size
    kv_rows = [kv_cache[physical_blocks, block_offsets]]
    expected = _reference_attention(query, kv_rows, sinks, 1.0 / math.sqrt(HEAD_DIM))

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=3e-2, rtol=3e-2)


@torch.inference_mode()
def test_sparse_original_indices_reject_unaligned_width():
    device = torch.device("npu:0")
    query = torch.zeros(1, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(1, 16, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    block_table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused_kv = torch.tensor([1], dtype=torch.int32, device=device)
    sinks = torch.zeros(NUM_Q_HEADS, dtype=torch.float32, device=device)
    sparse_indices = torch.zeros(1, NUM_KV_HEADS, 64, dtype=torch.int32, device=device)

    with pytest.raises(RuntimeError):
        _run_operator(query, kv_cache, block_table, cu_seqlens_q, seqused_kv, sinks, sparse_indices)
