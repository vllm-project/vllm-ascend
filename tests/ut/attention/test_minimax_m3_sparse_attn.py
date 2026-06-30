# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Correctness tests for MiniMax M3 sparse-attention Triton kernels on Ascend.

Covers ``msa_m3_triton`` index top-k and block-sparse attention operators.
Test cases are adapted from
``reference/vllm_cp/tests/kernels/attention/test_minimax_m3.py``.
"""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.attention.msa_m3_triton import (
    SPARSE_BLOCK_SIZE,
    _split_triton_main_kv_cache,
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
CUDA_OR_ROCM_AVAILABLE = torch.cuda.is_available() or torch.version.hip is not None
if NPU_AVAILABLE:
    DEVICE = "npu"
elif CUDA_OR_ROCM_AVAILABLE:
    DEVICE = "cuda"
else:
    pytest.skip(
        "MiniMax M3 sparse-attention Triton tests require NPU, CUDA, or ROCm.",
        allow_module_level=True,
    )

BLOCK_SIZE = SPARSE_BLOCK_SIZE
NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
DTYPE = torch.bfloat16
TOPK = 16
SM_SCALE = HEAD_DIM**-0.5
_SPARSE_MEAN_ATOL = 2.5e-4
_SPARSE_MAX_ATOL = 1.7e-2
# MiniMax-M3 production sparse_attention_config (w8a8 checkpoint).
PRODUCTION_SPARSE_TOPK = 16
PRODUCTION_INDEX_HEAD_DIM = 128
PRODUCTION_LOCAL_BLOCKS = 1
PRODUCTION_INIT_BLOCKS = 0
# TP=8 over 4 index/KV heads -> 1 head per rank during online serve.
PRODUCTION_NUM_IDX_HEADS_TP8 = 1


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


TOPK_COMPUTE_MIN_TILE = 16
TOPK_SELECTION_TILE = 128


def _topk_compute_width(topk: int) -> int:
    """Mirror msa_m3_triton pairwise top-k compute tile width."""
    return max(TOPK_COMPUTE_MIN_TILE, _next_power_of_2(topk))


def _topk_select_width(topk: int) -> int:
    """Mirror msa_m3_triton local top-k selection tile width."""
    return max(TOPK_SELECTION_TILE, _topk_compute_width(topk))


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    # vLLM cleanup calls torch.accelerator.empty_cache(), invalid on NPU.
    return False


def _synchronize() -> None:
    if DEVICE == "npu":
        torch.npu.synchronize()
    else:
        torch.accelerator.synchronize()


def _gather_index_k(
    index_kv_cache: torch.Tensor,
    pages: torch.Tensor,
) -> torch.Tensor:
    if index_kv_cache.ndim == 3:
        return index_kv_cache[pages]
    if index_kv_cache.ndim == 4:
        return index_kv_cache[pages, :, 0, :]
    if index_kv_cache.ndim == 5 and index_kv_cache.shape[0] == 2:
        return index_kv_cache[0, pages, :, 0, :]
    raise ValueError(f"Unexpected index cache ndim: {index_kv_cache.ndim}")


def _allocate_index_kv_cache(
    num_pages: int,
    head_dim: int,
    layout: str,
    *,
    device: str = DEVICE,
) -> torch.Tensor:
    if layout == "3d":
        return torch.empty(num_pages, BLOCK_SIZE, head_dim, device=device)
    if layout == "4d":
        return torch.empty(num_pages, BLOCK_SIZE, 1, head_dim, device=device)
    if layout == "5d":
        return torch.empty(2, num_pages, BLOCK_SIZE, 1, head_dim, device=device)
    raise ValueError(f"Unknown index cache layout: {layout}")


def _fill_index_block(
    index_kv_cache: torch.Tensor,
    page: int,
    value: float,
) -> None:
    if index_kv_cache.ndim == 3:
        index_kv_cache[page].fill_(value)
    elif index_kv_cache.ndim == 4:
        index_kv_cache[page, :, 0, :].fill_(value)
    else:
        index_kv_cache[0, page, :, 0, :].fill_(value)


def _reference_index_topk(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    q_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: float,
) -> torch.Tensor:
    total_q, num_idx_heads, _ = idx_q.shape
    out = torch.full(
        (num_idx_heads, total_q, topk), -1, device=idx_q.device, dtype=torch.int32
    )

    q_start = 0
    for req_id, (q_len, seq_len, prefix_len) in enumerate(
        zip(q_lens.tolist(), seq_lens.tolist(), prefix_lens.tolist())
    ):
        q_end = q_start + q_len
        q = idx_q[q_start:q_end]
        num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        pages = block_table[req_id, :num_blocks]
        k = _gather_index_k(index_kv_cache, pages).reshape(num_blocks * BLOCK_SIZE, -1)
        score = torch.einsum("qhd,kd->hqk", q.float(), k.float()) * sm_scale

        q_pos = prefix_len + torch.arange(q_len, device=idx_q.device)
        k_pos = torch.arange(k.shape[0], device=idx_q.device)
        score.masked_fill_(k_pos[None, :] > q_pos[:, None], -float("inf"))
        score = score.reshape(num_idx_heads, q_len, num_blocks, BLOCK_SIZE)
        score_tensor = score.max(dim=3).values

        valid_blocks = (q_pos + BLOCK_SIZE) // BLOCK_SIZE
        for local_q, num_valid_blocks in enumerate(valid_blocks.tolist()):
            end = min(init_blocks, num_valid_blocks)
            score_tensor[:, local_q, :end] = 1e30
            start = max(0, num_valid_blocks - local_blocks)
            score_tensor[:, local_q, start:num_valid_blocks] = 1e29

            picked = min(topk, num_valid_blocks)
            topk_idx = score_tensor[:, local_q].topk(picked, dim=1).indices
            out[:, q_start + local_q, :picked] = topk_idx
        q_start = q_end

    return out


def _assert_topk_indices_equal_unordered(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    assert actual.shape == expected.shape
    actual_flat = actual.cpu().reshape(-1, actual.shape[-1]).tolist()
    expected_flat = expected.cpu().reshape(-1, expected.shape[-1]).tolist()
    for actual_row, expected_row in zip(actual_flat, expected_flat):
        assert set(actual_row) == set(expected_row)


@pytest.mark.parametrize("index_layout", ["3d", "4d", "5d"])
def test_prefill_index_topk_correctness(index_layout: str) -> None:
    topk = 6
    init_blocks = 0
    local_blocks = 1
    num_idx_heads = 2
    head_dim = 16
    q_lens = torch.tensor((4, 3), device=DEVICE, dtype=torch.int32)
    prefix_lens = torch.tensor((0, 1024), device=DEVICE, dtype=torch.int32)
    seq_lens = prefix_lens + q_lens
    batch = q_lens.numel()
    max_seq_len = int(seq_lens.max().item())
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = batch * max_blocks

    cu_seqlens = torch.zeros(batch + 1, device=DEVICE, dtype=torch.int32)
    cu_seqlens[1:] = q_lens.cumsum(0)
    block_table = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32).reshape(
        batch, max_blocks
    )
    idx_q = torch.ones(q_lens.sum().item(), num_idx_heads, head_dim, device=DEVICE)
    index_kv_cache = _allocate_index_kv_cache(num_pages, head_dim, index_layout)
    for req_id in range(batch):
        for block_id in range(max_blocks):
            page = int(block_table[req_id, block_id].item())
            _fill_index_block(index_kv_cache, page, block_id + 1)

    sm_scale = head_dim**-0.5
    max_query_len = int(q_lens.max().item())
    score = minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        num_kv_heads=num_idx_heads,
        sm_scale=sm_scale,
    )
    actual = minimax_m3_index_topk(
        score,
        cu_seqlens,
        prefix_lens,
        max_query_len=max_query_len,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    expected = _reference_index_topk(
        idx_q,
        index_kv_cache,
        block_table,
        q_lens,
        seq_lens,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        sm_scale,
    )
    _synchronize()
    _assert_topk_indices_equal_unordered(actual, expected)


@pytest.mark.parametrize("index_layout", ["3d"])
@pytest.mark.parametrize("num_reqs", [1, 2])
def test_prefill_index_topk_production_shape(
    index_layout: str,
    num_reqs: int,
) -> None:
    """Reproduce online prefill shapes missed by the topk=6 correctness test.

    Production MiniMax-M3 uses sparse_topk_blocks=16 with TP=8 (1 index head
    per rank) and index_head_dim=128. The argmax top-k path compiles with
    BLOCK_SIZE_T=16 and BLOCK_SIZE_K=128; ``test_prefill_index_topk_correctness``
    keeps topk=6 (BLOCK_SIZE_T=16 via min tile), so it never exercises topk=16.
    """
    topk = PRODUCTION_SPARSE_TOPK
    init_blocks = PRODUCTION_INIT_BLOCKS
    local_blocks = PRODUCTION_LOCAL_BLOCKS
    num_idx_heads = PRODUCTION_NUM_IDX_HEADS_TP8
    head_dim = PRODUCTION_INDEX_HEAD_DIM

    topk_width = _topk_compute_width(topk)
    select_width = _topk_select_width(topk)
    assert topk_width == 16, (
        f"expected production prefill BLOCK_SIZE_T=16, got {topk_width}"
    )
    assert select_width == 128, (
        f"expected production prefill BLOCK_SIZE_K=128, got {select_width}"
    )

    if num_reqs == 1:
        q_lens = torch.tensor((64,), device=DEVICE, dtype=torch.int32)
        prefix_lens = torch.tensor((4096,), device=DEVICE, dtype=torch.int32)
    else:
        q_lens = torch.tensor((64, 32), device=DEVICE, dtype=torch.int32)
        prefix_lens = torch.tensor((4096, 8192), device=DEVICE, dtype=torch.int32)
    seq_lens = prefix_lens + q_lens
    batch = q_lens.numel()
    max_seq_len = int(seq_lens.max().item())
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = batch * max_blocks

    cu_seqlens = torch.zeros(batch + 1, device=DEVICE, dtype=torch.int32)
    cu_seqlens[1:] = q_lens.cumsum(0)
    block_table = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32).reshape(
        batch, max_blocks
    )
    idx_q = torch.randn(q_lens.sum().item(), num_idx_heads, head_dim, device=DEVICE)
    index_kv_cache = _allocate_index_kv_cache(num_pages, head_dim, index_layout)
    for req_id in range(batch):
        for block_id in range(max_blocks):
            page = int(block_table[req_id, block_id].item())
            _fill_index_block(index_kv_cache, page, block_id + 1)

    sm_scale = head_dim**-0.5
    max_query_len = int(q_lens.max().item())
    score = minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        num_kv_heads=num_idx_heads,
        sm_scale=sm_scale,
    )
    actual = minimax_m3_index_topk(
        score,
        cu_seqlens,
        prefix_lens,
        max_query_len=max_query_len,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    expected = _reference_index_topk(
        idx_q,
        index_kv_cache,
        block_table,
        q_lens,
        seq_lens,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        sm_scale,
    )
    _synchronize()
    _assert_topk_indices_equal_unordered(actual, expected)


@pytest.mark.parametrize("index_layout", ["3d", "4d", "5d"])
@pytest.mark.parametrize("decode_query_len", [1, 4])
@pytest.mark.parametrize("num_padded_reqs", [0, 2])
def test_decode_index_topk_correctness(
    index_layout: str,
    decode_query_len: int,
    num_padded_reqs: int,
) -> None:
    topk = 6
    init_blocks = 0
    local_blocks = 1
    num_idx_heads = 2
    head_dim = 16
    active_seq_lens = torch.tensor((7, 129, 1025), device=DEVICE, dtype=torch.int32)
    q_lens = torch.full_like(active_seq_lens, decode_query_len)
    prefix_lens = active_seq_lens - decode_query_len
    active_batch = active_seq_lens.numel()
    batch = active_batch + num_padded_reqs
    seq_lens = torch.cat(
        [
            active_seq_lens,
            torch.zeros(num_padded_reqs, device=DEVICE, dtype=torch.int32),
        ]
    )
    max_seq_len = int(active_seq_lens.max().item())
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = active_batch * max_blocks

    active_block_table = torch.randperm(
        num_pages, device=DEVICE, dtype=torch.int32
    ).reshape(active_batch, max_blocks)
    block_table = torch.zeros(batch, max_blocks, device=DEVICE, dtype=torch.int32)
    block_table[:active_batch] = active_block_table
    idx_q = torch.randn(
        batch * decode_query_len, num_idx_heads, head_dim, device=DEVICE
    )
    index_kv_cache = torch.randn(
        *_allocate_index_kv_cache(num_pages, head_dim, index_layout).shape,
        device=DEVICE,
    )

    actual = minimax_m3_index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len=max_seq_len,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        num_kv_heads=num_idx_heads,
        sm_scale=head_dim**-0.5,
        decode_query_len=decode_query_len,
    )
    expected = torch.full_like(actual, -1)
    active_tokens = active_batch * decode_query_len
    expected[:, :active_tokens] = _reference_index_topk(
        idx_q[:active_tokens],
        index_kv_cache,
        block_table[:active_batch],
        q_lens,
        active_seq_lens,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        head_dim**-0.5,
    )
    _synchronize()
    _assert_topk_indices_equal_unordered(actual, expected)


@pytest.mark.parametrize("index_layout", ["3d", "4d", "5d"])
@pytest.mark.parametrize("num_reqs", [1, 2])
def test_decode_index_topk_production_cudagraph_shape(
    index_layout: str,
    num_reqs: int,
) -> None:
    """Reproduce online cudagraph decode shapes with production topk=16.

    Online serve (max-num-seqs=2, FULL_DECODE_ONLY) profiles decode with
    total_q in {1, 2}, sparse_topk_blocks=16, and TP=8 (1 index head per rank).
    Exercises the pairwise hierarchical merge path with BLOCK_SIZE_T=16 on NPU.
    """
    decode_query_len = 1
    total_q = num_reqs * decode_query_len
    num_idx_heads = PRODUCTION_NUM_IDX_HEADS_TP8
    head_dim = PRODUCTION_INDEX_HEAD_DIM
    topk_width = _topk_compute_width(PRODUCTION_SPARSE_TOPK)
    select_width = _topk_select_width(PRODUCTION_SPARSE_TOPK)
    assert topk_width == 16, (
        f"expected production decode BLOCK_SIZE_T=16, got {topk_width}"
    )
    assert select_width == 128, (
        f"expected production decode BLOCK_SIZE_K=128, got {select_width}"
    )

    seq_lens = torch.full(
        (num_reqs,), 10240, device=DEVICE, dtype=torch.int32
    )
    max_seq_len = int(seq_lens.max().item())
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = num_reqs * max_blocks
    block_table = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32).reshape(
        num_reqs, max_blocks
    )
    idx_q = torch.randn(total_q, num_idx_heads, head_dim, device=DEVICE)
    index_kv_cache = torch.randn(
        *_allocate_index_kv_cache(num_pages, head_dim, index_layout).shape,
        device=DEVICE,
    )

    actual = minimax_m3_index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len=max_seq_len,
        topk=PRODUCTION_SPARSE_TOPK,
        init_blocks=PRODUCTION_INIT_BLOCKS,
        local_blocks=PRODUCTION_LOCAL_BLOCKS,
        num_kv_heads=num_idx_heads,
        sm_scale=head_dim**-0.5,
        decode_query_len=decode_query_len,
    )
    _synchronize()

    q_lens = torch.full((num_reqs,), decode_query_len, device=DEVICE, dtype=torch.int32)
    prefix_lens = seq_lens - q_lens
    expected = _reference_index_topk(
        idx_q,
        index_kv_cache,
        block_table,
        q_lens,
        seq_lens,
        prefix_lens,
        PRODUCTION_SPARSE_TOPK,
        PRODUCTION_INIT_BLOCKS,
        PRODUCTION_LOCAL_BLOCKS,
        head_dim**-0.5,
    )
    _assert_topk_indices_equal_unordered(actual, expected)


# ---------------------------------------------------------------------------
# Sparse attention Triton kernels (prefill / decode)
# ---------------------------------------------------------------------------


def _assert_sparse_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    error = (actual.float() - expected.float()).abs()
    assert error.mean().item() < _SPARSE_MEAN_ATOL, (
        f"mean error {error.mean().item():.6g} >= {_SPARSE_MEAN_ATOL}"
    )
    assert error.max().item() < _SPARSE_MAX_ATOL, (
        f"max error {error.max().item():.6g} >= {_SPARSE_MAX_ATOL}"
    )


def _allocate_main_kv_cache_fused(
    num_pages: int,
    *,
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> torch.Tensor:
    """Logical NHD main cache ``[num_blocks, 2, block, num_kv_heads, head_dim]``."""
    return torch.randn(
        num_pages, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype
    )


def _main_kv_cache_from_fused(
    kv_cache_fused: torch.Tensor,
    kv_format: str,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    key_cache, value_cache = kv_cache_fused.unbind(1)
    if kv_format == "fused":
        return kv_cache_fused
    if kv_format == "tuple":
        return key_cache, value_cache
    if kv_format == "ascend":
        return torch.stack((key_cache, value_cache), dim=0)
    raise ValueError(f"Unknown kv_format: {kv_format}")


def _reference_sparse_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    q_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(q, dtype=torch.float32)
    gqa_group_size = NUM_Q_HEADS // NUM_KV_HEADS
    q_start = 0
    for req_id, (q_len, seq_len, prefix_len) in enumerate(
        zip(q_lens.tolist(), seq_lens.tolist(), prefix_lens.tolist())
    ):
        q_end = q_start + q_len
        q_req = q[q_start:q_end]
        positions = torch.arange(seq_len, device=q.device)
        pages = block_table[req_id, positions // BLOCK_SIZE]
        rows = positions % BLOCK_SIZE
        k_req = kv_cache[pages, 0, rows]
        v_req = kv_cache[pages, 1, rows].float()

        q_pos = prefix_len + torch.arange(q_len, device=q.device)
        key_blocks = positions // BLOCK_SIZE
        causal_mask = positions.unsqueeze(0) <= q_pos.unsqueeze(1)

        for kv_head in range(NUM_KV_HEADS):
            selected = topk_idx[kv_head, q_start:q_end]
            selected_mask = (key_blocks[None, :, None] == selected[:, None, :]).any(-1)
            mask = causal_mask & selected_mask
            head_start = kv_head * gqa_group_size
            head_end = head_start + gqa_group_size

            q_heads = q_req[:, head_start:head_end].transpose(0, 1).float()
            k_head = k_req[:, kv_head].T.expand(gqa_group_size, -1, -1).float()
            scores = torch.bmm(q_heads, k_head)
            scores = scores.transpose(0, 1) * SM_SCALE
            probs = torch.softmax(
                scores.masked_fill(~mask[:, None, :], -float("inf")), -1
            )
            out[q_start:q_end, head_start:head_end] = torch.einsum(
                "qhk,kd->qhd", probs, v_req[:, kv_head]
            )
        q_start += q_len
    return out.to(q.dtype)


def _build_prefill_topk_idx(
    q_lens_t: torch.Tensor,
    prefix_lens: torch.Tensor,
    total_q: int,
) -> torch.Tensor:
    topk_idx = torch.full(
        (NUM_KV_HEADS, total_q, TOPK), -1, device=DEVICE, dtype=torch.int32
    )
    q_start = 0
    for q_len, prefix_len in zip(q_lens_t.tolist(), prefix_lens.tolist()):
        for local_q in range(q_len):
            current_block = (prefix_len + local_q) // BLOCK_SIZE
            older_blocks = torch.randperm(
                current_block, device=DEVICE, dtype=torch.int32
            )
            selected = torch.cat(
                [
                    torch.tensor([current_block], device=DEVICE, dtype=torch.int32),
                    older_blocks[: TOPK - 1],
                ]
            )
            topk_idx[:, q_start + local_q, : selected.numel()] = selected
        q_start += q_len
    return topk_idx


def _build_decode_inputs(
    seq_lens_list: tuple[int, ...],
    decode_query_len: int = 1,
    num_padded_reqs: int = 0,
):
    active_batch = len(seq_lens_list)
    batch = active_batch + num_padded_reqs
    pages_per_req = [(s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in seq_lens_list]
    max_blocks = max(pages_per_req)
    num_pages = sum(pages_per_req)
    physical_pages = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32)
    block_table = torch.zeros(batch, max_blocks, device=DEVICE, dtype=torch.int32)
    base_page = 0
    for req_id, num_req_pages in enumerate(pages_per_req):
        block_table[req_id, :num_req_pages] = physical_pages[
            base_page : base_page + num_req_pages
        ]
        base_page += num_req_pages

    seq_lens = torch.tensor(
        (*seq_lens_list, *([0] * num_padded_reqs)),
        device=DEVICE,
        dtype=torch.int32,
    )
    q = torch.randn(
        batch * decode_query_len, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE
    )

    topk_idx = torch.full(
        (NUM_KV_HEADS, batch * decode_query_len, TOPK),
        -1,
        device=DEVICE,
        dtype=torch.int32,
    )
    token_id = 0
    for seq_len in seq_lens_list:
        for local_q in range(decode_query_len):
            query_pos = seq_len - decode_query_len + local_q
            current_block = query_pos // BLOCK_SIZE
            older_blocks = torch.randperm(
                current_block, device=DEVICE, dtype=torch.int32
            )
            selected = torch.cat(
                [
                    torch.tensor([current_block], device=DEVICE, dtype=torch.int32),
                    older_blocks[: TOPK - 1],
                ]
            )
            topk_idx[:, token_id, : selected.numel()] = selected
            token_id += 1

    return q, block_table, seq_lens, topk_idx, num_pages


@pytest.mark.parametrize(
    ("q_lens", "kv_lens"),
    [
        ((129, 257), (129, 257)),
        ((65, 129, 257), (129, 257, 385)),
    ],
)
@pytest.mark.parametrize("kv_format", ["fused", "tuple", "ascend"])
def test_prefill_sparse_attention_correctness(
    q_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
    kv_format: str,
) -> None:
    assert len(q_lens) == len(kv_lens)
    assert all(kv_len >= q_len for q_len, kv_len in zip(q_lens, kv_lens))

    torch.manual_seed(0)
    batch = len(q_lens)
    pages_per_req = [(kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE for kv_len in kv_lens]
    max_blocks = max(pages_per_req)
    num_pages = sum(pages_per_req)
    physical_pages = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32)
    block_table = torch.zeros(batch, max_blocks, device=DEVICE, dtype=torch.int32)
    base_page = 0
    for req_id, num_req_pages in enumerate(pages_per_req):
        block_table[req_id, :num_req_pages] = physical_pages[
            base_page : base_page + num_req_pages
        ]
        base_page += num_req_pages

    q_lens_t = torch.tensor(q_lens, device=DEVICE, dtype=torch.int32)
    seq_lens = torch.tensor(kv_lens, device=DEVICE, dtype=torch.int32)
    prefix_lens = seq_lens - q_lens_t
    cu_seqlens = torch.zeros(batch + 1, device=DEVICE, dtype=torch.int32)
    cu_seqlens[1:] = q_lens_t.cumsum(0)
    total_q = sum(q_lens)
    max_seqlen_q = max(q_lens)

    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    kv_cache_fused = _allocate_main_kv_cache_fused(num_pages)
    kv_cache = _main_kv_cache_from_fused(kv_cache_fused, kv_format)
    topk_idx = _build_prefill_topk_idx(q_lens_t, prefix_lens, total_q)

    actual = torch.empty_like(q)
    minimax_m3_sparse_attn(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_seqlen_q,
        NUM_KV_HEADS,
        SM_SCALE,
        actual,
    )
    _synchronize()

    expected = _reference_sparse_attn(
        q,
        kv_cache_fused,
        topk_idx,
        block_table,
        q_lens_t,
        seq_lens,
        prefix_lens,
    )
    _assert_sparse_close(actual, expected)


def test_split_triton_main_kv_cache_formats_match() -> None:
    fused = _allocate_main_kv_cache_fused(3)
    key_cache, value_cache = fused.unbind(1)
    ascend = torch.stack((key_cache, value_cache), dim=0)

    k_fused, v_fused = _split_triton_main_kv_cache(fused)
    k_tuple, v_tuple = _split_triton_main_kv_cache((key_cache, value_cache))
    k_ascend, v_ascend = _split_triton_main_kv_cache(ascend)

    torch.testing.assert_close(k_fused, k_tuple)
    torch.testing.assert_close(v_fused, v_tuple)
    torch.testing.assert_close(k_fused, k_ascend)
    torch.testing.assert_close(v_fused, v_ascend)


@pytest.mark.parametrize(
    "seq_lens_list",
    [(130, 257), (129, 200, 384)],
)
@pytest.mark.parametrize("decode_query_len", [1, 4])
@pytest.mark.parametrize("num_padded_reqs", [0, 2])
@pytest.mark.parametrize("kv_format", ["fused", "tuple", "ascend"])
def test_decode_sparse_attention_correctness(
    seq_lens_list: tuple[int, ...],
    decode_query_len: int,
    num_padded_reqs: int,
    kv_format: str,
) -> None:
    torch.manual_seed(0)
    q, block_table, seq_lens, topk_idx, num_pages = _build_decode_inputs(
        seq_lens_list, decode_query_len, num_padded_reqs
    )
    kv_cache_fused = _allocate_main_kv_cache_fused(num_pages)
    kv_cache = _main_kv_cache_from_fused(kv_cache_fused, kv_format)

    actual = torch.empty_like(q)
    minimax_m3_sparse_attn_decode(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        NUM_KV_HEADS,
        SM_SCALE,
        actual,
        decode_query_len,
    )
    _synchronize()

    active_batch = len(seq_lens_list)
    active_tokens = active_batch * decode_query_len
    q_lens_t = torch.full(
        (active_batch,), decode_query_len, device=DEVICE, dtype=torch.int32
    )
    active_seq_lens = seq_lens[:active_batch]
    prefix_lens = active_seq_lens - q_lens_t
    expected = _reference_sparse_attn(
        q[:active_tokens],
        kv_cache_fused,
        topk_idx[:, :active_tokens],
        block_table[:active_batch],
        q_lens_t,
        active_seq_lens,
        prefix_lens,
    )
    _assert_sparse_close(actual[:active_tokens], expected)
