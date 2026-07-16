# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Correctness tests for MiniMax M3 sparse-attention kernels on Ascend.

Covers ``msa_m3_triton`` index top-k operators, ``msa_m3_npu`` block-sparse
attention (``npu_sparse_attention_score``), and optionally compares against the
Triton sparse-attention reference.

Default sparse-attention backend is Triton. Select the NPU op with::

    pytest tests/ut/attention/test_minimax_m3_sparse_attn.py --msa-m3-sparse-backend=torch_npu

Test cases are adapted from
``reference/vllm_cp/tests/kernels/attention/test_minimax_m3.py`` and
``csrc/attention/sparse_attention_score/tests/test_bf16.py``.
"""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pytest
import torch

from vllm_ascend.attention.msa_m3_npu import (
    minimax_m3_sparse_attn,
)
from vllm_ascend.attention.msa_m3_triton import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from vllm_ascend.attention.msa_m3_triton import (
    minimax_m3_sparse_attn as minimax_m3_sparse_attn_triton,
)
from vllm_ascend.attention.msa_m3_triton import (
    minimax_m3_sparse_attn_decode as minimax_m3_sparse_attn_decode_triton,
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
SparseAttnBackend = Literal["triton", "torch_npu"]
_NPU_SPARSE_OP_REGISTERED = False
# MiniMax-M3 production sparse_attention_config (w8a8 checkpoint).
PRODUCTION_SPARSE_TOPK = 16
PRODUCTION_INDEX_HEAD_DIM = 128
PRODUCTION_LOCAL_BLOCKS = 1
PRODUCTION_INIT_BLOCKS = 0
# Per-rank heads for TP>=4 (4 KV heads sharded across TP ranks).
PRODUCTION_NUM_IDX_HEADS_PER_RANK = 1
PRODUCTION_NUM_KV_HEADS_PER_RANK = 1
PRODUCTION_TP8_NUM_Q_HEADS = 8
PRODUCTION_TP16_NUM_Q_HEADS = 4
# ``num_nextn_predict_layers=1`` -> two query tokens per decode step.
PRODUCTION_MTP_DECODE_QUERY_LEN = 2
# Matches ``AscendMiniMaxM3SparseBackend.get_kv_cache_shape`` on Ascend.
PRODUCTION_KV_CACHE_LAYOUT = "ascend"
# Matches ``--max-model-len`` in online w8a8 serve scripts (10240 tokens).
PRODUCTION_MAX_SEQ_LEN = 10240
PRODUCTION_LONG_PREFILL_QUERY_LEN = 64


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


@pytest.fixture(scope="session")
def msa_m3_sparse_backend(request: pytest.FixtureRequest) -> SparseAttnBackend:
    backend: SparseAttnBackend = request.config.getoption("--msa-m3-sparse-backend")
    if backend == "torch_npu":
        if not NPU_AVAILABLE:
            pytest.skip("torch_npu sparse backend requires NPU.")
        _ensure_npu_sparse_attention_score_op()
    return backend


@pytest.fixture
def msa_m3_sparse_backend_triton_only(
    msa_m3_sparse_backend: SparseAttnBackend,
) -> SparseAttnBackend:
    if msa_m3_sparse_backend != "triton":
        pytest.skip("index / boundary tests only run with triton backend.")
    return msa_m3_sparse_backend


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    # vLLM cleanup calls torch.accelerator.empty_cache(), invalid on NPU.
    return False


def _ensure_npu_sparse_attention_score_op() -> None:
    """Ensure ``torch.ops._C_ascend.npu_sparse_attention_score`` is available."""
    global _NPU_SPARSE_OP_REGISTERED
    if _NPU_SPARSE_OP_REGISTERED:
        return

    from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

    bootstrap_custom_op_env(include_vendor_lib=True)
    if not enable_custom_op():
        pytest.skip("vllm-ascend custom ops are disabled in this build.")

    torch.npu.set_device(0)
    torch.npu.synchronize()

    try:
        _ = torch.ops._C_ascend.npu_sparse_attention_score
    except AttributeError:
        pytest.skip(
            "torch.ops._C_ascend.npu_sparse_attention_score is not available. "
            "Rebuild with: pip install -v --no-build-isolation -e ."
        )

    _NPU_SPARSE_OP_REGISTERED = True


def _sparse_tolerances(_backend: SparseAttnBackend) -> tuple[float, float]:
    return _SPARSE_MEAN_ATOL, _SPARSE_MAX_ATOL


def _run_prefill_sparse_attention(
    backend: SparseAttnBackend,
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv_cache_fused: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    q_lens_t: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
) -> None:
    del kv_cache_fused, q_lens_t
    if backend == "triton":
        minimax_m3_sparse_attn_triton(
            q,
            kv_cache,
            topk_idx,
            block_table,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            max_seqlen_q,
            num_kv_heads,
            sm_scale,
            output,
        )
        return

    minimax_m3_sparse_attn(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_seqlen_q,
        num_kv_heads,
        sm_scale,
        output,
        block_size=BLOCK_SIZE,
    )


def _run_decode_sparse_attention(
    backend: SparseAttnBackend,
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    kv_cache_fused: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    active_batch: int,
) -> None:
    del kv_cache_fused, active_batch
    if backend == "triton":
        minimax_m3_sparse_attn_decode_triton(
            q,
            kv_cache,
            topk_idx,
            block_table,
            seq_lens,
            num_kv_heads,
            sm_scale,
            output,
            decode_query_len,
        )
        return

    pytest.skip("MiniMax M3 NPU sparse attention currently supports prefill only; decode uses Triton.")


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
    out = torch.full((num_idx_heads, total_q, topk), -1, device=idx_q.device, dtype=torch.int32)

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


@pytest.mark.parametrize("index_layout", ["3d"])
def test_prefill_index_topk_correctness(
    msa_m3_sparse_backend_triton_only: SparseAttnBackend,
    index_layout: str,
) -> None:
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
    block_table = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32).reshape(batch, max_blocks)
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
    msa_m3_sparse_backend_triton_only: SparseAttnBackend,
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
    num_idx_heads = PRODUCTION_NUM_IDX_HEADS_PER_RANK
    head_dim = PRODUCTION_INDEX_HEAD_DIM

    topk_width = _topk_compute_width(topk)
    select_width = _topk_select_width(topk)
    assert topk_width == 16, f"expected production prefill BLOCK_SIZE_T=16, got {topk_width}"
    assert select_width == 128, f"expected production prefill BLOCK_SIZE_K=128, got {select_width}"

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
    block_table = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32).reshape(batch, max_blocks)
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


@pytest.mark.parametrize("index_layout", ["3d"])
def test_prefill_index_topk_production_long_sequence(
    msa_m3_sparse_backend_triton_only: SparseAttnBackend,
    index_layout: str,
) -> None:
    """Prefill index top-k at online max context length.

    Exercises the full 80-block KV table (10240 / 128) with production topk=16,
    which the shorter production-shape test (prefix <= 8192) does not reach.
    """
    topk = PRODUCTION_SPARSE_TOPK
    init_blocks = PRODUCTION_INIT_BLOCKS
    local_blocks = PRODUCTION_LOCAL_BLOCKS
    num_idx_heads = PRODUCTION_NUM_IDX_HEADS_PER_RANK
    head_dim = PRODUCTION_INDEX_HEAD_DIM
    q_len = PRODUCTION_LONG_PREFILL_QUERY_LEN
    prefix_len = PRODUCTION_MAX_SEQ_LEN - q_len

    q_lens = torch.tensor((q_len,), device=DEVICE, dtype=torch.int32)
    prefix_lens = torch.tensor((prefix_len,), device=DEVICE, dtype=torch.int32)
    seq_lens = prefix_lens + q_lens
    max_seq_len = int(seq_lens.max().item())
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    assert max_blocks == PRODUCTION_MAX_SEQ_LEN // BLOCK_SIZE
    num_pages = max_blocks

    cu_seqlens = torch.zeros(2, device=DEVICE, dtype=torch.int32)
    cu_seqlens[1] = q_len
    block_table = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32).reshape(1, max_blocks)
    idx_q = torch.randn(q_len, num_idx_heads, head_dim, device=DEVICE)
    index_kv_cache = _allocate_index_kv_cache(num_pages, head_dim, index_layout)
    for block_id in range(max_blocks):
        page = int(block_table[0, block_id].item())
        _fill_index_block(index_kv_cache, page, block_id + 1)

    sm_scale = head_dim**-0.5
    score = minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        max_query_len=q_len,
        max_seq_len=max_seq_len,
        num_kv_heads=num_idx_heads,
        sm_scale=sm_scale,
    )
    actual = minimax_m3_index_topk(
        score,
        cu_seqlens,
        prefix_lens,
        max_query_len=q_len,
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
@pytest.mark.parametrize("decode_query_len", [1, 4])
@pytest.mark.parametrize("num_padded_reqs", [0, 2])
def test_decode_index_topk_correctness(
    msa_m3_sparse_backend_triton_only: SparseAttnBackend,
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

    active_block_table = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32).reshape(active_batch, max_blocks)
    block_table = torch.zeros(batch, max_blocks, device=DEVICE, dtype=torch.int32)
    block_table[:active_batch] = active_block_table
    idx_q = torch.randn(batch * decode_query_len, num_idx_heads, head_dim, device=DEVICE)
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


@pytest.mark.parametrize("index_layout", ["3d"])
@pytest.mark.parametrize("decode_query_len", [1, PRODUCTION_MTP_DECODE_QUERY_LEN])
@pytest.mark.parametrize("num_reqs", [1, 2])
def test_decode_index_topk_production_cudagraph_shape(
    msa_m3_sparse_backend_triton_only: SparseAttnBackend,
    index_layout: str,
    decode_query_len: int,
    num_reqs: int,
) -> None:
    """Reproduce online cudagraph decode shapes with production topk=16.

    Online serve (max-num-seqs=2, FULL_DECODE_ONLY) profiles decode with
    total_q in {1, 2} for decode_query_len=1, or {2, 4} when MTP is enabled.
    Covers sparse_topk_blocks=16 with 1 index head per rank (TP>=4).
    """
    total_q = num_reqs * decode_query_len
    num_idx_heads = PRODUCTION_NUM_IDX_HEADS_PER_RANK
    head_dim = PRODUCTION_INDEX_HEAD_DIM
    topk_width = _topk_compute_width(PRODUCTION_SPARSE_TOPK)
    select_width = _topk_select_width(PRODUCTION_SPARSE_TOPK)
    assert topk_width == 16, f"expected production decode BLOCK_SIZE_T=16, got {topk_width}"
    assert select_width == 128, f"expected production decode BLOCK_SIZE_K=128, got {select_width}"

    seq_lens = torch.full((num_reqs,), 10240, device=DEVICE, dtype=torch.int32)
    max_seq_len = int(seq_lens.max().item())
    max_blocks = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = num_reqs * max_blocks
    block_table = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32).reshape(num_reqs, max_blocks)
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


def _assert_sparse_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    backend: SparseAttnBackend = "triton",
) -> None:
    mean_atol, max_atol = _sparse_tolerances(backend)
    error = (actual.float() - expected.float()).abs()
    assert error.mean().item() < mean_atol, f"mean error {error.mean().item():.6g} >= {mean_atol} (backend={backend})"
    assert error.max().item() < max_atol, f"max error {error.max().item():.6g} >= {max_atol} (backend={backend})"


def _allocate_main_kv_cache_fused(
    num_pages: int,
    *,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> torch.Tensor:
    """Logical NHD main cache ``[num_blocks, 2, block, num_kv_heads, head_dim]``."""
    return torch.randn(
        num_pages,
        2,
        BLOCK_SIZE,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )


def _main_kv_cache_for_kernel(
    kv_cache_fused: torch.Tensor,
    layout: str = PRODUCTION_KV_CACHE_LAYOUT,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    key_cache, value_cache = kv_cache_fused.unbind(1)
    if layout == "ascend":
        return torch.stack((key_cache, value_cache), dim=0)
    raise ValueError(f"Unsupported kv cache layout for Ascend tests: {layout}")


def _reference_sparse_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    q_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    *,
    num_q_heads: int = NUM_Q_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    sm_scale: float = SM_SCALE,
) -> torch.Tensor:
    out = torch.empty_like(q, dtype=torch.float32)
    gqa_group_size = num_q_heads // num_kv_heads
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

        for kv_head in range(num_kv_heads):
            selected = topk_idx[kv_head, q_start:q_end]
            selected_mask = (key_blocks[None, :, None] == selected[:, None, :]).any(-1)
            mask = causal_mask & selected_mask
            head_start = kv_head * gqa_group_size
            head_end = head_start + gqa_group_size

            q_heads = q_req[:, head_start:head_end].transpose(0, 1).float()
            k_head = k_req[:, kv_head].T.expand(gqa_group_size, -1, -1).float()
            scores = torch.bmm(q_heads, k_head)
            scores = scores.transpose(0, 1) * sm_scale
            probs = torch.softmax(scores.masked_fill(~mask[:, None, :], -float("inf")), -1)
            out[q_start:q_end, head_start:head_end] = torch.einsum("qhk,kd->qhd", probs, v_req[:, kv_head])
        q_start += q_len
    return out.to(q.dtype)


def _build_prefill_topk_idx(
    q_lens_t: torch.Tensor,
    prefix_lens: torch.Tensor,
    total_q: int,
    *,
    num_kv_heads: int = NUM_KV_HEADS,
    topk: int = TOPK,
) -> torch.Tensor:
    topk_idx = torch.full((num_kv_heads, total_q, topk), -1, device=DEVICE, dtype=torch.int32)
    q_start = 0
    for q_len, prefix_len in zip(q_lens_t.tolist(), prefix_lens.tolist()):
        for local_q in range(q_len):
            current_block = (prefix_len + local_q) // BLOCK_SIZE
            older_blocks = torch.randperm(current_block, device=DEVICE, dtype=torch.int32)
            selected = torch.cat(
                [
                    torch.tensor([current_block], device=DEVICE, dtype=torch.int32),
                    older_blocks[: topk - 1],
                ]
            )
            topk_idx[:, q_start + local_q, : selected.numel()] = selected
        q_start += q_len
    return topk_idx


def _build_decode_inputs(
    seq_lens_list: tuple[int, ...],
    decode_query_len: int = 1,
    num_padded_reqs: int = 0,
    *,
    num_q_heads: int = NUM_Q_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    topk: int = TOPK,
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
        block_table[req_id, :num_req_pages] = physical_pages[base_page : base_page + num_req_pages]
        base_page += num_req_pages

    seq_lens = torch.tensor(
        (*seq_lens_list, *([0] * num_padded_reqs)),
        device=DEVICE,
        dtype=torch.int32,
    )
    q = torch.randn(batch * decode_query_len, num_q_heads, head_dim, device=DEVICE, dtype=DTYPE)

    topk_idx = torch.full(
        (num_kv_heads, batch * decode_query_len, topk),
        -1,
        device=DEVICE,
        dtype=torch.int32,
    )
    token_id = 0
    for seq_len in seq_lens_list:
        for local_q in range(decode_query_len):
            query_pos = seq_len - decode_query_len + local_q
            current_block = query_pos // BLOCK_SIZE
            older_blocks = torch.randperm(current_block, device=DEVICE, dtype=torch.int32)
            selected = torch.cat(
                [
                    torch.tensor([current_block], device=DEVICE, dtype=torch.int32),
                    older_blocks[: topk - 1],
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
def test_prefill_sparse_attention_correctness(
    msa_m3_sparse_backend: SparseAttnBackend,
    q_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
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
        block_table[req_id, :num_req_pages] = physical_pages[base_page : base_page + num_req_pages]
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
    kv_cache = _main_kv_cache_for_kernel(kv_cache_fused)
    topk_idx = _build_prefill_topk_idx(q_lens_t, prefix_lens, total_q)

    actual = torch.empty_like(q)
    _run_prefill_sparse_attention(
        msa_m3_sparse_backend,
        q=q,
        kv_cache=kv_cache,
        kv_cache_fused=kv_cache_fused,
        topk_idx=topk_idx,
        block_table=block_table,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        q_lens_t=q_lens_t,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_seqlen_q,
        num_kv_heads=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        output=actual,
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
    _assert_sparse_close(actual, expected, backend=msa_m3_sparse_backend)


@pytest.mark.parametrize(
    ("tensor_parallel_size", "num_q_heads"),
    [
        (8, PRODUCTION_TP8_NUM_Q_HEADS),
        (16, PRODUCTION_TP16_NUM_Q_HEADS),
    ],
)
def test_prefill_sparse_attention_production_long_sequence(
    msa_m3_sparse_backend: SparseAttnBackend,
    tensor_parallel_size: int,
    num_q_heads: int,
) -> None:
    """Sparse prefill at online max context (10240 KV tokens, 80 blocks).

    The basic correctness tests cap kv_lens at 385; production-shape tests stop
    at 8224. This case matches ``--max-model-len 10240`` decode/prefill serve.
    """
    del tensor_parallel_size  # encoded via num_q_heads
    num_kv_heads = PRODUCTION_NUM_KV_HEADS_PER_RANK
    head_dim = HEAD_DIM
    sm_scale = head_dim**-0.5
    q_len = PRODUCTION_LONG_PREFILL_QUERY_LEN
    kv_len = PRODUCTION_MAX_SEQ_LEN

    torch.manual_seed(0)
    batch = 1
    max_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = max_blocks
    block_table = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32).reshape(1, max_blocks)

    q_lens_t = torch.tensor((q_len,), device=DEVICE, dtype=torch.int32)
    seq_lens = torch.tensor((kv_len,), device=DEVICE, dtype=torch.int32)
    prefix_lens = seq_lens - q_lens_t
    cu_seqlens = torch.zeros(batch + 1, device=DEVICE, dtype=torch.int32)
    cu_seqlens[1] = q_len

    q = torch.randn(q_len, num_q_heads, head_dim, device=DEVICE, dtype=DTYPE)
    kv_cache_fused = _allocate_main_kv_cache_fused(num_pages, num_kv_heads=num_kv_heads, head_dim=head_dim)
    kv_cache = _main_kv_cache_for_kernel(kv_cache_fused)
    topk_idx = _build_prefill_topk_idx(
        q_lens_t,
        prefix_lens,
        q_len,
        num_kv_heads=num_kv_heads,
        topk=PRODUCTION_SPARSE_TOPK,
    )

    actual = torch.empty_like(q)
    _run_prefill_sparse_attention(
        msa_m3_sparse_backend,
        q=q,
        kv_cache=kv_cache,
        kv_cache_fused=kv_cache_fused,
        topk_idx=topk_idx,
        block_table=block_table,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        q_lens_t=q_lens_t,
        prefix_lens=prefix_lens,
        max_seqlen_q=q_len,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
        output=actual,
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
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
    )
    _assert_sparse_close(actual, expected, backend=msa_m3_sparse_backend)


@pytest.mark.parametrize(
    ("tensor_parallel_size", "num_q_heads"),
    [
        (8, PRODUCTION_TP8_NUM_Q_HEADS),
        (16, PRODUCTION_TP16_NUM_Q_HEADS),
    ],
)
@pytest.mark.parametrize("num_reqs", [1, 2])
def test_prefill_sparse_attention_production_shape(
    msa_m3_sparse_backend: SparseAttnBackend,
    tensor_parallel_size: int,
    num_q_heads: int,
    num_reqs: int,
) -> None:
    """Sparse prefill with per-rank heads matching TP=8 / TP=16 online serve."""
    del tensor_parallel_size  # encoded via num_q_heads
    num_kv_heads = PRODUCTION_NUM_KV_HEADS_PER_RANK
    head_dim = HEAD_DIM
    sm_scale = head_dim**-0.5

    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    if num_reqs == 1:
        q_lens = (64,)
        kv_lens = (4096 + 64,)
    else:
        q_lens = (64, 32)
        kv_lens = (4096 + 64, 8192 + 32)

    torch.manual_seed(0)
    batch = len(q_lens)
    pages_per_req = [(kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE for kv_len in kv_lens]
    max_blocks = max(pages_per_req)
    num_pages = sum(pages_per_req)
    physical_pages = torch.randperm(num_pages, device=DEVICE, dtype=torch.int32)
    block_table = torch.zeros(batch, max_blocks, device=DEVICE, dtype=torch.int32)
    base_page = 0
    for req_id, num_req_pages in enumerate(pages_per_req):
        block_table[req_id, :num_req_pages] = physical_pages[base_page : base_page + num_req_pages]
        base_page += num_req_pages

    q_lens_t = torch.tensor(q_lens, device=DEVICE, dtype=torch.int32)
    seq_lens = torch.tensor(kv_lens, device=DEVICE, dtype=torch.int32)
    prefix_lens = seq_lens - q_lens_t
    cu_seqlens = torch.zeros(batch + 1, device=DEVICE, dtype=torch.int32)
    cu_seqlens[1:] = q_lens_t.cumsum(0)
    total_q = sum(q_lens)
    max_seqlen_q = max(q_lens)

    q = torch.randn(total_q, num_q_heads, head_dim, device=DEVICE, dtype=DTYPE)
    kv_cache_fused = _allocate_main_kv_cache_fused(num_pages, num_kv_heads=num_kv_heads, head_dim=head_dim)
    kv_cache = _main_kv_cache_for_kernel(kv_cache_fused)
    topk_idx = _build_prefill_topk_idx(
        q_lens_t,
        prefix_lens,
        total_q,
        num_kv_heads=num_kv_heads,
        topk=PRODUCTION_SPARSE_TOPK,
    )

    actual = torch.empty_like(q)
    _run_prefill_sparse_attention(
        msa_m3_sparse_backend,
        q=q,
        kv_cache=kv_cache,
        kv_cache_fused=kv_cache_fused,
        topk_idx=topk_idx,
        block_table=block_table,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        q_lens_t=q_lens_t,
        prefix_lens=prefix_lens,
        max_seqlen_q=max_seqlen_q,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
        output=actual,
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
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
    )
    _assert_sparse_close(actual, expected, backend=msa_m3_sparse_backend)


@pytest.mark.parametrize(
    "seq_lens_list",
    [(130, 257), (129, 200, 384)],
)
@pytest.mark.parametrize("decode_query_len", [1, 4])
@pytest.mark.parametrize("num_padded_reqs", [0, 2])
def test_decode_sparse_attention_correctness(
    msa_m3_sparse_backend: SparseAttnBackend,
    seq_lens_list: tuple[int, ...],
    decode_query_len: int,
    num_padded_reqs: int,
) -> None:
    torch.manual_seed(0)
    q, block_table, seq_lens, topk_idx, num_pages = _build_decode_inputs(
        seq_lens_list, decode_query_len, num_padded_reqs
    )
    kv_cache_fused = _allocate_main_kv_cache_fused(num_pages)
    kv_cache = _main_kv_cache_for_kernel(kv_cache_fused)

    actual = torch.empty_like(q)
    active_batch = len(seq_lens_list)
    _run_decode_sparse_attention(
        msa_m3_sparse_backend,
        q=q,
        kv_cache=kv_cache,
        kv_cache_fused=kv_cache_fused,
        topk_idx=topk_idx,
        block_table=block_table,
        seq_lens=seq_lens,
        num_kv_heads=NUM_KV_HEADS,
        sm_scale=SM_SCALE,
        output=actual,
        decode_query_len=decode_query_len,
        active_batch=active_batch,
    )
    _synchronize()

    active_tokens = active_batch * decode_query_len
    q_lens_t = torch.full((active_batch,), decode_query_len, device=DEVICE, dtype=torch.int32)
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
    _assert_sparse_close(actual[:active_tokens], expected, backend=msa_m3_sparse_backend)


@pytest.mark.parametrize(
    ("tensor_parallel_size", "num_q_heads"),
    [
        (8, PRODUCTION_TP8_NUM_Q_HEADS),
        (16, PRODUCTION_TP16_NUM_Q_HEADS),
    ],
)
@pytest.mark.parametrize("decode_query_len", [1, PRODUCTION_MTP_DECODE_QUERY_LEN])
@pytest.mark.parametrize("num_reqs", [1, 2])
def test_decode_sparse_attention_production_shape(
    msa_m3_sparse_backend: SparseAttnBackend,
    tensor_parallel_size: int,
    num_q_heads: int,
    decode_query_len: int,
    num_reqs: int,
) -> None:
    """Sparse decode with per-rank GQA matching TP=8 / TP=16 and optional MTP."""
    del tensor_parallel_size  # encoded via num_q_heads
    num_kv_heads = PRODUCTION_NUM_KV_HEADS_PER_RANK
    head_dim = HEAD_DIM
    sm_scale = head_dim**-0.5
    seq_lens_list = (10240,) if num_reqs == 1 else (10240, 8192)

    torch.manual_seed(0)
    q, block_table, seq_lens, topk_idx, num_pages = _build_decode_inputs(
        seq_lens_list,
        decode_query_len,
        num_padded_reqs=0,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        topk=PRODUCTION_SPARSE_TOPK,
    )
    kv_cache_fused = _allocate_main_kv_cache_fused(num_pages, num_kv_heads=num_kv_heads, head_dim=head_dim)
    kv_cache = _main_kv_cache_for_kernel(kv_cache_fused)

    actual = torch.empty_like(q)
    active_batch = len(seq_lens_list)
    _run_decode_sparse_attention(
        msa_m3_sparse_backend,
        q=q,
        kv_cache=kv_cache,
        kv_cache_fused=kv_cache_fused,
        topk_idx=topk_idx,
        block_table=block_table,
        seq_lens=seq_lens,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
        output=actual,
        decode_query_len=decode_query_len,
        active_batch=active_batch,
    )
    _synchronize()

    active_tokens = active_batch * decode_query_len
    q_lens_t = torch.full((active_batch,), decode_query_len, device=DEVICE, dtype=torch.int32)
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
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
    )
    _assert_sparse_close(actual[:active_tokens], expected, backend=msa_m3_sparse_backend)


# ---------------------------------------------------------------------------
# Decode sparse attention boundary / robustness (synthetic adversarial metadata)
# ---------------------------------------------------------------------------

_DECODE_BOUNDARY_BASE_SEED = 20260701
_DECODE_BOUNDARY_RANDOM_ROUNDS = max(
    0,
    int(os.environ.get("MINIMAX_M3_DECODE_BOUNDARY_ROUNDS", "4")),
)
_DECODE_BOUNDARY_REPEAT_LAUNCHES = max(
    1,
    int(os.environ.get("MINIMAX_M3_DECODE_BOUNDARY_REPEATS", "8")),
)
_DECODE_BOUNDARY_MEAN_ATOL = 2.5e-3
_DECODE_BOUNDARY_MAX_ATOL = 5.5e-2


@dataclass(frozen=True)
class _DecodeBoundaryGeometry:
    batch: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    max_blocks: int
    decode_query_len: int


def _decode_visibility_limits(
    seq_lens: Sequence[int],
    decode_query_len: int,
) -> list[int]:
    limits: list[int] = []
    for seq_len in seq_lens:
        for query_offset in range(decode_query_len):
            query_pos = seq_len - decode_query_len + query_offset
            limits.append(max(query_pos + 1, 0))
    return limits


def _visible_logical_block_count(limit: int, max_blocks: int) -> int:
    return min(max_blocks, (max(limit, 0) + BLOCK_SIZE - 1) // BLOCK_SIZE)


def _choose_adversarial_topk_blocks(
    *,
    visible_limit: int,
    max_blocks: int,
    max_topk: int,
    pattern: str,
    rng: random.Random,
) -> list[int]:
    """Packed unique top-k prefix; tail padded with -1 in the caller."""
    visible_count = _visible_logical_block_count(visible_limit, max_blocks)
    visible = list(range(visible_count))
    future = list(range(visible_count, max_blocks))
    all_blocks = list(range(max_blocks))
    selected: list[int] = []

    def append(block: int) -> None:
        if block not in selected and len(selected) < max_topk:
            selected.append(block)

    if pattern == "empty":
        return selected

    if pattern == "future_only":
        if future:
            rng.shuffle(future)
            for block in future[: max(1, min(len(future), rng.randint(1, 4)))]:
                append(block)
        return selected

    if pattern == "future_then_visible":
        if future:
            append(future[rng.randrange(len(future))])
        if visible:
            append(visible[rng.randrange(len(visible))])
        remainder = all_blocks[:]
        rng.shuffle(remainder)
        for block in remainder:
            if len(selected) >= min(max_topk, 5):
                break
            append(block)
        return selected

    if pattern != "random_mixed":
        raise ValueError(f"Unknown top-k pattern: {pattern}")

    candidates = all_blocks[:]
    rng.shuffle(candidates)
    if future and rng.randrange(3) == 0:
        append(future[rng.randrange(len(future))])
    desired = rng.randint(0, min(max_topk, max_blocks))
    for block in candidates:
        if len(selected) >= desired:
            break
        append(block)
    return selected


def _build_adversarial_decode_topk(
    *,
    num_kv_heads: int,
    total_q: int,
    max_blocks: int,
    visible_limits: Sequence[int],
    pattern: str,
    seed: int,
    topk: int = TOPK,
) -> torch.Tensor:
    assert len(visible_limits) == total_q
    cpu = torch.full((num_kv_heads, total_q, topk), -1, dtype=torch.int32)
    for kv_head in range(num_kv_heads):
        for token_id, limit in enumerate(visible_limits):
            rng = random.Random(seed + kv_head * 100_003 + token_id * 1_009)
            selected = _choose_adversarial_topk_blocks(
                visible_limit=limit,
                max_blocks=max_blocks,
                max_topk=topk,
                pattern=pattern,
                rng=rng,
            )
            if selected:
                cpu[kv_head, token_id, : len(selected)] = torch.tensor(
                    selected,
                    dtype=torch.int32,
                )
    return cpu.to(device=DEVICE)


def _make_shuffled_block_table(
    *,
    batch: int,
    max_blocks: int,
    seed: int,
) -> tuple[torch.Tensor, int]:
    num_pages = batch * max_blocks + 5
    rng = random.Random(seed)
    pages = list(range(num_pages))
    rng.shuffle(pages)
    rows = [pages[req * max_blocks : (req + 1) * max_blocks] for req in range(batch)]
    return torch.tensor(rows, device=DEVICE, dtype=torch.int32), num_pages


def _make_boundary_kv_cache_fused(
    num_pages: int,
    *,
    num_kv_heads: int,
    head_dim: int,
    seed: int,
    dtype: torch.dtype = DTYPE,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    kv = (
        torch.randn(
            (num_pages, 2, BLOCK_SIZE, num_kv_heads, head_dim),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.25
    )
    page_bias = torch.arange(num_pages, dtype=torch.float32).reshape(num_pages, 1, 1, 1) * 0.015625
    kv[:, 0] = kv[:, 0] + page_bias * 0.5
    kv[:, 1] = kv[:, 1] + page_bias
    return kv.to(device=DEVICE, dtype=dtype)


def _make_boundary_query(
    total_q: int,
    num_q_heads: int,
    head_dim: int,
    *,
    seed: int,
    dtype: torch.dtype = DTYPE,
    noncontiguous: bool,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if not noncontiguous:
        return (
            torch.randn(
                (total_q, num_q_heads, head_dim),
                generator=generator,
                dtype=torch.float32,
            )
            * 0.25
        ).to(device=DEVICE, dtype=dtype)
    base = (
        torch.randn(
            (total_q, num_q_heads, head_dim * 2),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.25
    ).to(device=DEVICE, dtype=dtype)
    return base[..., ::2]


def _build_decode_boundary_inputs(
    *,
    seed: int,
    geometry: _DecodeBoundaryGeometry,
    seq_lens_list: tuple[int, ...],
    pattern: str,
    noncontiguous_q: bool = False,
    topk: int = TOPK,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    assert len(seq_lens_list) == geometry.batch
    dq = geometry.decode_query_len
    total_q = geometry.batch * dq
    assert geometry.num_q_heads % geometry.num_kv_heads == 0

    block_table, num_pages = _make_shuffled_block_table(
        batch=geometry.batch,
        max_blocks=geometry.max_blocks,
        seed=seed + 11,
    )
    kv_cache_fused = _make_boundary_kv_cache_fused(
        num_pages,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        seed=seed + 29,
    )
    q = _make_boundary_query(
        total_q,
        geometry.num_q_heads,
        geometry.head_dim,
        seed=seed + 47,
        noncontiguous=noncontiguous_q,
    )
    limits = _decode_visibility_limits(seq_lens_list, dq)
    topk_idx = _build_adversarial_decode_topk(
        num_kv_heads=geometry.num_kv_heads,
        total_q=total_q,
        max_blocks=geometry.max_blocks,
        visible_limits=limits,
        pattern=pattern,
        seed=seed + 71,
        topk=topk,
    )
    seq_lens = torch.tensor(seq_lens_list, device=DEVICE, dtype=torch.int32)
    return q, kv_cache_fused, topk_idx, block_table, seq_lens, num_pages


def _selected_decode_positions(
    topk_row: torch.Tensor,
    *,
    visible_limit: int,
) -> torch.Tensor:
    positions: list[torch.Tensor] = []
    seen: set[int] = set()
    for raw in topk_row.detach().cpu().tolist():
        block = int(raw)
        if block < 0 or block in seen:
            continue
        seen.add(block)
        start = block * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, max(visible_limit, 0))
        if start < end:
            positions.append(torch.arange(start, end, device=DEVICE, dtype=torch.long))
    if not positions:
        return torch.empty((0,), device=DEVICE, dtype=torch.long)
    return torch.cat(positions)


def _reference_decode_sparse_boundary(
    q: torch.Tensor,
    kv_cache_fused: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    decode_query_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    sm_scale: float,
) -> torch.Tensor:
    total_q, _, _ = q.shape
    group_size = num_q_heads // num_kv_heads
    out = torch.zeros_like(q, dtype=torch.float32)
    seq_lens_list = [int(x) for x in seq_lens.detach().cpu().tolist()]
    limit_for_token = _decode_visibility_limits(seq_lens_list, decode_query_len)
    request_for_token = [token_id // decode_query_len for token_id in range(total_q)]

    for token_id in range(total_q):
        req_id = request_for_token[token_id]
        visible_limit = limit_for_token[token_id]
        for kv_head in range(num_kv_heads):
            positions = _selected_decode_positions(
                topk_idx[kv_head, token_id],
                visible_limit=visible_limit,
            )
            if positions.numel() == 0:
                continue
            logical_blocks = positions // BLOCK_SIZE
            page_rows = positions % BLOCK_SIZE
            pages = block_table[req_id, logical_blocks].long()
            k = kv_cache_fused[pages, 0, page_rows, kv_head].float()
            v = kv_cache_fused[pages, 1, page_rows, kv_head].float()
            begin = kv_head * group_size
            end = begin + group_size
            scores = torch.matmul(q[token_id, begin:end].float(), k.T) * sm_scale
            probs = torch.softmax(scores, dim=-1)
            out[token_id, begin:end] = torch.matmul(probs, v)

    return out.to(dtype=q.dtype)


def _decode_boundary_case_hint(
    *,
    name: str,
    seed: int,
    geometry: _DecodeBoundaryGeometry,
    seq_lens_list: tuple[int, ...],
    pattern: str,
    topk_idx: torch.Tensor,
) -> str:
    return (
        f"decode_boundary:{name}; seed={seed}; pattern={pattern}; "
        f"seq_lens={list(seq_lens_list)}; dq={geometry.decode_query_len}; "
        f"q_heads={geometry.num_q_heads}; kv_heads={geometry.num_kv_heads}; "
        f"topk_preview={topk_idx[:, : min(topk_idx.shape[1], 4)].detach().cpu().tolist()}"
    )


def _assert_sparse_finite_and_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    hint: str,
) -> None:
    nonfinite = ~torch.isfinite(actual)
    if nonfinite.any():
        first = int(nonfinite.reshape(-1).nonzero()[0].item())
        raise AssertionError(
            f"{hint}; output has {int(nonfinite.sum().item())} non-finite elements; "
            f"first_flat={first}; actual={actual.reshape(-1)[first].float().item()}"
        )
    error = (actual.float() - expected.float()).abs()
    mean_error = error.mean().item()
    max_error = error.max().item()
    if mean_error >= _DECODE_BOUNDARY_MEAN_ATOL or max_error >= _DECODE_BOUNDARY_MAX_ATOL:
        worst = int(error.reshape(-1).argmax().item())
        raise AssertionError(
            f"{hint}; mean_error={mean_error:.6g} "
            f"(tol={_DECODE_BOUNDARY_MEAN_ATOL}); "
            f"max_error={max_error:.6g} (tol={_DECODE_BOUNDARY_MAX_ATOL}); "
            f"worst_flat={worst}; actual={actual.reshape(-1)[worst].float().item():.6g}; "
            f"expected={expected.reshape(-1)[worst].float().item():.6g}"
        )


def _decode_boundary_length(round_id: int, *, max_tokens: int, rng: random.Random) -> int:
    boundaries = [
        0,
        1,
        BLOCK_SIZE - 1,
        BLOCK_SIZE,
        BLOCK_SIZE + 1,
        2 * BLOCK_SIZE - 1,
        2 * BLOCK_SIZE,
        2 * BLOCK_SIZE + 1,
        3 * BLOCK_SIZE - 1,
        3 * BLOCK_SIZE + 3,
    ]
    value = boundaries[round_id % len(boundaries)]
    if round_id >= len(boundaries):
        value = rng.randint(0, max_tokens)
    return min(max(value, 0), max_tokens)


def _decode_boundary_fixed_plans() -> list[tuple[str, _DecodeBoundaryGeometry, tuple[int, ...], str, bool]]:
    return [
        (
            "boundary_future_then_visible",
            _DecodeBoundaryGeometry(2, 8, 1, 128, 5, decode_query_len=1),
            (BLOCK_SIZE, BLOCK_SIZE + 1),
            "future_then_visible",
            False,
        ),
        (
            "all_future_selected_blocks",
            _DecodeBoundaryGeometry(2, 8, 1, 64, 5, decode_query_len=1),
            (BLOCK_SIZE, 1),
            "future_only",
            False,
        ),
        (
            "all_topk_padding_and_zero_length_request",
            _DecodeBoundaryGeometry(3, 8, 2, 64, 5, decode_query_len=2),
            (0, BLOCK_SIZE, 2 * BLOCK_SIZE + 1),
            "empty",
            True,
        ),
        (
            "multi_token_cross_page_boundary",
            _DecodeBoundaryGeometry(2, 16, 2, 128, 6, decode_query_len=4),
            (BLOCK_SIZE + 2, 2 * BLOCK_SIZE + 1),
            "future_then_visible",
            False,
        ),
        (
            "long_context_shuffled_pages",
            _DecodeBoundaryGeometry(2, 16, 4, 64, 8, decode_query_len=1),
            (3 * BLOCK_SIZE + 3, 5 * BLOCK_SIZE + 7),
            "random_mixed",
            True,
        ),
    ]


def _random_decode_boundary_plan(
    round_id: int,
    seed: int,
) -> tuple[str, _DecodeBoundaryGeometry, tuple[int, ...], str, bool]:
    rng = random.Random(seed)
    geometries = [
        _DecodeBoundaryGeometry(1, 8, 1, 64, 5, decode_query_len=1),
        _DecodeBoundaryGeometry(2, 8, 2, 128, 6, decode_query_len=2),
        _DecodeBoundaryGeometry(3, 16, 2, 64, 7, decode_query_len=4),
        _DecodeBoundaryGeometry(2, 16, 4, 128, 8, decode_query_len=1),
    ]
    geometry = geometries[round_id % len(geometries)]
    max_tokens = geometry.max_blocks * BLOCK_SIZE
    seq_lens = tuple(
        _decode_boundary_length(round_id + req_id * 3, max_tokens=max_tokens, rng=rng)
        for req_id in range(geometry.batch)
    )
    patterns = ("future_then_visible", "future_only", "empty", "random_mixed")
    return (
        f"random_decode_{round_id}",
        geometry,
        seq_lens,
        patterns[round_id % len(patterns)],
        bool(round_id % 3 == 0),
    )


def _run_decode_boundary_case(
    *,
    name: str,
    seed: int,
    geometry: _DecodeBoundaryGeometry,
    seq_lens_list: tuple[int, ...],
    pattern: str,
    noncontiguous_q: bool,
) -> None:
    q, kv_cache_fused, topk_idx, block_table, seq_lens, _ = _build_decode_boundary_inputs(
        seed=seed,
        geometry=geometry,
        seq_lens_list=seq_lens_list,
        pattern=pattern,
        noncontiguous_q=noncontiguous_q,
    )
    kv_cache = _main_kv_cache_for_kernel(kv_cache_fused)
    sm_scale = geometry.head_dim**-0.5
    hint = _decode_boundary_case_hint(
        name=name,
        seed=seed,
        geometry=geometry,
        seq_lens_list=seq_lens_list,
        pattern=pattern,
        topk_idx=topk_idx,
    )

    actual = torch.empty_like(q)
    minimax_m3_sparse_attn_decode_triton(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        geometry.num_kv_heads,
        sm_scale,
        actual,
        geometry.decode_query_len,
    )
    _synchronize()

    expected = _reference_decode_sparse_boundary(
        q,
        kv_cache_fused,
        topk_idx,
        block_table,
        seq_lens,
        decode_query_len=geometry.decode_query_len,
        num_q_heads=geometry.num_q_heads,
        num_kv_heads=geometry.num_kv_heads,
        sm_scale=sm_scale,
    )
    _assert_sparse_finite_and_close(actual, expected, hint=hint)


@pytest.mark.parametrize(
    ("case_index", "name", "geometry", "seq_lens_list", "pattern", "noncontiguous_q"),
    [(index, *plan) for index, plan in enumerate(_decode_boundary_fixed_plans())],
    ids=[plan[0] for plan in _decode_boundary_fixed_plans()],
)
def test_decode_sparse_attention_boundary_fixed_cases(
    msa_m3_sparse_backend_triton_only: SparseAttnBackend,
    case_index: int,
    name: str,
    geometry: _DecodeBoundaryGeometry,
    seq_lens_list: tuple[int, ...],
    pattern: str,
    noncontiguous_q: bool,
) -> None:
    """Decode sparse attention with adversarial but address-valid top-k metadata.

    Covers future-before-visible ordering, all-future selections, empty top-k rows,
    zero-length requests, multi-token page boundaries, and shuffled page tables.
    The kernel must stay finite and match the dense oracle on visible tokens.
    """
    seed = _DECODE_BOUNDARY_BASE_SEED + 20_000 + case_index * 271
    _run_decode_boundary_case(
        name=name,
        seed=seed,
        geometry=geometry,
        seq_lens_list=seq_lens_list,
        pattern=pattern,
        noncontiguous_q=noncontiguous_q,
    )


def test_decode_sparse_attention_boundary_random_cases(
    msa_m3_sparse_backend_triton_only: SparseAttnBackend,
) -> None:
    """Seeded random decode boundary cases around block-size edges."""
    plans = [
        _random_decode_boundary_plan(
            round_id,
            _DECODE_BOUNDARY_BASE_SEED + 10_000 + round_id * 97,
        )
        for round_id in range(_DECODE_BOUNDARY_RANDOM_ROUNDS)
    ]
    for index, (name, geometry, seq_lens_list, pattern, noncontiguous_q) in enumerate(plans):
        seed = _DECODE_BOUNDARY_BASE_SEED + 30_000 + index * 271
        _run_decode_boundary_case(
            name=name,
            seed=seed,
            geometry=geometry,
            seq_lens_list=seq_lens_list,
            pattern=pattern,
            noncontiguous_q=noncontiguous_q,
        )


def test_decode_sparse_attention_boundary_repeatable(
    msa_m3_sparse_backend_triton_only: SparseAttnBackend,
) -> None:
    """Repeated eager launches for the historical future-then-visible failure shape.

    Selected block 1 precedes valid block 0 while seq_len is exactly one page.
    Output must stay finite, match the oracle, and be identical across launches.
    """
    geometry = _DecodeBoundaryGeometry(1, 8, 1, 128, 5, decode_query_len=1)
    seq_lens_list = (BLOCK_SIZE,)
    pattern = "future_then_visible"
    seed = _DECODE_BOUNDARY_BASE_SEED + 50_000
    q, kv_cache_fused, topk_idx, block_table, seq_lens, _ = _build_decode_boundary_inputs(
        seed=seed,
        geometry=geometry,
        seq_lens_list=seq_lens_list,
        pattern=pattern,
        noncontiguous_q=False,
    )
    kv_cache = _main_kv_cache_for_kernel(kv_cache_fused)
    sm_scale = geometry.head_dim**-0.5
    hint = _decode_boundary_case_hint(
        name="repeat_boundary_future_then_valid",
        seed=seed,
        geometry=geometry,
        seq_lens_list=seq_lens_list,
        pattern=pattern,
        topk_idx=topk_idx,
    )
    expected = _reference_decode_sparse_boundary(
        q,
        kv_cache_fused,
        topk_idx,
        block_table,
        seq_lens,
        decode_query_len=geometry.decode_query_len,
        num_q_heads=geometry.num_q_heads,
        num_kv_heads=geometry.num_kv_heads,
        sm_scale=sm_scale,
    )

    first: torch.Tensor | None = None
    for launch_id in range(_DECODE_BOUNDARY_REPEAT_LAUNCHES):
        actual = torch.empty_like(q)
        minimax_m3_sparse_attn_decode_triton(
            q,
            kv_cache,
            topk_idx,
            block_table,
            seq_lens,
            geometry.num_kv_heads,
            sm_scale,
            actual,
            geometry.decode_query_len,
        )
        _synchronize()
        _assert_sparse_finite_and_close(
            actual,
            expected,
            hint=f"{hint}; launch={launch_id}",
        )
        if first is None:
            first = actual.clone()
            continue
        delta = (actual.float() - first.float()).abs().max().item()
        if delta != 0.0:
            raise AssertionError(
                f"{hint}; launch={launch_id}; output changed across eager launches; max_delta={delta:.6g}"
            )
