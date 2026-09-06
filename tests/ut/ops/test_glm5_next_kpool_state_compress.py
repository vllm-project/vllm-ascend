# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM-Next paged KPool compressor op."""

import torch

from vllm_ascend.ops.glm5_next_kpool_state_compress import (
    glm5_next_kpool_state_compress_and_write_cache,
)


def _compressed(
    window_k: torch.Tensor,
    window_gate: torch.Tensor,
    ape: torch.Tensor,
) -> torch.Tensor:
    weights = torch.softmax(window_gate.float() + ape, dim=0)
    return (weights * window_k.float()).sum(dim=0).to(torch.bfloat16)


def test_prefill_writes_fp32_state_and_compressed_bfloat16_cache():
    head_dim = 4
    index_kpool = 4
    state_cache = torch.full(
        (2, 4, 2 * head_dim),
        -3.0,
        dtype=torch.float32,
    )
    indexer_cache = torch.full(
        (2, 2, 1, head_dim),
        -7.0,
        dtype=torch.bfloat16,
    )

    k_storage = torch.arange(
        9 * 2 * head_dim,
        dtype=torch.float32,
    ).reshape(9, 2, head_dim)
    gate_storage = torch.arange(
        9 * 2 * head_dim,
        dtype=torch.float32,
    ).reshape(9, 2, head_dim)
    k = (k_storage * 0.05)[:, 0]
    gate_score = (gate_storage * 0.02)[:, 0]
    assert not k.is_contiguous()
    ape = (
        torch.arange(index_kpool * head_dim, dtype=torch.float32)
        .reshape(index_kpool, head_dim)
        .mul(0.01)
    )

    positions = torch.arange(9, dtype=torch.int64)
    state_slots = torch.arange(9, dtype=torch.int64)
    state_slots[8] = -1
    indexer_slots = torch.tensor(
        [-1, -1, -1, 0, -1, -1, -1, 1, -1],
        dtype=torch.int64,
    )
    glm5_next_kpool_state_compress_and_write_cache(
        state_cache,
        indexer_cache,
        k,
        gate_score,
        ape,
        positions,
        cum_query_lens=torch.tensor([8], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        state_slot_mapping=state_slots,
        state_block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        indexer_slot_mapping=indexer_slots,
        index_kpool=index_kpool,
    )

    for token_idx in range(8):
        expected = torch.cat([k[token_idx], gate_score[token_idx]])
        torch.testing.assert_close(
            state_cache[token_idx // 4, token_idx % 4],
            expected,
        )
    torch.testing.assert_close(
        indexer_cache[0, 0, 0],
        _compressed(k[:4], gate_score[:4], ape),
    )
    torch.testing.assert_close(
        indexer_cache[0, 1, 0],
        _compressed(k[4:8], gate_score[4:8], ape),
    )
    torch.testing.assert_close(
        indexer_cache[1],
        torch.full_like(indexer_cache[1], -7.0),
    )


def test_decode_uses_block_table_with_block_major_pooled_views():
    head_dim = 1
    index_kpool = 4
    num_blocks = 4
    block_stride_bytes = 64
    page_offset_bytes = 16
    raw_cache = torch.zeros(
        num_blocks * block_stride_bytes,
        dtype=torch.int8,
    )
    state_cache = torch.as_strided(
        raw_cache.view(torch.float32),
        size=(num_blocks, index_kpool, 2 * head_dim),
        stride=(block_stride_bytes // 4, 2 * head_dim, 1),
        storage_offset=page_offset_bytes // 4,
    )
    indexer_cache = torch.as_strided(
        raw_cache.view(torch.bfloat16),
        size=(num_blocks, index_kpool, 1, head_dim),
        stride=(block_stride_bytes // 2, 1, 1, 1),
        storage_offset=page_offset_bytes // 2,
    )
    indexer_cache.fill_(-7.0)

    state_cache[2, :3] = torch.tensor(
        [[20.0, 0.1], [21.0, 0.2], [22.0, 0.3]]
    )
    state_cache[3, :3] = torch.tensor(
        [[30.0, 0.4], [31.0, 0.5], [32.0, 0.6]]
    )
    k = torch.tensor([[100.0], [200.0]], dtype=torch.float32)
    gate_score = torch.tensor([[0.7], [0.8]], dtype=torch.float32)
    ape = torch.zeros((index_kpool, head_dim), dtype=torch.float32)

    glm5_next_kpool_state_compress_and_write_cache(
        state_cache,
        indexer_cache,
        k,
        gate_score,
        ape,
        positions=torch.tensor([7, 7], dtype=torch.int64),
        cum_query_lens=torch.tensor([1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([8, 8], dtype=torch.int32),
        state_slot_mapping=torch.tensor([11, 15], dtype=torch.int64),
        state_block_table=torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
        indexer_slot_mapping=torch.tensor([1, 2], dtype=torch.int64),
        index_kpool=index_kpool,
    )

    torch.testing.assert_close(
        state_cache[2, 3],
        torch.tensor([100.0, 0.7]),
    )
    torch.testing.assert_close(
        state_cache[3, 3],
        torch.tensor([200.0, 0.8]),
    )
    window0 = torch.cat(
        [state_cache[2, :3], torch.tensor([[100.0, 0.7]])]
    )
    window1 = torch.cat(
        [state_cache[3, :3], torch.tensor([[200.0, 0.8]])]
    )
    torch.testing.assert_close(
        indexer_cache[0, 1, 0],
        _compressed(window0[:, :1], window0[:, 1:], ape),
    )
    torch.testing.assert_close(
        indexer_cache[0, 2, 0],
        _compressed(window1[:, :1], window1[:, 1:], ape),
    )
    assert state_cache.stride(0) * state_cache.element_size() == 64
    assert indexer_cache.stride(0) * indexer_cache.element_size() == 64
    assert raw_cache[page_offset_bytes + 32].item() == 0
