# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM-Next KPool lightning indexer op."""

from types import SimpleNamespace

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import (
    SparseAttnIndexerKpool,
)
from vllm_ascend.ops.glm5_next_lightning_indexer import (
    glm5_next_lightning_indexer,
)


def _make_forward_context(attn_metadata):
    return ForwardContext(
        no_compile_layers={},
        attn_metadata=attn_metadata,
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )


def test_lightning_indexer_selects_pools_expands_tokens_and_appends_tail():
    index_topk = 4
    index_kpool = 2
    logical_keys = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 2.0],
            [3.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )
    indexer_cache = torch.zeros((3, 2, 1, 2), dtype=torch.bfloat16)
    block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    for pool_id, logical_key in enumerate(logical_keys):
        logical_page, offset = divmod(pool_id, 2)
        physical_page = int(block_table[0, logical_page])
        indexer_cache[physical_page, offset, 0] = logical_key
    cache_before = indexer_cache.clone()

    query = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [2.0, 0.0]],
        ],
        dtype=torch.bfloat16,
    )
    weights = torch.tensor(
        [[1.0, 2.0], [1.0, -1.0]],
        dtype=torch.bfloat16,
    )
    result = glm5_next_lightning_indexer(
        query,
        indexer_cache,
        weights,
        cum_query_lens=torch.tensor([2], dtype=torch.int32),
        indexer_seq_lens=torch.tensor([5], dtype=torch.int32),
        indexer_block_table=block_table,
        positions=torch.tensor([4, 8], dtype=torch.int64),
        index_topk=index_topk,
        index_kpool=index_kpool,
        max_pool_seq_len=5,
    )

    assert result.dtype == torch.int32
    assert result.shape == (2, 1, index_topk + index_kpool - 1)
    assert sorted(result[0, 0, :4].tolist()) == [0, 1, 2, 3]
    assert result[0, 0, 4].item() == 4
    assert set(result[1, 0, :4].tolist()) == {2, 3, 6, 7}
    assert result[1, 0, 4].item() == 8
    torch.testing.assert_close(indexer_cache, cache_before)


def test_lightning_indexer_reads_block_major_pooled_cache_by_block_table():
    num_blocks = 4
    cache_block_size = 2
    head_dim = 2
    block_stride_bytes = 32
    page_offset_bytes = 8
    raw_cache = torch.zeros(
        num_blocks * block_stride_bytes,
        dtype=torch.int8,
    )
    indexer_cache = torch.as_strided(
        raw_cache.view(torch.bfloat16),
        size=(num_blocks, cache_block_size, 1, head_dim),
        stride=(block_stride_bytes // 2, head_dim, head_dim, 1),
        storage_offset=page_offset_bytes // 2,
    )
    block_table = torch.tensor(
        [[3, 1], [2, 0]],
        dtype=torch.int32,
    )
    request_keys = (
        ((1.0, 0.0), (4.0, 0.0), (2.0, 0.0)),
        ((0.0, 1.0), (0.0, 2.0), (0.0, 5.0)),
    )
    for request_id, logical_keys in enumerate(request_keys):
        for pool_id, key in enumerate(logical_keys):
            logical_page, offset = divmod(pool_id, cache_block_size)
            physical_page = int(block_table[request_id, logical_page])
            indexer_cache[physical_page, offset, 0] = torch.tensor(key)
    raw_before = raw_cache.clone()

    result = glm5_next_lightning_indexer(
        query=torch.tensor(
            [[[1.0, 0.0]], [[0.0, 1.0]]],
            dtype=torch.bfloat16,
        ),
        indexer_cache=indexer_cache,
        weights=torch.ones((2, 1), dtype=torch.bfloat16),
        cum_query_lens=torch.tensor([1, 2], dtype=torch.int32),
        indexer_seq_lens=torch.tensor([3, 3], dtype=torch.int32),
        indexer_block_table=block_table,
        positions=torch.tensor([5, 5], dtype=torch.int64),
        index_topk=2,
        index_kpool=2,
        max_pool_seq_len=3,
    )

    assert result[:, 0].tolist() == [[2, 3, -1], [4, 5, -1]]
    assert indexer_cache.stride(0) * indexer_cache.element_size() == 32
    torch.testing.assert_close(raw_cache, raw_before)
    assert raw_cache[page_offset_bytes + 8].item() == 0


def test_indexer_kpool_forward_native_writes_caches_and_buffer():
    index_topk = 2
    index_kpool = 2
    head_dim = 4
    k = torch.tensor(
        [[1.0] * head_dim, [2.0] * head_dim, [3.0] * head_dim, [4.0] * head_dim],
        dtype=torch.float32,
    )
    gate = torch.zeros((4, head_dim), dtype=torch.float32)
    ape = torch.zeros((index_kpool, head_dim), dtype=torch.float32)
    positions = torch.arange(4, dtype=torch.int64)
    query = k.unsqueeze(1).expand(4, 2, head_dim).contiguous().to(torch.bfloat16)
    weights = torch.ones((4, 2), dtype=torch.bfloat16)

    state_cache_tensor = torch.zeros(
        (2, index_kpool, 2 * head_dim), dtype=torch.float32
    )
    indexer_cache_tensor = torch.zeros(
        (2, index_kpool, 1, head_dim), dtype=torch.bfloat16
    )
    buffer = torch.zeros((4, 128), dtype=torch.int32)
    k_cache = SimpleNamespace(
        kv_cache=indexer_cache_tensor,
        prefix="model.layers.3.indexer.k_cache",
        compress_ratio=index_kpool,
    )
    state_cache = SimpleNamespace(
        kv_cache=state_cache_tensor,
        prefix="model.layers.3.indexer.compressor.state_cache",
    )
    with set_current_vllm_config(VllmConfig()):
        op = SparseAttnIndexerKpool(
            k_cache=k_cache,
            quant_block_size=128,
            scale_fmt=None,
            topk_tokens=index_topk,
            head_dim=head_dim,
            max_model_len=16,
            max_total_seq_len=16,
            topk_indices_buffer=buffer,
            state_cache=state_cache,
        )

    indexer_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([-1, 0, -1, 1], dtype=torch.int64),
        cum_query_lens=torch.tensor([4], dtype=torch.int32),
        raw_seq_lens=torch.tensor([4], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([2], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        num_actual_tokens=4,
    )
    state_metadata = SimpleNamespace(
        slot_mapping=torch.arange(4, dtype=torch.int64),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
    )
    attn_metadata = {
        k_cache.prefix: indexer_metadata,
        state_cache.prefix: state_metadata,
    }

    with override_forward_context(_make_forward_context(attn_metadata)):
        result = op.forward_native(
            torch.zeros(4),
            query,
            k,
            weights,
            gate_score=gate,
            compress_ape=ape,
            index_kpool=index_kpool,
            positions=positions,
        )

    expected_state = torch.tensor(
        [
            [[1, 1, 1, 1, 0, 0, 0, 0], [2, 2, 2, 2, 0, 0, 0, 0]],
            [[3, 3, 3, 3, 0, 0, 0, 0], [4, 4, 4, 4, 0, 0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(state_cache_tensor, expected_state)
    torch.testing.assert_close(
        indexer_cache_tensor[0, 0, 0], torch.full((4,), 1.5).bfloat16()
    )
    torch.testing.assert_close(
        indexer_cache_tensor[0, 1, 0], torch.full((4,), 3.5).bfloat16()
    )
    assert result.dtype == torch.int32
    assert result.shape == (4, 1, index_topk + index_kpool - 1)
    assert [row.tolist() for row in result[:, 0]] == [
        [-1, -1, 0],
        [0, 1, -1],
        [0, 1, 2],
        [2, 3, -1],
    ]
    torch.testing.assert_close(buffer[:, :3], result[:, 0, :])
    assert torch.all(buffer[:, 3:] == -1)
