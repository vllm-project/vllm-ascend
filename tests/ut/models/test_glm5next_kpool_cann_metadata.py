# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.indexer_kpool import (
    AscendIndexerKPoolMetadataBuilder,
    AscendIndexerKPoolTailBackend,
    AscendIndexerKPoolTailMetadataBuilder,
)
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendMLAAttentionSpec
from vllm_ascend.utils import vllm_version_is


def _config():
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=2),
        model_config=SimpleNamespace(max_model_len=256),
    )


def _common():
    # Two requests resume at positions 2 and 7. They complete three pools;
    # the first completed pool has an evicted cache slot. Two padded tokens
    # also look like pool boundaries and must not affect output addressing.
    return SimpleNamespace(
        num_reqs=2,
        num_input_tokens=10,
        num_actual_tokens=8,
        query_start_loc=torch.tensor([0, 6, 8], dtype=torch.int32),
        seq_lens=torch.tensor([8, 9], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([8, 9], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0, -1], [1, -1]], dtype=torch.int32),
        slot_mapping=torch.tensor([2, -1, 4, 5, 6, 7, 135, 136, -1, -1]),
        positions=torch.tensor([2, 3, 4, 5, 6, 7, 7, 8, 3, 7]),
    )


def test_cann_metadata_addresses_flat_pools_across_chunks_eviction_and_padding():
    ratio = {"compress_ratio": 4} if vllm_version_is("0.28.0") else {"tokens_per_state": 4}
    builder = AscendIndexerKPoolMetadataBuilder(
        AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            **ratio,
        ),
        ["layer.indexer.k_cache"],
        _config(),
        torch.device("cpu"),
    )
    common = _common()
    metadata = builder.build(0, common)

    assert metadata.start_pos is not None
    assert metadata.query_start_loc is not None
    assert metadata.cum_query_lens is not None
    assert metadata.pool_tail is not None
    assert metadata.pooled_key_indices is not None
    assert metadata.start_pos.tolist() == [2, 7]
    assert metadata.query_start_loc.tolist() == [0, 6, 8]
    assert metadata.cum_query_lens.tolist() == [6, 8]
    assert metadata.seq_lens.tolist() == [2, 2]
    assert metadata.pool_tail.tolist() == [0, 1]
    assert metadata.pooled_key_indices.tolist() == [0, 0, 0, 0, 0, 1, 2, 2, 2, 2]
    assert metadata.slot_mapping.tolist() == [-1, -1, -1, -1, -1, 1, 33, -1, -1, -1]
    assert metadata.start_pos.dtype == metadata.query_start_loc.dtype == torch.int32
    assert metadata.pool_tail.dtype == metadata.cum_query_lens.dtype == metadata.seq_lens.dtype == torch.int64

    # A captured graph retains these addresses. Reusing the same batch shape
    # must update their contents in place as the next decode step advances.
    fields = ("start_pos", "query_start_loc", "pool_tail", "pooled_key_indices", "cum_query_lens", "seq_lens")
    pointers = {name: getattr(metadata, name).data_ptr() for name in fields}
    common.seq_lens.add_(1)
    common.positions.add_(1)
    next_metadata = builder.build(0, common)
    assert pointers == {name: getattr(next_metadata, name).data_ptr() for name in fields}
    assert metadata.start_pos.tolist() == [3, 8]
    assert metadata.pool_tail.tolist() == [1, 2]


def test_key_pool_tail_table_aliases_one_physical_page_and_refreshes_in_place():
    builder = AscendIndexerKPoolTailMetadataBuilder(
        AscendIndexerKPoolTailSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float32,
            sliding_window=4,
            compress_ratio=4,
        ),
        ["layer.indexer.tail_cache"],
        _config(),
        torch.device("cpu"),
    )
    common = _common()
    metadata = builder.build(0, common)
    assert metadata.block_table.shape == (2, 64)
    assert metadata.block_table.is_contiguous()
    assert metadata.block_table[0].tolist() == [0] * 64
    assert metadata.block_table[1].tolist() == [1] * 64
    pointer = metadata.block_table.data_ptr()
    common.block_table_tensor[:, 0] = torch.tensor([3, 2])
    updated = builder.build(0, common)
    assert updated.block_table.data_ptr() == pointer
    assert metadata.block_table[0].tolist() == [3] * 64
    assert metadata.block_table[1].tolist() == [2] * 64
    assert AscendIndexerKPoolTailBackend.get_kv_cache_shape(4, 4, 1, 128) == (4, 4, 256)


@pytest.mark.parametrize("padded_end", [-1, 2])
def test_cann_graph_padding_has_empty_query_and_nonnegative_start(padded_end):
    ratio = {"compress_ratio": 4} if vllm_version_is("0.28.0") else {"tokens_per_state": 4}
    builder = AscendIndexerKPoolMetadataBuilder(
        AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            **ratio,
        ),
        ["layer.indexer.k_cache"],
        _config(),
        torch.device("cpu"),
    )
    common = _common()
    captured = builder.build(0, common)
    assert captured.query_start_loc is not None
    assert captured.cum_query_lens is not None
    assert captured.start_pos is not None
    assert captured.raw_seq_lens is not None
    assert captured.pool_tail is not None
    common.num_input_tokens = 2
    common.num_actual_tokens = 1
    common.query_start_loc = torch.tensor([0, 1, padded_end], dtype=torch.int32)
    common.seq_lens = common._seq_lens_cpu = torch.tensor([9, 0], dtype=torch.int32)
    common.positions = torch.tensor([8, 3])
    common.slot_mapping = torch.tensor([8, -1])
    updated = builder.build(0, common)
    assert updated.query_start_loc is not None
    assert updated.pooled_key_indices is not None
    assert updated.query_start_loc.data_ptr() == captured.query_start_loc.data_ptr()
    assert captured.query_start_loc.tolist() == [0, 1, 1]
    assert captured.cum_query_lens.tolist() == [1, 1]
    assert captured.start_pos.tolist() == [8, 0]
    assert captured.raw_seq_lens.tolist() == [9, 0]
    assert captured.seq_lens.tolist() == [2, 0]
    assert captured.pool_tail.tolist() == [1, 0]
    assert updated.pooled_key_indices.tolist() == [0, 0]
    assert updated.slot_mapping.tolist() == [-1, -1]
    # The shared FIA metadata must retain its own padding contract.
    assert common.query_start_loc.tolist() == [0, 1, padded_end]
