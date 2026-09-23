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
from vllm_ascend.spec_decode.multi_kv_cache_group_proposer import AscendMultiKVCacheGroupMTPProposer
from vllm_ascend.utils import vllm_version_is


def _config():
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
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


class _NoHostLengths(SimpleNamespace):
    @property
    def _seq_lens_cpu(self):
        raise AssertionError("CANN metadata must not access host sequence lengths")

    @property
    def seq_lens_cpu(self):
        raise AssertionError("CANN metadata must not trigger a sequence-length D2H copy")


@pytest.mark.parametrize("host_lengths_available", [True, False])
def test_cann_metadata_addresses_flat_pools_across_chunks_eviction_and_padding(host_lengths_available):
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
    if not host_lengths_available:
        common = _NoHostLengths(**{name: value for name, value in vars(common).items() if name != "_seq_lens_cpu"})
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


def test_key_pool_tail_table_retains_one_physical_page_and_refreshes_in_place():
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
    assert metadata.block_table.shape == (2, 1)
    assert metadata.block_table.is_contiguous()
    assert metadata.block_table[0].tolist() == [0]
    assert metadata.block_table[1].tolist() == [1]
    pointer = metadata.block_table.data_ptr()
    common.block_table_tensor[:, 0] = torch.tensor([3, 2])
    updated = builder.build(0, common)
    assert updated.block_table.data_ptr() == pointer
    assert metadata.block_table[0].tolist() == [3]
    assert metadata.block_table[1].tolist() == [2]
    assert AscendIndexerKPoolTailBackend.get_kv_cache_shape(4, 4, 1, 128) == (4, 4, 256)


@pytest.mark.parametrize("lookahead", [0, 1, 3, 7])
def test_speculative_retention_selects_unique_current_rows_without_host_lengths(lookahead):
    config = _config()
    config.speculative_config = SimpleNamespace(num_speculative_tokens=lookahead) if lookahead else None
    ratio = {"compress_ratio": 4} if vllm_version_is("0.28.0") else {"tokens_per_state": 4}
    builder = AscendIndexerKPoolMetadataBuilder(
        AscendMLAAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, model_version="glm5_next", **ratio
        ),
        ["layer.indexer.k_cache"],
        config,
        torch.device("cpu"),
    )
    common = _NoHostLengths(**{k: v for k, v in vars(_common()).items() if k != "_seq_lens_cpu"})
    metadata = builder.build(0, common)
    if not lookahead:
        assert metadata.retained_tail_indices is None
        return
    retained = metadata.retained_tail_indices
    assert retained is not None
    capacity = 4 + lookahead
    expected = [[-1] * max(capacity - 6, 0) + list(range(max(0, 6 - capacity), 6)), [-1] * (capacity - 2) + [6, 7]]
    assert retained.tolist() == expected
    pointer = retained.data_ptr()
    common.num_input_tokens = 2
    common.num_actual_tokens = 1
    common.query_start_loc = torch.tensor([0, 1, -1], dtype=torch.int32)
    common.seq_lens = torch.tensor([10, 0], dtype=torch.int32)
    common.positions = torch.tensor([9, 0])
    common.slot_mapping = torch.tensor([9, -1])
    updated = builder.build(0, common)
    assert updated.retained_tail_indices is not None
    assert updated.retained_tail_indices.data_ptr() == pointer
    assert retained.tolist() == [[-1] * (capacity - 1) + [0], [-1] * capacity]


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


@pytest.mark.parametrize("logical_block_size", [128, 256, 1152])
@pytest.mark.parametrize("primary_group", [True, False])
def test_mtp_metadata_preserves_expanded_pages_for_long_history(logical_block_size, primary_group):
    config = _config()
    config.cache_config.block_size = logical_block_size
    config.model_config.max_model_len = 8192
    config.scheduler_config.max_num_seqs = 3
    config.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    config.parallel_config = SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1)
    ratio = {"compress_ratio": 4} if vllm_version_is("0.28.0") else {"tokens_per_state": 4}
    builder = AscendIndexerKPoolMetadataBuilder(
        AscendMLAAttentionSpec(
            block_size=logical_block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            **ratio,
        ),
        ["draft.indexer.k_cache"],
        config,
        torch.device("cpu"),
    )
    # The failed real MTP batch had these compressed lengths. Its scheduler
    # pages were already expanded to 128-token / 32-pool kernel pages.
    logical_width = (8192 + logical_block_size - 1) // logical_block_size
    split = logical_block_size // 128
    physical_pages = torch.arange(1, 3 * logical_width + 1, dtype=torch.int32).reshape(3, logical_width)
    table = (physical_pages[:, :, None] * split + torch.arange(split, dtype=torch.int32)).reshape(3, -1)
    lengths = torch.tensor([935, 938, 936], dtype=torch.int32) * 4
    positions = (lengths[:, None] - 4 + torch.arange(4)).flatten()
    common = _NoHostLengths(
        num_reqs=3,
        num_actual_tokens=12,
        num_input_tokens=12,
        query_start_loc=torch.tensor([0, 4, 8, 12], dtype=torch.int32),
        seq_lens=lengths,
        positions=positions,
        block_table_tensor=table,
        slot_mapping=torch.full((12,), -1, dtype=torch.int32),
    )
    proposer = AscendMultiKVCacheGroupMTPProposer.__new__(AscendMultiKVCacheGroupMTPProposer)
    proposer._uses_multi_group_kv_cache = True
    proposer.kv_cache_gid = 0
    proposer.max_model_len = 8192
    proposer.vllm_config = config
    gid = 0 if primary_group else 1
    group = SimpleNamespace(kv_cache_group_id=gid, get_metadata_builder=lambda: builder)
    secondary_table = SimpleNamespace(
        compute_slot_mapping=lambda *_args: None,
        get_device_tensor=lambda: table,
        slot_mapping=SimpleNamespace(gpu=common.slot_mapping),
    )
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(block_table=[secondary_table, secondary_table]))
    draft_common = proposer._common_attn_metadata_for_draft_group(common, group, 12)
    metadata = builder.build(0, draft_common)

    assert metadata.block_table.shape == table.shape
    # Check every visible pool against its physical scheduler page, including
    # those beyond the old eight-column crop, rather than just table width.
    for req in range(3):
        for pool in range(int(lengths[req]) // 4):
            token = pool * 4
            physical = physical_pages[req, token // logical_block_size]
            expected = int(physical) * (logical_block_size // 4) + (token % logical_block_size) // 4
            actual = int(metadata.block_table[req, pool // 32]) * 32 + pool % 32
            assert actual == expected
    assert common.block_table_tensor is table
