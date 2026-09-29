# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-Next cache specs and compressed-cache addressing."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.indexer_kpool import (
    AscendIndexerKPoolBackend,
    AscendIndexerKPoolMetadataBuilder,
    AscendIndexerKPoolTailBackend,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
    is_prefix_cacheable,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.models.glm5next.attention import Indexer
from vllm_ascend.models.glm5next.kv_cache import (
    Glm5NextIndexerCache,
    Glm5NextTailCache,
    KpoolTailManager,
    format_indexer_kpool_slot_mapping,
    get_kpool_tail_ring_capacity,
)
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import get_kv_cache_coordinator
from vllm_ascend.patch.platform.patch_kv_cache_utils import _ascend_resolve_kv_cache_block_sizes


def _ratio_kwargs(ratio: int) -> dict[str, int]:
    return {"tokens_per_state": ratio}


@pytest.mark.parametrize(("pool", "lookahead", "capacity"), [(4, 0, 4), (4, 3, 7), (16, 5, 21)])
def test_tail_ring_capacity_retains_speculative_lookahead(pool, lookahead, capacity):
    config = SimpleNamespace(
        speculative_config=(None if lookahead == 0 else SimpleNamespace(num_speculative_tokens=lookahead))
    )
    assert get_kpool_tail_ring_capacity(config, pool) == capacity


@pytest.mark.parametrize("capacity", [4, 12])
def test_tail_uses_one_full_precision_page(capacity):
    register_ascend_kv_cache_specs()
    spec = AscendIndexerKPoolTailSpec(
        block_size=capacity,
        sliding_window=4,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float32,
    )
    assert spec.page_size_bytes == 2 * capacity * 128 * 4
    assert spec.real_page_size_bytes == spec.unpadded_page_size_bytes == spec.page_size_bytes
    padded_spec = replace(spec, page_size_padded=spec.page_size_bytes + 256)
    assert padded_spec.page_size_bytes == spec.page_size_bytes + 256
    assert padded_spec.real_page_size_bytes == padded_spec.unpadded_page_size_bytes == spec.page_size_bytes
    with pytest.raises(AssertionError):
        _ = replace(spec, page_size_padded=spec.page_size_bytes - 1).page_size_bytes
    merged_spec = AscendIndexerKPoolTailSpec.merge([spec, replace(spec)])
    assert merged_spec == spec and merged_spec is not spec
    with pytest.raises(AssertionError):
        AscendIndexerKPoolTailSpec.merge([spec, padded_spec])
    with pytest.raises(AssertionError):
        AscendIndexerKPoolTailSpec.merge([object()])
    assert KVCacheSpecRegistry.get_manager_class(spec) is KpoolTailManager
    assert spec.max_admission_blocks_per_request(16, 1024) == 1
    assert spec.max_admission_blocks_per_request(8192, 131072) == 1
    assert not spec.prefix_cacheable
    assert not is_prefix_cacheable(spec)
    assert spec.is_circular
    context_parallel_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            prefill_context_parallel_size=2,
        ),
    )
    assert spec.max_memory_usage_bytes(context_parallel_config) == spec.page_size_bytes


@pytest.mark.parametrize(
    ("dtype", "block_size", "sliding_window"),
    [
        (torch.bfloat16, 4, 4),
        (torch.float32, 2, 4),
    ],
)
def test_invalid_state_layout_is_rejected(dtype, block_size, sliding_window):
    with pytest.raises(ValueError):
        AscendIndexerKPoolTailSpec(
            block_size=block_size,
            sliding_window=sliding_window,
            compress_ratio=4,
            num_kv_heads=1,
            head_size=256,
            dtype=dtype,
        )


def test_completed_pool_slots_preserve_logical_block_padding():
    slots = torch.tensor([0, 14, 15, 16, 127, 128, 143, -1])
    positions = torch.tensor([0, 14, 15, 16, 127, 128, 143, 15])
    actual = format_indexer_kpool_slot_mapping(slots, positions, 128, 16)
    assert actual.tolist() == [-1, -1, 0, -1, 7, -1, 8, -1]


def test_tail_manager_retains_one_private_block_until_free():
    spec = AscendIndexerKPoolTailSpec(
        block_size=4,
        sliding_window=4,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float32,
    )
    pool = BlockPool(12, True, 4)
    manager = KpoolTailManager(spec, pool, True, 1, scheduler_block_size=16)
    initially_free = pool.get_num_free_blocks()
    assert manager.get_num_blocks_to_allocate("a", 1000, [], 0, 0, 1000) == 1
    a = manager.allocate_new_blocks("a", 1000, 1000)
    b = manager.allocate_new_blocks("b", 3, 3)
    assert len(a) == len(b) == 1
    assert a[0].block_id != b[0].block_id
    for tokens in (1001, 4096, 131072):
        assert manager.allocate_new_blocks("a", tokens, tokens) == []
        manager.remove_skipped_blocks("a", tokens)
        assert manager.req_to_blocks["a"] == a
        assert manager.get_num_blocks_to_allocate("a", tokens, [], tokens - 1, tokens - 1, tokens) == 0
    request = SimpleNamespace(request_id="a")
    manager.cache_blocks(request, 131072)
    manager.cache_blocks(request, 131072, replay_boundary=16)
    manager.cache_blocks(request, 131072, replay_boundaries=[16])
    assert a[0].block_hash is None
    assert manager.get_num_common_prefix_blocks("a") == 0
    manager.free("a")
    assert pool.get_num_free_blocks() == initially_free - 1
    assert manager.req_to_blocks["b"] == b
    manager.free("b")
    assert pool.get_num_free_blocks() == initially_free


def test_tail_manager_prefix_hit_has_no_shared_blocks():
    spec = AscendIndexerKPoolTailSpec(
        block_size=4,
        sliding_window=4,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float32,
    )
    pool = BlockPool(8, True, 4)
    manager = KpoolTailManager(spec, pool, True, 1, scheduler_block_size=16)
    blocks, hit = manager.find_longest_cache_hit([], 4096, [1, 3], pool, spec, False, 16)
    assert blocks == ([], []) and hit == 0
    manager.allocate_external_computed_blocks("hit", 1024, 0)
    assert len(manager.req_to_blocks["hit"]) == 1
    manager.allocate_external_computed_blocks("hit", 1024, 512)
    assert len(manager.req_to_blocks["hit"]) == 1
    manager.free("hit")


@pytest.mark.parametrize("ratio", [0, 1, 3])
def test_invalid_pool_geometry_is_rejected(ratio):
    with pytest.raises(ValueError):
        format_indexer_kpool_slot_mapping(torch.tensor([0]), torch.tensor([0]), 128, ratio)


@pytest.mark.parametrize(
    "storage_block_size,kernel_block_size",
    [(8, None), (24, None), (144, None), (1536, None), (2048, None), (1536, 128)],
)
def test_indexer_metadata_addresses_complete_storage_pages(storage_block_size, kernel_block_size):
    pool_size = 16
    logical_size = storage_block_size * pool_size
    kernel_size = kernel_block_size or 128
    split = logical_size // kernel_size
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=logical_size),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4, max_num_seqs=1),
        model_config=SimpleNamespace(max_model_len=logical_size * 3),
    )
    spec = AscendMLAAttentionSpec(
        block_size=logical_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=pool_size,
        model_version="glm5_next",
    )
    if kernel_block_size is not None:
        group = AttentionGroup(AscendIndexerKPoolBackend, ["layer.indexer.k_cache"], spec, 0)
        group.create_metadata_builders(
            config, torch.device("cpu"), kernel_block_size=kernel_block_size, num_metadata_builders=2
        )
        builders = group.metadata_builders
    else:
        builders = [
            AscendIndexerKPoolMetadataBuilder(spec, ["layer.indexer.k_cache"], config, torch.device("cpu"))
            for _ in range(2)
        ]
    pages = torch.tensor([[7, 2, -1]], dtype=torch.int32)
    expanded = (pages.unsqueeze(-1) * split + torch.arange(split)).reshape(1, -1).int()
    expanded[:, -split:] = -1
    common = SimpleNamespace(
        num_reqs=1,
        num_input_tokens=4,
        num_actual_tokens=3,
        max_query_len=3,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([logical_size + pool_size], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([logical_size + pool_size], dtype=torch.int32),
        seq_lens_cpu=None,
        positions=torch.tensor([logical_size - 1, logical_size, logical_size + pool_size - 1, 0]),
        slot_mapping=torch.tensor([8 * logical_size - 1, 2 * logical_size, 2 * logical_size + pool_size - 1, -1]),
        block_table_tensor=expanded,
    )
    captured = builders[0].build_for_cudagraph_capture(common)
    legacy_capture = builders[0].build_for_graph_capture(common)
    assert captured.num_tokens == legacy_capture.num_tokens == 4
    assert captured.max_pool_seq_len == legacy_capture.max_pool_seq_len == 3 * storage_block_size
    first, draft = [builder.build(0, common) for builder in builders]
    assert first.num_tokens == draft.num_tokens == 3
    assert first.max_pool_seq_len == draft.max_pool_seq_len == storage_block_size + 1
    assert captured.num_tokens == 4
    assert captured.max_pool_seq_len == 3 * storage_block_size
    # Kernel-granularity blocks: the metadata reports the natural kernel rows
    # (kernel size / pool ratio) and passes the common expanded table through
    # as a view, so both builders observe the same persistent buffer.
    assert first.block_size == kernel_size // pool_size
    torch.testing.assert_close(first.block_table, expanded)
    assert first.slot_mapping.tolist() == [8 * storage_block_size - 1, -1, 2 * storage_block_size, -1]
    assert first.seq_lens.tolist() == [storage_block_size + 1]
    address = first.block_table.data_ptr()
    assert draft.block_table.data_ptr() == address
    common.block_table_tensor[:, :split] = 3 * split + torch.arange(split)
    refreshed = builders[0].build(0, common)
    assert refreshed.block_table.data_ptr() == address
    torch.testing.assert_close(refreshed.block_table, common.block_table_tensor[:1])
    torch.testing.assert_close(draft.block_table, common.block_table_tensor[:1])


def test_model_cache_layers_publish_source_compatible_specs():
    current_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    cache_config = SimpleNamespace(block_size=256)
    with patch(
        "vllm_ascend.models.glm5next.kv_cache.get_current_vllm_config",
        return_value=current_config,
    ):
        indexer = Glm5NextIndexerCache(
            head_dim=128,
            dtype=torch.bfloat16,
            cache_role="indexer",
            cache_config=cache_config,
            prefix="model.layers.0.indexer.k_cache",
            compress_ratio=16,
        )
        state = Glm5NextTailCache(
            head_dim=128,
            dtype=torch.float32,
            compress_ratio=16,
            prefix="model.layers.0.indexer.tail_cache",
        )

    indexer_spec = indexer.get_kv_cache_spec(None)
    state_spec = state.get_kv_cache_spec(None)
    assert len(indexer.kv_cache) == len(state.kv_cache) == 2
    assert isinstance(indexer_spec, AscendMLAAttentionSpec)
    assert indexer_spec.block_size == 256
    assert get_storage_block_size(indexer_spec) == 16
    assert get_kv_cache_compression_ratio(indexer_spec) == 16
    assert indexer_spec.model_version == "glm5_next"
    assert indexer_spec.indexes_kv_by_block_stride
    assert state_spec.block_size == state_spec.sliding_window == 16
    assert state_spec.head_size == 128
    assert state_spec.dtype == torch.float32
    assert state_spec.model_version == "glm5_next"
    assert state_spec.indexes_kv_by_block_stride
    assert indexer.get_attn_backend() is AscendIndexerKPoolBackend
    assert state.get_attn_backend() is AscendIndexerKPoolTailBackend
    assert indexer.get_attn_backend().get_kv_cache_shape(3, 16, 1, 128, cache_dtype_str="auto") == (3, 16, 1, 128)
    assert state.get_attn_backend().get_kv_cache_shape(3, 16, 1, 128, cache_dtype_str="auto") == (3, 2, 16, 128)
    assert set(current_config.compilation_config.static_forward_context) == {
        indexer.prefix,
        state.prefix,
    }


@pytest.mark.parametrize(("num_speculative_tokens", "expected_capacity"), [(0, 4), (1, 8), (3, 8), (7, 16)])
@pytest.mark.parametrize("start_residue", range(4))
def test_indexer_tail_retains_history_after_speculative_rejection(
    num_speculative_tokens, expected_capacity, start_residue
):
    pool_size = 4
    config = SimpleNamespace(
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=num_speculative_tokens) if num_speculative_tokens else None
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, decode_context_parallel_size=1),
        cache_config=SimpleNamespace(block_size=128, enable_prefix_caching=False),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    model_config = SimpleNamespace(
        index_topk=8, index_n_heads=1, index_head_dim=2, index_kpool=pool_size, qk_rope_head_dim=0
    )
    with (
        patch("vllm_ascend.models.glm5next.kv_cache.get_current_vllm_config", return_value=config),
        patch("vllm_ascend.models.glm5next.attention.ReplicatedLinear", return_value=torch.nn.Identity()),
        patch("vllm_ascend.models.glm5next.attention.MergedColumnParallelLinear", return_value=torch.nn.Identity()),
    ):
        indexer = Indexer(
            config,
            model_config,
            hidden_size=4,
            q_lora_rank=4,
            quant_config=None,
            cache_config=SimpleNamespace(block_size=128),
            topk_indices_buffer=torch.empty(8, 11, dtype=torch.int32),
            prefix="model.layers.0.indexer",
        )
    tail_spec = indexer.tail_cache.get_kv_cache_spec(config)
    indexer_spec = indexer.k_cache.get_kv_cache_spec(config)
    capacity = tail_spec.block_size
    assert capacity == expected_capacity
    # Exercise the real resolution/constructor path: private rings are excluded
    # from scheduler-size calculation, but the no-prefix coordinator validates
    # every original group. Raw capacities such as 7 cannot divide that size.
    register_ascend_kv_cache_specs()
    cache_plan = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([indexer.k_cache.prefix], indexer_spec),
            KVCacheGroupSpec([indexer.tail_cache.prefix], tail_spec),
        ],
    )
    scheduler_size, hash_size = _ascend_resolve_kv_cache_block_sizes(cache_plan, config)
    coordinator = get_kv_cache_coordinator(
        kv_cache_config=cache_plan,
        max_model_len=1024,
        max_in_flight_tokens=128,
        enable_caching=False,
        scheduler_block_size=scheduler_size,
        hash_block_size=hash_size,
    )
    assert isinstance(coordinator.single_type_managers[1], KpoolTailManager)
    # Store absolute positions as distinct raw-key values. The verifier writes
    # every candidate before the sampler decides how many tokens to retain.
    start = 4 * pool_size + start_residue
    ring = [-1] * capacity
    for position in range(start + num_speculative_tokens + 1):
        ring[position % capacity] = position
    for accepted in range(1, num_speculative_tokens + 2):
        replay_start = start + accepted
        pool_start = replay_start // pool_size * pool_size
        assert [ring[position % capacity] for position in range(pool_start, replay_start)] == list(
            range(pool_start, replay_start)
        )


def test_indexer_metadata_preserves_raw_request_boundaries():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=16,
            max_num_seqs=2,
        ),
        model_config=SimpleNamespace(max_model_len=512),
    )
    builder = AscendIndexerKPoolMetadataBuilder(
        AscendMLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            **_ratio_kwargs(16),
        ),
        ["model.layers.0.indexer.k_cache"],
        config,
        torch.device("cpu"),
    )
    common = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=torch.tensor([18, 35], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([18, 35], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=3,
        num_input_tokens=5,
        max_seq_len=35,
        block_table_tensor=torch.tensor([[0, -1], [1, 2]], dtype=torch.int32),
        slot_mapping=torch.tensor([16, 17, 32, 33, 34]),
        positions=torch.tensor([16, 17, 32, 33, 34]),
    )

    metadata = builder.build(0, common)

    assert metadata.cum_query_lens.tolist() == [2, 5]
    assert metadata.raw_seq_lens.tolist() == [18, 35]
    assert metadata.seq_lens.tolist() == [1, 2]
    assert metadata.num_tokens == 5
    assert metadata.max_pool_seq_len == 2

    common._seq_lens_cpu = None
    common.seq_lens_cpu = None
    assert builder.build(0, common).max_pool_seq_len == 16

    common.num_reqs = common.num_input_tokens = common.num_actual_tokens = 0
    common._seq_lens_cpu = torch.empty(0, dtype=torch.int32)
    empty = builder.build(0, common)
    assert empty.num_tokens == empty.max_pool_seq_len == 0


def test_indexer_metadata_request_buffers_cover_graph_token_padding():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8,
            max_num_seqs=2,
        ),
        model_config=SimpleNamespace(max_model_len=512),
    )
    builder = AscendIndexerKPoolMetadataBuilder(
        AscendMLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            **_ratio_kwargs(16),
        ),
        ["model.layers.0.indexer.k_cache"],
        config,
        torch.device("cpu"),
    )
    common = SimpleNamespace(
        num_reqs=4,
        num_input_tokens=4,
        num_actual_tokens=2,
        query_start_loc=torch.arange(5, dtype=torch.int32),
        seq_lens=torch.tensor([17, 33, 0, 0], dtype=torch.int32),
        _seq_lens_cpu=None,
        seq_lens_cpu=None,
        positions=torch.tensor([16, 32, 0, 0], dtype=torch.int32),
        slot_mapping=torch.tensor([16, 32, -1, -1], dtype=torch.int64),
        block_table_tensor=torch.zeros((4, 2), dtype=torch.int32),
    )

    metadata = builder.build_for_drafting(common, draft_index=1)

    assert metadata.seq_lens.tolist() == [1, 2, 0, 0]
    assert metadata.cum_query_lens.tolist() == [1, 2, 3, 4]
    assert metadata.raw_seq_lens.tolist() == [17, 33, 0, 0]


def test_indexer_metadata_buffers_are_stable_per_draft_step():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8,
            max_num_seqs=2,
        ),
        model_config=SimpleNamespace(max_model_len=512),
    )
    builder = AscendIndexerKPoolMetadataBuilder(
        AscendMLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            model_version="glm5_next",
            **_ratio_kwargs(16),
        ),
        ["model.layers.0.indexer.k_cache"],
        config,
        torch.device("cpu"),
    )
    common = SimpleNamespace(
        num_reqs=1,
        num_input_tokens=2,
        num_actual_tokens=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([18], dtype=torch.int32),
        _seq_lens_cpu=None,
        seq_lens_cpu=None,
        positions=torch.tensor([16, 17], dtype=torch.int32),
        slot_mapping=torch.tensor([16, 17], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0, 1]], dtype=torch.int32),
    )

    first = builder.build(0, common)
    first_ptrs = (
        first.slot_mapping.data_ptr(),
        first.seq_lens.data_ptr(),
        first.cum_query_lens.data_ptr(),
        first.raw_seq_lens.data_ptr(),
        first.positions.data_ptr(),
    )
    captured = builder.build_for_graph_capture(common, attn_state=object())
    drafted = builder.build_for_drafting(
        common,
        draft_index=1,
        common_ratio_to_sas_metadata={},
    )
    assert captured.slot_mapping.data_ptr() == first.slot_mapping.data_ptr()
    assert drafted.slot_mapping.data_ptr() == first.slot_mapping.data_ptr()
    common.positions.copy_(torch.tensor([32, 33], dtype=torch.int32))
    common.slot_mapping.copy_(torch.tensor([32, 33], dtype=torch.int32))
    common.seq_lens.fill_(34)
    refreshed = builder.build(0, common)
    refreshed_ptrs = (
        refreshed.slot_mapping.data_ptr(),
        refreshed.seq_lens.data_ptr(),
        refreshed.cum_query_lens.data_ptr(),
        refreshed.raw_seq_lens.data_ptr(),
        refreshed.positions.data_ptr(),
    )
    assert refreshed_ptrs == first_ptrs
    assert refreshed.positions.tolist() == [32, 33]
    assert refreshed.raw_seq_lens.tolist() == [34]

    other_step = SimpleNamespace(**vars(common))
    other_step.slot_mapping = common.slot_mapping.clone()
    second = builder.build(0, other_step)
    second_ptrs = (
        second.slot_mapping.data_ptr(),
        second.seq_lens.data_ptr(),
        second.cum_query_lens.data_ptr(),
        second.raw_seq_lens.data_ptr(),
        second.positions.data_ptr(),
    )
    assert all(left != right for left, right in zip(first_ptrs, second_ptrs))
