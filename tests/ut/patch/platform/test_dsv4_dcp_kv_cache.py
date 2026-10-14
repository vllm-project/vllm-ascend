# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import VllmConfig
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import KVCacheConfig

import vllm_ascend.patch.platform.patch_kv_cache_utils as kv_cache_utils_patch
from vllm_ascend.core.kv_cache_interface import (
    AscendDSAReplicatedFullMLASpec,
    AscendDSAReplicatedSlidingWindowMLASpec,
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator


class _VllmConfig(SimpleNamespace):
    # Use the real property so changing the prefill budget also changes the
    # recycling-aware admission bound during the preflight's binary search.
    max_in_flight_tokens = VllmConfig.max_in_flight_tokens


def _make_config(*, dcp=2, batched_tokens=128, async_scheduling=False):
    return _VllmConfig(
        cache_config=SimpleNamespace(
            block_size=32,
            enable_prefix_caching=False,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=batched_tokens, disable_hybrid_kv_cache_manager=False),
        model_config=SimpleNamespace(max_model_len=8192),
        max_concurrent_batches=2 if async_scheduling else 1,
        speculative_config=None,
    )


def _make_specs(*, dcp=2, with_mtp=False):
    replicated = {"dcp_scheduler_replication_size": dcp} if dcp > 1 else {}
    full_spec_cls = AscendDSAReplicatedFullMLASpec if dcp > 1 else AscendMLAAttentionSpec
    sliding_spec_cls = AscendDSAReplicatedSlidingWindowMLASpec if dcp > 1 else AscendSlidingWindowMLASpec
    c4 = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.bfloat16,
        tokens_per_state=4,
        model_version="deepseek_v4",
    )
    indexer = full_spec_cls(
        block_size=128,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        scale_dim=1,
        scale_dtype=torch.float16,
        tokens_per_state=4,
        model_version="deepseek_v4",
        **replicated,
    )
    c128 = full_spec_cls(
        block_size=4096,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.bfloat16,
        tokens_per_state=128,
        model_version="deepseek_v4",
        **replicated,
    )
    swa = sliding_spec_cls(
        block_size=32,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=128,
        model_version="deepseek_v4",
        **replicated,
    )
    c4_state = sliding_spec_cls(
        block_size=2,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        sliding_window=8,
        **replicated,
    )
    c128_state = sliding_spec_cls(
        block_size=128,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        sliding_window=128,
        **replicated,
    )
    specs = {
        "c4_attn.0": c4,
        "c4_attn.1": replace(c4),
        "c4_indexer.0": indexer,
        "c4_indexer.1": replace(indexer),
        "c128_attn": c128,
        "swa.0": swa,
        "swa.1": replace(swa),
        "c4_state.0": c4_state,
        "c4_state.1": replace(c4_state),
        "c128_state": c128_state,
    }
    if with_mtp:
        specs["mtp_attn"] = replace(c4)
    return specs


def _make_groups(specs):
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    grouped_specs = kv_cache_utils_patch.group_and_unify_kv_cache_specs(specs)
    return kv_cache_utils_patch._get_kv_cache_groups_uniform_groups(grouped_specs)


def test_dcp_groups_separate_c4_indexer_and_preserve_replicated_pages():
    specs = _make_specs()
    original_sizes = {name: spec.page_size_bytes for name, spec in specs.items()}
    groups = _make_groups(specs)

    assert [group.layer_names for group in groups[:3]] == [
        ["c4_attn.0", "c4_attn.1"],
        ["c4_indexer.0", "c4_indexer.1"],
        ["c128_attn"],
    ]
    assert {name for group in groups for name in group.layer_names} == set(specs)
    assert {name: spec.page_size_bytes for name, spec in specs.items()} == original_sizes
    # C128 state is larger than either full-MLA page. It must retain its own
    # region, while the tiny C4 state must not be padded to the attention page.
    assert specs["c4_state.0"].page_size_bytes == 64
    assert specs["c128_state"].page_size_bytes == 4096


def test_dcp_one_keeps_combined_c4_indexer_group_and_legacy_swa_padding():
    specs = _make_specs(dcp=1)
    # The legacy layout pads state pages into the full-MLA page buckets.
    del specs["c128_state"]
    groups = _make_groups(specs)

    assert groups[0].layer_names == ["c4_attn.0", "c4_attn.1", "c4_indexer.0", "c4_indexer.1"]
    assert groups[1].layer_names == ["c128_attn"]
    assert specs["swa.0"].page_size_bytes == 576
    assert specs["c4_state.0"].page_size_bytes == 576


@pytest.mark.parametrize("with_mtp", [False, True])
def test_dcp_planner_stride_includes_all_page_buckets_and_mtp(with_mtp):
    specs = _make_specs(with_mtp=with_mtp)
    groups = _make_groups(specs)
    config = _make_config()
    page_sizes = [64, 512, 576, 1024, 4096]
    slots = 3 if with_mtp else 2
    bytes_per_block = sum(page_sizes) * slots
    num_blocks = 7
    available_memory = bytes_per_block * num_blocks + bytes_per_block - 1

    assert kv_cache_utils_patch._ascend_pool_bytes_per_block(groups) == bytes_per_block
    result = kv_cache_utils_patch._ascend_get_kv_cache_config_from_groups(config, groups, available_memory)
    assert result.num_blocks == num_blocks
    tuple_stride = sum(page_sizes) * num_blocks
    assert {tensor.size for tensor in result.kv_cache_tensors} == {bytes_per_block * num_blocks}
    assert {name for tensor in result.kv_cache_tensors for name in tensor.layers} == set(specs)
    for tensor in result.kv_cache_tensors:
        page_size = specs[tensor.layers[0]].page_size_bytes
        bucket_offset = sum(size for size in page_sizes if size < page_size) * num_blocks
        assert tensor.block_stride == page_size
        if tensor.layers == ["mtp_attn"]:
            assert tensor.offset == 2 * tuple_stride + bucket_offset
            assert tensor.layer_stride == 0
        else:
            assert tensor.offset == bucket_offset
            assert tensor.layer_stride == tuple_stride
        last_byte = tensor.offset + (len(tensor.layers) - 1) * tensor.layer_stride + num_blocks * page_size
        assert last_byte <= tensor.size

    peak_blocks = 32 + 64 + 2 + 69 + 9 + 3
    assert kv_cache_utils_patch._ascend_max_memory_usage_bytes_from_groups(config, groups) == (
        peak_blocks * bytes_per_block
    )


def test_mixed_dcp_scheduler_alignment_uses_each_group_ownership():
    groups = _make_groups(_make_specs())
    config = _make_config()
    cache = KVCacheConfig(num_blocks=512, kv_cache_tensors=[], kv_cache_groups=groups)

    scheduler_size, hash_size = kv_cache_utils_patch._ascend_resolve_kv_cache_block_sizes(cache, config)
    assert scheduler_size == 4096
    assert hash_size == 4096
    replicated_only = replace(cache, kv_cache_groups=[groups[1]])
    assert kv_cache_utils_patch._ascend_resolve_kv_cache_block_sizes(replicated_only, config) == (128, 128)


def test_mixed_dcp_coordinator_allocates_with_group_ownership_and_frees_blocks():
    specs = {name: spec for name, spec in _make_specs().items() if "state" not in name}
    groups = _make_groups(specs)
    cache = KVCacheConfig(num_blocks=512, kv_cache_tensors=[], kv_cache_groups=groups)
    coordinator = AscendHybridKVCacheCoordinator(
        cache,
        max_model_len=8192,
        use_eagle=False,
        enable_caching=False,
        enable_kv_cache_events=False,
        dcp_world_size=2,
        pcp_world_size=1,
        hash_block_size=32,
        scheduler_block_size=4096,
        max_in_flight_tokens=128,
    )
    managers = coordinator.single_type_managers
    assert [manager.dcp_world_size for manager in managers] == [2, 1, 1, 1]
    assert [manager.block_size for manager in managers] == [256, 128, 4096, 32]
    assert coordinator.lcm_block_size == 4096
    request_id = "old"
    for manager in managers:
        manager.allocate_new_blocks(request_id, 4096, 4096)
    assert [len(manager.req_to_blocks[request_id]) for manager in managers] == [16, 32, 1, 128]
    assert coordinator.block_pool.get_num_free_blocks() == cache.num_blocks - 1 - sum([16, 32, 1, 128])
    coordinator.free(request_id)
    assert coordinator.block_pool.get_num_free_blocks() == cache.num_blocks - 1


@pytest.mark.parametrize("async_scheduling,expected_tokens", [(False, 63), (True, 31)])
def test_dcp_preflight_fits_real_peak_reserves_null_block_and_uses_smallest_rank(
    monkeypatch, async_scheduling, expected_tokens
):
    config = _make_config(async_scheduling=async_scheduling)
    specs = _make_specs()
    groups = _make_groups(copy.deepcopy(specs))
    stride = kv_cache_utils_patch._ascend_pool_bytes_per_block(groups)
    # At 63 in-flight tokens the six groups need 144 block IDs. The 145th
    # physical block is reserved by BlockPool, so 64 tokens cannot fit.
    available_memory = [152 * stride, 145 * stride]
    workers = [specs, copy.deepcopy(specs)]
    original_workers = copy.deepcopy(workers)
    upstream = MagicMock(return_value=[])
    monkeypatch.setattr(kv_cache_utils_patch, "_orig_get_kv_cache_configs", upstream)

    assert kv_cache_utils_patch._ascend_get_kv_cache_configs(config, workers, available_memory) == []
    assert config.scheduler_config.max_num_batched_tokens == expected_tokens
    assert config.model_config.max_model_len == 8192
    assert workers == original_workers
    upstream.assert_called_once_with(config, workers, available_memory)
    assert kv_cache_utils_patch._ascend_max_memory_usage_bytes_from_groups(config, groups) == 144 * stride
    config.scheduler_config.max_num_batched_tokens += 1
    assert kv_cache_utils_patch._ascend_max_memory_usage_bytes_from_groups(config, groups) > 144 * stride


def test_dcp_preflight_keeps_requested_budget_when_minimum_cannot_fit(monkeypatch):
    config = _make_config()
    workers = [_make_specs()]
    upstream = MagicMock(side_effect=ValueError("insufficient KV cache memory"))
    monkeypatch.setattr(kv_cache_utils_patch, "_orig_get_kv_cache_configs", upstream)

    with pytest.raises(ValueError, match="insufficient KV cache memory"):
        kv_cache_utils_patch._ascend_get_kv_cache_configs(config, workers, [1])

    assert config.scheduler_config.max_num_batched_tokens == 128
    upstream.assert_called_once_with(config, workers, [1])


@pytest.mark.parametrize("excluded", ["dcp_one", "pcp", "pp", "override", "other_model", "no_specs", "fits"])
def test_dcp_preflight_preserves_non_target_or_already_fitting_config(monkeypatch, excluded):
    config = _make_config()
    workers = [_make_specs()]
    if excluded == "dcp_one":
        config.parallel_config.decode_context_parallel_size = 1
    elif excluded == "pcp":
        config.parallel_config.prefill_context_parallel_size = 2
    elif excluded == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    elif excluded == "override":
        config.cache_config.num_gpu_blocks_override = 1
    elif excluded == "other_model":
        workers = [{"attn": replace(workers[0]["c4_attn.0"], model_version=None)}]
    elif excluded == "no_specs":
        workers = []
    upstream = MagicMock(return_value=[])
    monkeypatch.setattr(kv_cache_utils_patch, "_orig_get_kv_cache_configs", upstream)
    grouping = MagicMock(wraps=kv_cache_utils_patch.vllm.v1.core.kv_cache_utils.get_kv_cache_groups)
    monkeypatch.setattr(kv_cache_utils_patch.vllm.v1.core.kv_cache_utils, "get_kv_cache_groups", grouping)
    available_memory = [1 << 30] if excluded == "fits" else [1]

    assert kv_cache_utils_patch._ascend_get_kv_cache_configs(config, workers, available_memory) == []
    assert config.scheduler_config.max_num_batched_tokens == 128
    upstream.assert_called_once_with(config, workers, available_memory)
    if excluded == "fits":
        grouping.assert_called_once()
    else:
        grouping.assert_not_called()
