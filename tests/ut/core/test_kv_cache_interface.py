# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import (
    AscendDSAReplicatedFullMLASpec,
    AscendDSAReplicatedSlidingWindowMLASpec,
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    get_dsv4_dcp_replication_size,
    get_dsv4_scheduler_dcp_world_size,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
)


def _mla_spec(spec_cls=AscendMLAAttentionSpec, **kwargs):
    return spec_cls(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        **kwargs,
    )


def test_get_storage_block_size_and_dcp_memory():
    spec = _mla_spec()
    # On main, storage_block_size is an optional dataclass field and may be
    # None. Ascend derives physical rows from block_size / compression ratio.
    expected = spec.block_size // get_kv_cache_compression_ratio(spec)
    assert get_storage_block_size(spec) == expected

    uniform = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"layer": spec})
    assert get_storage_block_size(uniform) == expected

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
    )
    assert spec.max_memory_usage_bytes(vllm_config) > 0


def test_sliding_window_mla_storage_and_page_size():
    spec = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    assert spec.storage_block_size == 16
    assert spec.real_page_size_bytes == 16 * 128 * 2


@pytest.mark.parametrize(
    ("dcp_size", "pcp_size", "expected"),
    [(1, 1, 1), (2, 1, 2), (4, 1, 4), (2, 2, 1)],
)
def test_dsv4_replication_is_enabled_only_for_dcp(dcp_size, pcp_size, expected):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=pcp_size,
        ),
    )
    assert get_dsv4_dcp_replication_size(config) == expected


@pytest.mark.parametrize("dcp_size", [1, 2])
def test_dsv4_full_history_uses_global_blocks_without_widening_pages(dcp_size):
    sharded = AscendMLAAttentionSpec(
        block_size=32,
        tokens_per_state=4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
    )
    replicated = AscendDSAReplicatedFullMLASpec(
        block_size=32,
        tokens_per_state=4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        dcp_scheduler_replication_size=dcp_size,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=65),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size),
    )

    # Eight physical rows each hold 128 K bytes and a two-byte scale.
    assert get_storage_block_size(replicated) == 8
    assert replicated.page_size_bytes == sharded.page_size_bytes == 1040
    # 65 global tokens need three blocks. DCP=2 stores 33 local tokens in
    # the sharded cache (two blocks), but all three blocks in the Indexer.
    assert sharded.max_memory_usage_bytes(config) == (3120 if dcp_size == 1 else 2080)
    assert replicated.max_memory_usage_bytes(config) == 3120
    assert replicated.max_num_blocks_per_req(config, 65) == 3
    assert not replicated.dcp_sharded


@pytest.mark.parametrize(
    ("max_model_len", "in_flight_tokens", "expected_bytes"),
    [(65, 5, 1024), (65, 6, 1280), (9, 6, 1024)],
)
def test_dsv4_replicated_window_memory_accounts_for_global_chunk_and_padding(
    max_model_len, in_flight_tokens, expected_bytes
):
    spec = AscendDSAReplicatedSlidingWindowMLASpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        sliding_window=8,
        page_size_padded=256,
        dcp_scheduler_replication_size=2,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
        max_in_flight_tokens=in_flight_tokens,
    )

    # The seven retained rows plus five in-flight rows span at most four
    # blocks when the window starts inside a block. A sixth in-flight row
    # can require a fifth block; the model length caps that held history.
    assert spec.real_page_size_bytes == 128
    assert spec.page_size_bytes == 256
    assert spec.max_memory_usage_bytes(config) == expected_bytes
    # Position-indexed worker tables also contain recycled/null blocks.
    assert spec.max_num_blocks_per_req(config, 65) == 17


@pytest.mark.parametrize("spec_cls", [AscendDSAReplicatedFullMLASpec, AscendDSAReplicatedSlidingWindowMLASpec])
def test_dsv4_replicated_spec_merge_preserves_scheduler_ownership(spec_cls):
    kwargs = {"sliding_window": 8} if spec_cls is AscendDSAReplicatedSlidingWindowMLASpec else {}
    spec = spec_cls(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        page_size_padded=256,
        dcp_scheduler_replication_size=2,
        **kwargs,
    )
    merged = spec_cls.merge([spec, spec])

    assert type(merged) is spec_cls
    assert merged.dcp_scheduler_replication_size == 2
    assert merged.page_size_bytes == 256
    assert get_dsv4_scheduler_dcp_world_size(merged, 2) == 1

    other = replace(spec, dcp_scheduler_replication_size=4)
    with pytest.raises(AssertionError):
        spec_cls.merge([spec, other])


@pytest.mark.parametrize("wrapped", [False, True])
def test_dsv4_scheduler_dcp_width_distinguishes_sharded_and_replicated_specs(wrapped):
    sharded = _mla_spec()
    replicated = _mla_spec(AscendDSAReplicatedFullMLASpec, dcp_scheduler_replication_size=2)
    if wrapped:
        sharded = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"main": sharded})
        replicated = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"indexer": replicated})

    assert get_dsv4_scheduler_dcp_world_size(sharded, 1) == 1
    assert get_dsv4_scheduler_dcp_world_size(sharded, 2) == 2
    assert get_dsv4_scheduler_dcp_world_size(replicated, 2) == 1
    with pytest.raises(ValueError, match="must match DCP world size"):
        get_dsv4_scheduler_dcp_world_size(replicated, 4)


def test_dsv4_uniform_group_rejects_mixed_scheduler_ownership():
    replicated = _mla_spec(AscendDSAReplicatedFullMLASpec, dcp_scheduler_replication_size=2)
    mixed = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"main": _mla_spec(), "indexer": replicated})
    with pytest.raises(ValueError, match="same scheduler replication width"):
        get_dsv4_scheduler_dcp_world_size(mixed, 2)
