# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention import dsa_attn_kv_plan
from vllm_ascend.core.kv_cache_interface import (
    AscendDSAReplicatedFullMLASpec,
    AscendDSAReplicatedSlidingWindowMLASpec,
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    get_dsv4_scheduler_dcp_world_size,
    get_storage_block_size,
)
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.deepseek_v4 import indexer
from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorStateCache
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache
from vllm_ascend.models.layer.attention import layer as attention_layer


@pytest.fixture(params=[(1, 1, False), (2, 1, True), (2, 2, False)], ids=["default", "dcp2", "pcp2-dcp2"])
def cache_config(request, monkeypatch):
    dcp_size, pcp_size, replicated = request.param
    profile = get_hardware_profile(AscendDeviceType.A2)
    for module in (attention_layer, indexer, dsa_attn_kv_plan):
        monkeypatch.setattr(module, "get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(attention_layer, "DSV4_BLOCK_SIZES", attention_layer.get_dsv4_block_sizes())
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=32, cache_dtype="auto"),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=pcp_size,
        ),
    )
    return config, replicated


@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_dsv4_main_cache_shards_only_c4_history(cache_config, compress_ratio):
    config, replicated = cache_config
    cache = attention_layer.DSAAttention.__new__(attention_layer.DSAAttention)
    cache.compress_ratio = compress_ratio
    cache.head_size = 512
    cache.kv_cache_dtype = "auto"

    spec = cache.get_kv_cache_spec(config)

    expected_cls = AscendDSAReplicatedFullMLASpec if replicated and compress_ratio == 128 else AscendMLAAttentionSpec
    assert type(spec) is expected_cls
    assert spec.block_size == (128 if compress_ratio == 4 else 4096)
    assert get_storage_block_size(spec) == 32
    assert spec.page_size_bytes == 32768
    if replicated:
        assert get_dsv4_scheduler_dcp_world_size(spec, 2) == (2 if compress_ratio == 4 else 1)


def test_dsv4_indexer_history_is_replicated_without_physical_lanes(cache_config):
    config, replicated = cache_config
    cache = indexer.AscendDeepseekV4IndexerCache.__new__(indexer.AscendDeepseekV4IndexerCache)
    cache.head_dim = 128
    cache.dtype = torch.int8
    cache.compress_ratio = 4
    cache.cache_config = config.cache_config

    spec = cache.get_kv_cache_spec(config)

    expected_cls = AscendDSAReplicatedFullMLASpec if replicated else AscendMLAAttentionSpec
    assert type(spec) is expected_cls
    assert spec.block_size == 128
    assert get_storage_block_size(spec) == 32
    # Each A2 row holds 128 int8 K elements and one FP16 scale.
    assert spec.page_size_bytes == 4160
    assert spec.scale_dtype == torch.float16
    if replicated:
        assert spec.dcp_scheduler_replication_size == 2
        assert get_dsv4_scheduler_dcp_world_size(spec, 2) == 1


def test_dsv4_swa_preserves_physical_window_pages_under_dcp(cache_config):
    config, replicated = cache_config
    cache = AscendDeepseekV4SWACache.__new__(AscendDeepseekV4SWACache)
    cache.head_dim = 512
    cache.dtype = torch.bfloat16
    cache.block_size = 32
    cache.window_size = 128
    cache.cache_config = config.cache_config

    spec = cache.get_kv_cache_spec(config)

    expected_cls = AscendDSAReplicatedSlidingWindowMLASpec if replicated else AscendSlidingWindowMLASpec
    assert type(spec) is expected_cls
    assert spec.block_size == 32
    assert spec.sliding_window == 128
    assert spec.page_size_bytes == 32768
    if replicated:
        assert spec.dcp_scheduler_replication_size == 2
        assert get_dsv4_scheduler_dcp_world_size(spec, 2) == 1


@pytest.mark.parametrize(
    ("state_dim", "compress_ratio", "block_size", "window_size", "page_bytes"),
    [(512, 4, 2, 8, 4160), (2048, 4, 2, 8, 32768), (1024, 128, 8, 128, 32768)],
    ids=["c4-indexer", "c4-main", "c128-main"],
)
def test_dsv4_state_preserves_padded_pages_and_global_ownership(
    cache_config, state_dim, compress_ratio, block_size, window_size, page_bytes
):
    config, replicated = cache_config
    cache = AscendCompressorStateCache.__new__(AscendCompressorStateCache)
    cache.state_dim = state_dim
    cache.dtype = torch.float32
    cache.compress_ratio = compress_ratio
    cache.block_size = block_size
    cache.sliding_window = window_size

    spec = cache.get_kv_cache_spec(config)

    expected_cls = AscendDSAReplicatedSlidingWindowMLASpec if replicated else AscendSlidingWindowMLASpec
    assert type(spec) is expected_cls
    assert spec.block_size == block_size
    assert spec.head_size == state_dim
    assert spec.sliding_window == window_size
    assert spec.page_size_padded == spec.page_size_bytes == page_bytes
    if replicated:
        assert spec.dcp_scheduler_replication_size == 2
        assert get_dsv4_scheduler_dcp_world_size(spec, 2) == 1
