# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Configuration tests for native KV offloading on Ascend."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CacheConfig, KVTransferConfig, ParallelConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import build_offloading_config
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.utils import refresh_block_size

UPSTREAM_BLOCK_SIZE = 16
NPU_BLOCK_SIZE = 128


def test_offloading_chunk_matches_effective_npu_block_size() -> None:
    cache_config = CacheConfig(block_size=UPSTREAM_BLOCK_SIZE, enable_prefix_caching=True)
    transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"block_size": UPSTREAM_BLOCK_SIZE},
    )
    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        kv_transfer_config=transfer_config,
        parallel_config=ParallelConfig(),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        model_config=SimpleNamespace(
            model="facebook/opt-125m",
            hf_config=SimpleNamespace(model_type="opt"),
            is_hybrid=False,
            use_mla=False,
            dtype=torch.bfloat16,
            max_model_len=256,
            get_total_num_kv_heads=lambda: 1,
        ),
        kv_events_config=None,
        speculative_config=None,
    )
    refresh_block_size(vllm_config)
    assert cache_config.block_size == NPU_BLOCK_SIZE

    spec = FullAttentionSpec(
        block_size=cache_config.block_size,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["model.layers.0.self_attn"], kv_cache_spec=spec)],
    )
    with pytest.raises(ValueError, match=rf"must be a multiple.*\({NPU_BLOCK_SIZE} tokens\)"):
        build_offloading_config(vllm_config, kv_cache_config)

    transfer_config.kv_connector_extra_config["block_size"] = NPU_BLOCK_SIZE
    offloading_config = build_offloading_config(vllm_config, kv_cache_config)
    assert offloading_config.cache.blocks_per_chunk == 1
    assert offloading_config.cache.tokens_per_hash == NPU_BLOCK_SIZE
    assert offloading_config.groups[0].tokens_per_block == NPU_BLOCK_SIZE
