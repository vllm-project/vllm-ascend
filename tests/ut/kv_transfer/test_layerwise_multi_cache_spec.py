# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import plan_layerwise_cache_reuse
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_layerwise_kv_cache_specs,
)


@pytest.mark.parametrize(
    "components",
    [
        ("attn", "indexer.k_cache", "indexer.compressor.state_cache", "compressor.state_cache", "swa_cache"),
        ("attn", "indexer.k_cache"),
        ("attn", "swa_cache"),
        ("indexer.k_cache",),
        ("a.indexer.k_cache", "b.indexer.k_cache"),
    ],
)
def test_deepseek_v4_native_groups_keep_every_cache_entry(components):
    groups = []
    tensors = []
    specs = {}
    for index, component in enumerate(components):
        spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8 * (index + 1), dtype=torch.int8)
        names = [f"model.layers.{layer}.self_attn.{component}" for layer in range(4)]
        groups.append(KVCacheGroupSpec(names, spec))
        for name in names:
            specs[name] = spec
            tensors.append(
                KVCacheTensor(
                    size=spec.page_size_bytes, layers=[name], layer_stride=0, block_stride=spec.page_size_bytes
                )
            )
    config = KVCacheConfig(num_blocks=1, kv_cache_tensors=tensors, kv_cache_groups=groups)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="deepseek_v4")),
        kv_transfer_config=SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={"use_layerwise": True, "layerwise_num_shared_buffers": 1},
        ),
    )

    assert get_layerwise_kv_cache_specs(config) == specs
    assert plan_layerwise_cache_reuse(config, vllm_config) is None
    assert config.kv_cache_tensors == tensors
