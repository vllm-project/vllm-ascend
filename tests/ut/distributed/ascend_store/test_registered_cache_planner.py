# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-independent byte-window planning with explicit eligibility rules."""

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import LayerwiseCachePlanner
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_views import allocate_cache_backing


def test_component_views_require_both_sources_before_same_layer_reuse():
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.uint8)
    names = ["model.layers.0.a", "model.layers.1.a", "model.layers.2.a", "model.layers.2.b"]
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_groups=[KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)],
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * spec.page_size_bytes, layers=[name], layer_stride=0, block_stride=spec.page_size_bytes
            )
            for name in names
        ],
    )
    model = SimpleNamespace(
        model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 3),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )
    planner = LayerwiseCachePlanner(
        config,
        model,
        {"layerwise_num_shared_buffers": 2, "layerwise_independent_layers": []},
        spec_reuse_rules={FullAttentionSpec: True},
    )
    planner.apply()
    assert config.layerwise_reuse_layout.prefetch_layer_map == {2: (0, 1)}
    allocations = []

    def allocate(size):
        allocations.append(size)
        return torch.zeros(size, dtype=torch.uint8)

    views = allocate_cache_backing(config, dict.fromkeys(names, spec), allocate)
    assert allocations == [4 * spec.page_size_bytes]
    assert views[names[0]].data_ptr() == views[names[2]].data_ptr()
    assert views[names[1]].data_ptr() == views[names[3]].data_ptr()
    assert views[names[0]].data_ptr() != views[names[1]].data_ptr()
    views[names[0]].fill_(17)
    views[names[1]].fill_(23)
    assert torch.all(views[names[2]] == 17)
    assert torch.all(views[names[3]] == 23)


def test_unknown_spec_rule_leaves_native_storage_unchanged():
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.uint8)
    names = ["model.layers.0.a", "model.layers.1.a"]
    tensors = [
        KVCacheTensor(size=spec.page_size_bytes, layers=[name], layer_stride=0, block_stride=spec.page_size_bytes)
        for name in names
    ]
    config = KVCacheConfig(num_blocks=1, kv_cache_groups=[KVCacheGroupSpec(names, spec)], kv_cache_tensors=tensors)
    planner = LayerwiseCachePlanner(
        config,
        SimpleNamespace(model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 2)),
        {"layerwise_num_shared_buffers": 1, "layerwise_independent_layers": []},
        spec_reuse_rules={},
    )
    planner.apply()
    assert not config.layerwise_reuse_layout.has_layer_reuse
    assert config.kv_cache_tensors is tensors
