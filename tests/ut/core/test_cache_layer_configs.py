# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

from tests.kv_cache_groups import ascend_ring_spec_type, ring_spec_type
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.core.kv_cache_layers import (
    HAS_LAYER_CONFIG,
    CacheGroup,
    FullAttentionCacheLayer,
    group_block_size,
    group_layer_specs,
)
from vllm_ascend.core.single_type_kv_cache_manager import AscendCircularBufferManager
from vllm_ascend.models.deepseek_v4.cache_config import _place_deepseek_v4_cache


def _layout(width):
    return AscendMLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=width, dtype=torch.bfloat16)


class _Layer(FullAttentionCacheLayer):
    def __init__(self, layout):
        self.layout = layout

    def _get_cache_layout(self, config):
        return self.layout


def _group(members):
    if HAS_LAYER_CONFIG:
        configs = {name: _Layer(spec).get_kv_cache_config(None) for name, spec in members.items()}
        assert all(c.manager_cls is FullAttentionManager for c in configs.values())
        return CacheGroup.from_layers(configs)
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    specs = {name: _Layer(spec).get_kv_cache_spec(None) for name, spec in members.items()}
    return CacheGroup(list(specs), UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs=specs))


def test_shared_tuple_preserves_component_offsets_and_block_strides():
    small, large = _layout(16), _layout(64)
    groups = [_group({"index": small, "long": large}), _group({"state": replace(large)})]
    assert group_block_size(groups[0]) == 16
    assert group_layer_specs(groups[0]) == {"index": small, "long": large}

    tensors = _place_deepseek_v4_cache(groups, 3)
    assert [(t.layers, t.size, t.offset, t.layer_stride, t.block_stride) for t in tensors] == [
        (["index"], 7680, 0, 7680, 512),
        (["long"], 7680, 1536, 7680, 2048),
        (["state"], 7680, 1536, 7680, 2048),
    ]


def test_ascend_ring_claims_one_page_and_reports_it_for_zeroing():
    ring = ascend_ring_spec_type(block_size=32, num_kv_heads=1, head_size=1024, head_size_v=0, dtype=torch.float32)
    assert ring.page_size_bytes == 131072
    allocation = (
        CacheGroup.from_layers({"ring": AscendCircularBufferManager.layer_config(ring)}) if HAS_LAYER_CONFIG else ring
    )
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=32)
    manager = AscendCircularBufferManager(
        allocation,
        pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=128,
        needs_kv_cache_zeroing=True,
    )
    free = pool.get_num_free_blocks()
    claimed = manager.allocate_new_blocks("req", 256, 256)
    assert manager.take_new_block_ids() == [claimed[0].block_id]
    assert manager.allocate_new_blocks("req", 512, 512) == []
    manager.free("req")
    assert pool.get_num_free_blocks() == free


def test_plain_attention_with_ring_policy_preserves_its_physical_page():
    if not HAS_LAYER_CONFIG:
        layout = ring_spec_type(block_size=32, num_kv_heads=1, head_size=16, head_size_v=0, dtype=torch.float32)
        assert layout.page_size_bytes == 2048
        assert not layout.uses_slot_mapping
        return
    from types import SimpleNamespace

    from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager, FullAttentionManager
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroup

    from vllm_ascend.core.kv_cache_interface import is_circular_cache_group
    from vllm_ascend.core.kv_cache_layer_config_builder import AscendKVCacheConfigBuilder
    from vllm_ascend.core.kv_cache_layers import is_plain_full_attention

    layout = AttentionSpec(block_size=32, num_kv_heads=1, head_size=16, head_size_v=0, dtype=torch.float32)
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="test"), supports_kv_cache_grouping=False),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    ring_config = CircularBufferManager.layer_config(layout)
    assert is_circular_cache_group(KVCacheGroup.from_layers({"state": ring_config}))
    builder = AscendKVCacheConfigBuilder()
    (full,) = builder.get_kv_cache_groups(runtime, {"full": FullAttentionManager.layer_config(layout)})
    (group,) = builder.get_kv_cache_groups(runtime, {"state": ring_config})
    assert not is_circular_cache_group(full)
    assert is_plain_full_attention(layout, full)
    assert not is_plain_full_attention(layout, group)
    assert full.uses_slot_mapping
    ring = group_layer_specs(group)["state"]
    assert ring is layout
    assert ring.page_size_bytes == layout.page_size_bytes == 2048
    assert group.manager_cls is CircularBufferManager
    assert not group.uses_slot_mapping
