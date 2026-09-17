# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.worker.flash_kv_cache import create_flash_kernel_block_view


@pytest.mark.parametrize("head_slots", [1, 4])
def test_blh_kernel_subpages_keep_manager_ownership_and_layer_isolation(head_slots):
    blocks, layers, manager_size, kernel_size = 4, 3, 768, 128
    ratio = manager_size // kernel_size
    spec = FullAttentionSpec(
        block_size=manager_size,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        num_head_slots=head_slots,
        state_content_bytes=128,
    )
    page = spec.unpadded_page_size_bytes
    pitch = layers * page
    prefix = 64
    backing = torch.full((prefix + blocks * pitch + 64,), -91, dtype=torch.int8)
    config = get_kv_cache_config_from_groups(
        SimpleNamespace(
            cache_config=SimpleNamespace(
                get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
                num_gpu_blocks_override=None,
                prefix_cache_retention_interval=None,
            )
        ),
        [KVCacheGroupSpec(layer_names=[f"layer{i}" for i in range(layers)], kv_cache_spec=spec)],
        blocks * pitch,
    )
    descriptor = config.kv_cache_tensors[0]
    assert descriptor.block_stride == pitch and descriptor.layer_stride == page
    views, raw_views = [], []
    for layer in range(layers):
        raw = backing.as_strided((blocks, page), (pitch, 1), prefix + layer * page)
        view = create_flash_kernel_block_view(raw, spec, kernel_size, KVCacheLayout.BLHNC, descriptor, layer)
        assert view.shape == (blocks * ratio, head_slots, kernel_size, 64)
        assert view.stride(0) == pitch // ratio // 2
        assert view.storage_offset() * 2 == prefix + layer * page // ratio
        assert view.untyped_storage().data_ptr() == backing.data_ptr()
        views.append(view)
        raw_views.append(raw)
    # Manager block 1 belongs to attention; its subpages may interleave layers.
    for layer, view in enumerate(views):
        for subpage in range(ratio):
            view[ratio + subpage].fill_(layer * ratio + subpage + 1)
    # Manager block 2 belongs to a recurrent group using its original layout.
    raw_views[1][2, :128].fill_(-13)
    for layer, view in enumerate(views):
        for subpage in range(ratio):
            assert torch.all(view[ratio + subpage] == layer * ratio + subpage + 1)
    touched = torch.zeros_like(backing, dtype=torch.bool)
    touched[prefix + pitch : prefix + 2 * pitch] = True
    touched[prefix + 2 * pitch + page : prefix + 2 * pitch + page + 128] = True
    assert torch.all(backing[~touched] == -91)


def test_dense_replicated_draft_splits_each_lane_without_changing_its_region():
    blocks, replication, manager_size, kernel_size = 3, 2, 768, 128
    spec = FullAttentionSpec(
        block_size=manager_size,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
        num_head_slots=4,
        state_content_bytes=128,
    )
    page = spec.unpadded_page_size_bytes
    prefix = 96
    backing = torch.full((prefix + blocks * replication * page + 32,), -91, dtype=torch.int8)
    raw = backing.as_strided((blocks * replication, page), (page, 1), prefix)
    descriptor = KVCacheTensor(
        size=backing.numel(),
        layers=["draft"],
        offset=prefix,
        layer_stride=0,
        block_stride=replication * page,
    )
    cache = create_flash_kernel_block_view(raw, spec, kernel_size, KVCacheLayout.BLHNC, descriptor, 0)
    ratio = manager_size // kernel_size
    assert cache.shape == (blocks * replication * ratio, 4, kernel_size, 64)
    assert cache.storage_offset() * 2 == prefix
    for physical_page in range(cache.shape[0]):
        cache[physical_page].fill_(physical_page + 1)
    for physical_page in range(cache.shape[0]):
        assert torch.all(cache[physical_page] == physical_page + 1)
    assert torch.all(backing[:prefix] == -91) and torch.all(backing[-32:] == -91)


def test_kernel_split_rejects_indivisible_physical_stride():
    spec = FullAttentionSpec(block_size=768, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    page = spec.unpadded_page_size_bytes
    backing = torch.empty(2 * (page + 1), dtype=torch.int8)
    raw = backing.as_strided((2, page), (page + 1, 1))
    descriptor = KVCacheTensor(size=backing.numel(), layers=["target"], offset=0, layer_stride=0, block_stride=page + 1)
    with pytest.raises(ValueError, match="divide evenly"):
        create_flash_kernel_block_view(raw, spec, 128, KVCacheLayout.BLHNC, descriptor, 0)
