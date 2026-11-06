# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NPU cache view/restore smoke; does not claim full-model correctness."""

import os
import uuid

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.models.qwen4_exp.test_layerwise import make_config, make_views
from vllm_ascend.core.layerwise_cache_config import get_layerwise_layout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend
from vllm_ascend.models.qwen4_exp.cache_config import build_qwen4_kv_cache_layout


def test_qwen4_layerwise_cache_restore_on_npu():
    torch.npu.set_device(0)
    config = make_config(8)
    original = build_qwen4_kv_cache_layout(config.kv_cache_groups, config.num_blocks)
    reuse = get_layerwise_layout(config)
    backings, caches = make_views(config, device="npu", alignment=2 * 1024 * 1024)
    addresses = {name: tuple(t.data_ptr() for t in parts) for name, parts in caches.items()}
    saved = {}
    # Different groups own different block IDs. Restoring one logical owner
    # must not overwrite another group's block or a private resident region.
    ordered = sorted(
        original.owners, key=lambda owner: (int(owner.layer_name.split(".layers.")[1].split(".")[0]), owner.layer_name)
    )
    for step in range(3):
        for ordinal, owner in enumerate(ordered):
            block = owner.group_id + 1
            for component, tensor in enumerate(caches[owner.layer_name]):
                scale = tensor.shape[0] // config.num_blocks
                selected = tensor[block * scale : (block + 1) * scale]
                key = (owner.layer_name, component)
                if step:
                    if owner.layer_name in reuse.rotating_layer_names:
                        selected.copy_(saved[key].to("npu"))
                    torch.testing.assert_close(selected.cpu(), saved[key], rtol=0, atol=0)
                selected.fill_(ordinal + step + component)
                saved[key] = selected.cpu().clone()
        torch.npu.synchronize()
    assert {name: tuple(t.data_ptr() for t in parts) for name, parts in caches.items()} == addresses


@pytest.mark.skipif(not os.getenv("MMC_LOCAL_CONFIG_PATH"), reason="Requires an existing Memcache service")
def test_qwen4_layerwise_regions_roundtrip_through_memcache():
    """Restore actual segmented owner regions after their NPU slots were reused."""
    torch.npu.set_device(0)
    config = make_config(8)
    original = build_qwen4_kv_cache_layout(config.kv_cache_groups, config.num_blocks)
    reuse = get_layerwise_layout(config)
    backings, caches = make_views(config, device="npu", alignment=2 * 1024 * 1024)
    addresses = {name: tuple(t.data_ptr() for t in parts) for name, parts in caches.items()}
    store = MemcacheBackend(None, device_id=0)
    namespace = "qwen4-layerwise-test-" + uuid.uuid4().hex
    saved, previous_gvas = {}, {}
    try:
        for backing in backings.values():
            assert store.store.register_buffer(backing.data_ptr(), backing.numel()) == 0
        for step in range(3):
            for ordinal, owner in enumerate(
                sorted(
                    original.owners,
                    key=lambda item: (int(item.layer_name.split(".layers.")[1].split(".")[0]), item.layer_name),
                )
            ):
                block = owner.group_id + 1
                parts = caches[owner.layer_name]
                ptrs, sizes = [], []
                for tensor in parts:
                    pages = tensor.shape[0] // config.num_blocks
                    for page in range(pages):
                        selected = tensor[block * pages + page]
                        ptrs.append(selected.data_ptr())
                        sizes.append(selected.numel() * selected.element_size())
                if step and owner.layer_name in reuse.rotating_layer_names:
                    assert store.store.batch_copy(previous_gvas[owner.layer_name], ptrs, sizes, 1) == 0
                for component, tensor in enumerate(caches[owner.layer_name]):
                    scale = tensor.shape[0] // config.num_blocks
                    selected = tensor[block * scale : (block + 1) * scale]
                    key = (owner.layer_name, component)
                    if step:
                        torch.testing.assert_close(selected.cpu(), saved[key], rtol=0, atol=0)
                    selected.fill_(ordinal + step + component)
                    saved[key] = selected.cpu().clone()
                if owner.layer_name in reuse.rotating_layer_names:
                    keys = [f"{namespace}-{step}-{ordinal}-{i}" for i in range(len(ptrs))]
                    gvas = store.batch_alloc(keys, sizes)
                    assert all(gva > 0 for gva in gvas)
                    torch.npu.synchronize()
                    assert store.store.batch_copy(gvas, ptrs, sizes, 0) == 0
                    assert store.batch_write_finish(keys, [0] * len(keys)) == [0] * len(keys)
                    previous_gvas[owner.layer_name] = gvas
        assert {name: tuple(t.data_ptr() for t in parts) for name, parts in caches.items()} == addresses
    finally:
        for backing in backings.values():
            store.unregister_buffer([backing.data_ptr()], [backing.numel()])
        store.close()
