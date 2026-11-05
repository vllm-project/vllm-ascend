# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NPU cache view/restore smoke; does not claim full-model correctness."""

import os
import uuid

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.models.qwen4_exp.test_layerwise import make_plan
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend


def test_qwen4_layerwise_cache_restore_on_npu():
    torch.npu.set_device(0)
    adapter, groups, plan = make_plan(8)
    backings = {b.backing_id: torch.zeros(b.size_bytes, dtype=torch.int8, device="npu") for b in plan.backings}
    specs = {name: spec for group in groups for name, spec in group.kv_cache_spec.kv_cache_specs.items()}
    caches = {}
    for owner in plan.owners:
        value = adapter.reshape(plan, owner.name, specs[owner.name], backings[plan.view(owner.name).backing_id], 128)
        caches[owner.name] = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    addresses = {name: tuple(t.data_ptr() for t in parts) for name, parts in caches.items()}
    saved = {}
    # Different groups own different block IDs. Restoring one logical owner
    # must not overwrite another group's block or a private resident region.
    ordered = sorted(plan.owners, key=lambda owner: (owner.physical_layer, owner.name))
    for step in range(3):
        for ordinal, owner in enumerate(ordered):
            block = owner.group_id + 1
            for component, tensor in enumerate(caches[owner.name]):
                scale = tensor.shape[0] // plan.num_blocks
                selected = tensor[block * scale : (block + 1) * scale]
                key = (owner.name, component)
                if step:
                    if not owner.resident:
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
    adapter, groups, plan = make_plan(8)
    alignment = 2 * 1024 * 1024
    backings = {}
    for backing in plan.backings:
        raw = torch.zeros(backing.size_bytes + alignment, dtype=torch.int8, device="npu")
        offset = (-raw.data_ptr()) % alignment
        backings[backing.backing_id] = raw[offset : offset + backing.size_bytes]
    specs = {name: spec for group in groups for name, spec in group.kv_cache_spec.kv_cache_specs.items()}
    caches = {}
    for owner in plan.owners:
        value = adapter.reshape(plan, owner.name, specs[owner.name], backings[plan.view(owner.name).backing_id], 128)
        caches[owner.name] = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    addresses = {name: tuple(t.data_ptr() for t in parts) for name, parts in caches.items()}
    store = MemcacheBackend(None, device_id=0)
    namespace = "qwen4-layerwise-test-" + uuid.uuid4().hex
    saved, previous_gvas = {}, {}
    try:
        for backing in backings.values():
            assert store.store.register_buffer(backing.data_ptr(), backing.numel()) == 0
        for step in range(3):
            for ordinal, owner in enumerate(sorted(plan.owners, key=lambda item: (item.physical_layer, item.name))):
                block = owner.group_id + 1
                regions = [region for region in plan.transfer_regions if region.owner == owner.name]
                ptrs = [
                    backings[r.backing_id].data_ptr() + r.offset_bytes + block * r.block_stride_bytes for r in regions
                ]
                sizes = [r.content_bytes_per_block for r in regions]
                if step and not owner.resident:
                    assert store.store.batch_copy(previous_gvas[owner.name], ptrs, sizes, 1) == 0
                for component, tensor in enumerate(caches[owner.name]):
                    scale = tensor.shape[0] // plan.num_blocks
                    selected = tensor[block * scale : (block + 1) * scale]
                    key = (owner.name, component)
                    if step:
                        torch.testing.assert_close(selected.cpu(), saved[key], rtol=0, atol=0)
                    selected.fill_(ordinal + step + component)
                    saved[key] = selected.cpu().clone()
                if not owner.resident:
                    keys = [f"{namespace}-{step}-{ordinal}-{i}" for i in range(len(regions))]
                    gvas = store.batch_alloc(keys, sizes)
                    assert all(gva > 0 for gva in gvas)
                    torch.npu.synchronize()
                    assert store.store.batch_copy(gvas, ptrs, sizes, 0) == 0
                    assert store.batch_write_finish(keys, [0] * len(keys)) == [0] * len(keys)
                    previous_gvas[owner.name] = gvas
        assert {name: tuple(t.data_ptr() for t in parts) for name, parts in caches.items()} == addresses
    finally:
        for backing in backings.values():
            store.unregister_buffer([backing.data_ptr()], [backing.numel()])
        store.close()
