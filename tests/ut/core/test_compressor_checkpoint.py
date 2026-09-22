# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host tests of page ownership; these do not emulate compressor math.

The module has only typing-time vLLM dependencies. Load it directly so these
tests also run on a host without vLLM, torch or torch_npu installed:
pytest --confcutdir=tests/ut/core tests/ut/core/test_compressor_checkpoint.py
"""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/core/compressor_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("_compressor_checkpoint_test_module", _SOURCE)
checkpoint = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = checkpoint
_SPEC.loader.exec_module(checkpoint)


@dataclass
class Block:
    block_id: int
    ref_cnt: int = 0


class BlockPool:
    """Reference-counted allocator with destructive page reuse."""

    def __init__(self, count=100):
        self.blocks = [Block(i) for i in range(count)]
        self.payload = {i: [0] * 8 for i in range(count)}

    def get_num_free_blocks(self):
        return sum(block.ref_cnt == 0 for block in self.blocks)

    def get_new_blocks(self, count):
        blocks = [block for block in self.blocks if block.ref_cnt == 0][:count]
        assert len(blocks) == count
        for block in blocks:
            block.ref_cnt = 1
            self.payload[block.block_id] = [-1] * 8
        return blocks

    def touch(self, blocks):
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks):
        for block in blocks:
            assert block.ref_cnt > 0
            block.ref_cnt -= 1

    def copy(self, plan):
        for operation in plan.copies:
            self.payload[operation.destination] = self.payload[operation.source].copy()


def rings(pool, layout):
    return {group: pool.get_new_blocks(count) for group, count in layout.items()}


def key(position=16384, prefix=b"prefix-a"):
    return checkpoint.TailCheckpointKey(prefix, position)


@pytest.mark.parametrize("layout", [{1: 1, 3: 4}, {1: 4, 3: 16}])
def test_history_survives_ring_wrap_and_source_completion(layout):
    allocator = BlockPool(200)
    store = checkpoint.CompressorCheckpointPool(allocator, layout, sum(layout.values()) * 2)
    source = rings(allocator, layout)
    for blocks in source.values():
        for block in blocks:
            allocator.payload[block.block_id] = [block.block_id * 10 + slot for slot in range(8)]
    plan = store.reserve(key(), source)
    assert store.lookup(key()) is None
    allocator.copy(plan)
    store.complete_save(plan.handle)
    saved = {op.destination: allocator.payload[op.destination].copy() for op in plan.copies}
    for _ in range(3):
        for blocks in source.values():
            for block in blocks:
                allocator.payload[block.block_id] = [999] * 8
    for blocks in source.values():
        allocator.free_blocks(blocks)

    # Both borrowers get their own ring; neither owns the checkpoint pages.
    borrowers = [rings(allocator, layout), rings(allocator, layout)]
    for destination in borrowers:
        store.acquire(plan.handle)
        restore = store.restore_plan(plan.handle, destination)
        assert restore.position == key().position
        allocator.copy(restore)
        for op in restore.copies:
            assert allocator.payload[op.destination] == saved[op.source]
        store.release(plan.handle)
    for blocks in borrowers[0].values():
        for block in blocks:
            allocator.payload[block.block_id][0] = 777
    for op in plan.copies:
        assert allocator.payload[op.destination] == saved[op.destination]
    assert all(allocator.payload[b.block_id][0] != 777 for bs in borrowers[1].values() for b in bs)
    for destination in borrowers:
        for blocks in destination.values():
            allocator.free_blocks(blocks)
    store.reset()
    assert store.used_blocks == 0
    assert allocator.get_num_free_blocks() == len(allocator.blocks)


def test_cancelled_source_cannot_be_recycled_before_copy_completion():
    allocator = BlockPool(6)
    store = checkpoint.CompressorCheckpointPool(allocator, {0: 2}, 2)
    source = rings(allocator, {0: 2})
    plan = store.reserve(key(), source)
    allocator.free_blocks(source[0])
    assert all(block.ref_cnt == 1 for block in source[0])
    replacement = allocator.get_new_blocks(2)
    assert {b.block_id for b in replacement}.isdisjoint(b.block_id for b in source[0])
    allocator.copy(plan)
    store.complete_save(plan.handle)
    assert all(block.ref_cnt == 0 for block in source[0])


def test_budget_and_eviction_preserve_readers_and_pending_saves():
    allocator = BlockPool(16)
    store = checkpoint.CompressorCheckpointPool(allocator, {0: 2}, 4)
    source = rings(allocator, {0: 2})
    first = store.reserve(key(), source)
    second = store.reserve(key(32768), source)
    assert store.reserve(key(49152), source) is None
    assert not store.can_reserve()
    store.complete_save(first.handle)
    store.acquire(first.handle)
    store.discard(first.handle)
    assert store.lookup(key()) == first.handle
    assert store.reserve(key(49152), source) is None
    store.release(first.handle)
    third = store.reserve(key(49152), source)
    assert third is not None
    assert store.lookup(key()) is None
    assert store.used_blocks == 4
    store.complete_save(second.handle)
    store.complete_save(third.handle)
    store.reset()
    assert store.used_blocks == 0
    assert all(block.ref_cnt == 1 for block in source[0])


@pytest.mark.parametrize("during_restore", [False, True])
def test_reset_requires_completed_copies(during_restore):
    allocator = BlockPool()
    store = checkpoint.CompressorCheckpointPool(allocator, {0: 1}, 1)
    source = rings(allocator, {0: 1})
    save = store.reserve(key(), source)
    if during_restore:
        store.complete_save(save.handle)
        store.acquire(save.handle)
    free_before = allocator.get_num_free_blocks()
    with pytest.raises(RuntimeError, match="completed saves and restores"):
        store.reset()
    assert allocator.get_num_free_blocks() == free_before
    assert store.has_checkpoint(key())
    if during_restore:
        store.release(save.handle)
    else:
        store.complete_save(save.handle)
    store.reset()
    assert store.used_blocks == 0
    assert store.lookup(key()) is None


def test_prefix_identity_and_duplicate_saves():
    allocator = BlockPool()
    store = checkpoint.CompressorCheckpointPool(allocator, {0: 1}, 3)
    source = rings(allocator, {0: 1})
    first = store.reserve(key(), source)
    assert store.reserve(key(), source) is None
    second = store.reserve(key(prefix=b"different-history"), source)
    third = store.reserve(key(position=32768), source)
    for plan in (first, second, third):
        store.complete_save(plan.handle)
    assert store.lookup(key()) == first.handle
    assert store.lookup(key(prefix=b"different-history")) == second.handle
    assert store.lookup(key(position=32768)) == third.handle


def test_failed_prepare_releases_reader_and_allows_reclaim():
    allocator = BlockPool(5)
    store = checkpoint.CompressorCheckpointPool(allocator, {0: 2}, 2)
    source = rings(allocator, {0: 2})
    save = store.reserve(key(), source)
    store.complete_save(save.handle)
    store.acquire(save.handle)
    store.reclaim(3)
    assert allocator.get_num_free_blocks() == 1
    # Private ring allocation cannot succeed; roll back recovery preparation.
    store.release(save.handle)
    store.reclaim(3)
    assert allocator.get_num_free_blocks() == 3
    assert store.lookup(key()) is None


def test_copy_requires_complete_private_rings_and_ready_checkpoint():
    allocator = BlockPool()
    store = checkpoint.CompressorCheckpointPool(allocator, {0: 1, 2: 2}, 3)
    source = rings(allocator, {0: 1, 2: 2})
    with pytest.raises(ValueError, match="every tail group"):
        store.reserve(key(), {0: source[0]})
    assert store.used_blocks == 0
    save = store.reserve(key(), source)
    with pytest.raises(ValueError, match="published"):
        store.acquire(save.handle)
    store.complete_save(save.handle)
    with pytest.raises(ValueError, match="Acquire"):
        store.restore_plan(save.handle, source)
    store.acquire(save.handle)
    snapshot_as_ring = {
        group: [allocator.blocks[op.destination] for op in save.copies if op.group_id == group] for group in source
    }
    with pytest.raises(ValueError, match="writable"):
        store.restore_plan(save.handle, snapshot_as_ring)
    store.release(save.handle)


def test_zero_budget_never_changes_pool_or_requests_chunk_split():
    allocator = BlockPool()
    store = checkpoint.CompressorCheckpointPool(allocator, {0: 1}, 0)
    source = rings(allocator, {0: 1})
    free_before = allocator.get_num_free_blocks()
    assert not store.can_reserve()
    assert store.reserve(key(), source) is None
    assert allocator.get_num_free_blocks() == free_before
    assert source[0][0].ref_cnt == 1


def test_candidate_respects_logit_recompute_limit_and_rechecks_sparse_boundaries():
    allocator = BlockPool()
    store = checkpoint.CompressorCheckpointPool(allocator, {0: 1}, 3)
    source = rings(allocator, {0: 1})
    hashes = [b"prefix-4", b"prefix-8", b"prefix-12", b"prefix-16"]
    first = store.reserve(key(8, hashes[1]), source)
    second = store.reserve(key(16, hashes[3]), source)
    store.complete_save(first.handle)
    store.complete_save(second.handle)
    assert store.find_candidate(hashes, 16, 8, 4) == (second.handle, 16)
    # Engine retains at least one token to produce logits.
    assert store.find_candidate(hashes, 15, 8, 4) == (first.handle, 8)
    # SWA rejects 16 and lowers the bound to 12; there is no checkpoint at 12.
    assert store.find_candidate(hashes, 12, 8, 4) == (first.handle, 8)
    # Rechecking SWA at 8 can reject that window too; do not use a stale 16 window.
    assert store.find_candidate(hashes, 7, 8, 4) is None
    assert store.find_candidate(hashes[:1], 16, 8, 4) is None
    assert store.find_candidate([b"other"] * 4, 16, 8, 4) is None
