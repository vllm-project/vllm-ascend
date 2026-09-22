# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-owned, bounded storage for immutable compressor tail checkpoints.

Pages come from the existing KV block pool, so snapshots consume the same
device-memory budget as normal KV. This module owns references, not tensors.
The scheduler/worker must order forward -> save -> overwrite and must call
``complete_save`` only after every participating worker completes the copy.
Pinning source pages prevents recycling; it does not prevent ring overwrite.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock


@dataclass(frozen=True)
class TailCheckpointKey:
    prefix_hash: BlockHash
    position: int


@dataclass(frozen=True)
class TailBlockCopy:
    """Copy one physical state page in every layer of a tail cache group."""

    group_id: int
    source: int
    destination: int


@dataclass(frozen=True)
class TailSavePlan:
    handle: int
    position: int
    copies: tuple[TailBlockCopy, ...]


@dataclass(frozen=True)
class TailRestorePlan:
    handle: int
    position: int
    copies: tuple[TailBlockCopy, ...]


@dataclass
class _Checkpoint:
    key: TailCheckpointKey
    blocks: dict[int, tuple[KVCacheBlock, ...]]
    sources: tuple[KVCacheBlock, ...]
    ready: bool = False
    readers: int = 0


class CompressorCheckpointPool:
    """One pool per coordinator, using that coordinator's prefix hash identity.

    ``max_blocks`` caps allocator pages, including pending saves. Whole ring
    pages are copied in ring order: physical padding and the absolute-position
    phase are preserved without repacking. KV/SWA availability is deliberately
    checked by the coordinator at lookup, not pinned for the snapshot lifetime.
    """

    def __init__(
        self,
        block_pool: BlockPool,
        ring_blocks: Mapping[int, int],
        max_blocks: int,
    ) -> None:
        if max_blocks < 0 or not ring_blocks or any(count <= 0 for count in ring_blocks.values()):
            raise ValueError("Checkpoint capacity must be nonnegative and tail groups must have positive ring sizes")
        self.block_pool = block_pool
        self.ring_blocks = dict(ring_blocks)
        self.blocks_per_checkpoint = sum(ring_blocks.values())
        self.max_blocks = max_blocks
        self.used_blocks = 0
        self._next_handle = 0
        self._entries: dict[int, _Checkpoint] = {}
        self._index: dict[TailCheckpointKey, int] = {}
        self._block_handles: dict[int, int] = {}
        self._evictable: OrderedDict[int, None] = OrderedDict()

    def _validate_ring(self, blocks: Mapping[int, Sequence[KVCacheBlock]]) -> None:
        if blocks.keys() != self.ring_blocks.keys() or any(
            len(blocks[group_id]) != count for group_id, count in self.ring_blocks.items()
        ):
            raise ValueError("Tail copy must cover exactly the configured ring pages of every tail group")

    def can_reserve(self) -> bool:
        """Check before splitting a chunk; this does not promise allocation."""
        reclaimable = len(self._evictable) * self.blocks_per_checkpoint
        return (
            self.used_blocks + self.blocks_per_checkpoint <= self.max_blocks + reclaimable
            and self.block_pool.get_num_free_blocks() + reclaimable >= self.blocks_per_checkpoint
        )

    def reclaim(self, required_free_blocks: int) -> None:
        """Allow ordinary request admission to reclaim unused snapshots first."""
        while self._evictable and self.block_pool.get_num_free_blocks() < required_free_blocks:
            self._drop(next(iter(self._evictable)))

    def reserve(
        self,
        key: TailCheckpointKey,
        source_rings: Mapping[int, Sequence[KVCacheBlock]],
    ) -> TailSavePlan | None:
        """Reserve a save at an actual forward endpoint, before dispatch.

        Duplicate, over-budget and unavailable allocations simply skip the
        save. Source rings must already be request-owned allocations.
        """
        self._validate_ring(source_rings)
        if key.position <= 0:
            raise ValueError("Checkpoint position must be positive")
        if key in self._index or not self.can_reserve():
            return None
        while self.used_blocks + self.blocks_per_checkpoint > self.max_blocks:
            self._drop(next(iter(self._evictable)))
        self.reclaim(self.blocks_per_checkpoint)
        allocated = self.block_pool.get_new_blocks(self.blocks_per_checkpoint)
        sources = tuple(block for group_id in self.ring_blocks for block in source_rings[group_id])
        self.block_pool.touch(sources)
        snapshot_blocks = {}
        copies = []
        offset = 0
        for group_id, count in self.ring_blocks.items():
            destination = tuple(allocated[offset : offset + count])
            snapshot_blocks[group_id] = destination
            copies.extend(
                TailBlockCopy(group_id, src.block_id, dst.block_id)
                for src, dst in zip(source_rings[group_id], destination, strict=True)
            )
            offset += count
        handle = self._next_handle
        self._next_handle += 1
        self._entries[handle] = _Checkpoint(key, snapshot_blocks, sources)
        self._block_handles[allocated[0].block_id] = handle
        self._index[key] = handle
        self.used_blocks += self.blocks_per_checkpoint
        return TailSavePlan(handle, key.position, tuple(copies))

    def complete_save(self, handle: int) -> None:
        """Publish only after the save finishes on all participating workers."""
        entry = self._entries[handle]
        if entry.ready:
            raise RuntimeError("Checkpoint save was already completed")
        self.block_pool.free_blocks(entry.sources)
        entry.sources = ()
        entry.ready = True
        self._evictable[handle] = None

    def lookup(self, key: TailCheckpointKey) -> int | None:
        """A ready tail alone is not a complete KV/SWA recovery point."""
        handle = self._index.get(key)
        if handle is None or not self._entries[handle].ready:
            return None
        return handle

    def has_checkpoint(self, key: TailCheckpointKey) -> bool:
        """Whether a save is already pending or published at this boundary."""
        return key in self._index

    def find_candidate(
        self,
        block_hashes: Sequence[BlockHash],
        max_position: int,
        alignment: int,
        hash_block_size: int,
    ) -> tuple[int, int] | None:
        """Find a tail candidate; the caller must still validate ALL KV groups.

        After any KV/SWA group reduces the position, call again with the new
        upper bound and recheck all sparse groups at the resulting position.
        The hash at the boundary identifies the complete chained prefix.
        """
        if alignment <= 0 or hash_block_size <= 0 or alignment % hash_block_size:
            raise ValueError("Checkpoint alignment must be a positive multiple of hash block size")
        upper = min(max_position, len(block_hashes) * hash_block_size)
        for position in range(upper // alignment * alignment, 0, -alignment):
            key = TailCheckpointKey(block_hashes[position // hash_block_size - 1], position)
            handle = self.lookup(key)
            if handle is not None:
                return handle, position
        return None

    def acquire(self, handle: int) -> None:
        """Pin a READY snapshot during recovery preparation and execution."""
        entry = self._entries[handle]
        if not entry.ready:
            raise ValueError("Only a published checkpoint can be acquired")
        entry.readers += 1
        self._evictable.pop(handle, None)

    def snapshot_blocks(self, handle: int) -> dict[int, tuple[KVCacheBlock, ...]]:
        """Return copy sources, never request-owned writable allocations."""
        entry = self._entries[handle]
        if not entry.ready:
            raise ValueError("Only a published checkpoint can be used for recovery")
        return dict(entry.blocks)

    def handle_for_blocks(self, blocks: Mapping[int, Sequence[KVCacheBlock]]) -> int:
        """Resolve a complete tail hit passed through the KV allocation API."""
        self._validate_ring(blocks)
        first_group = next(iter(self.ring_blocks))
        handle = self._block_handles[blocks[first_group][0].block_id]
        expected = self.snapshot_blocks(handle)
        if any(
            tuple(block.block_id for block in blocks[group]) != tuple(block.block_id for block in expected[group])
            for group in self.ring_blocks
        ):
            raise ValueError("Tail hit groups must belong to the same checkpoint")
        return handle

    def restore_plan(
        self,
        handle: int,
        destination_rings: Mapping[int, Sequence[KVCacheBlock]],
    ) -> TailRestorePlan:
        self._validate_ring(destination_rings)
        entry = self._entries[handle]
        if not entry.readers:
            raise ValueError("Acquire the checkpoint before preparing a restore")
        source_ids = {block.block_id for blocks in entry.blocks.values() for block in blocks}
        if any(block.block_id in source_ids for blocks in destination_rings.values() for block in blocks):
            raise ValueError("A checkpoint cannot be used as a writable request ring")
        return TailRestorePlan(
            handle,
            entry.key.position,
            tuple(
                TailBlockCopy(group_id, source.block_id, destination.block_id)
                for group_id, blocks in entry.blocks.items()
                for source, destination in zip(blocks, destination_rings[group_id], strict=True)
            ),
        )

    def release(self, handle: int) -> None:
        """Release after restore completion, or rollback before dispatch."""
        entry = self._entries[handle]
        if not entry.readers:
            raise RuntimeError("Checkpoint has no outstanding reader")
        entry.readers -= 1
        if not entry.readers:
            self._evictable[handle] = None

    def discard(self, handle: int) -> None:
        """Abandon a failed hit unless another scheduled restore still uses it."""
        if handle in self._evictable:
            self._drop(handle)

    def reset(self) -> None:
        """Release snapshots after the synchronous scheduler drains all copies."""
        if len(self._evictable) != len(self._entries):
            raise RuntimeError("Checkpoint reset requires completed saves and restores")
        for handle in list(self._entries):
            self._drop(handle)

    def _drop(self, handle: int) -> None:
        entry = self._entries.pop(handle)
        assert not entry.readers and not entry.sources
        del self._index[entry.key]
        self._evictable.pop(handle, None)
        first_group = next(iter(self.ring_blocks))
        del self._block_handles[entry.blocks[first_group][0].block_id]
        self.block_pool.free_blocks(block for blocks in entry.blocks.values() for block in blocks)
        self.used_blocks -= self.blocks_per_checkpoint
