"""Project logical KV regions into Backend objects and local allocations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    get_block_hashes,
)

from ..protocol.coordinates import TokenRange
from .region import KVRegion


@dataclass(frozen=True, slots=True)
class KVObject:
    """One content-addressed Backend object before local memory binding."""

    block_index: int
    token_range: TokenRange
    backend_key: str
    content_hash: BlockHash | str


@dataclass(frozen=True, slots=True)
class KVObjectProjection:
    """Backend objects projected from one original vLLM cache group."""

    group_id: int
    logical_block_count: int
    objects: tuple[KVObject, ...]


@dataclass(frozen=True, slots=True)
class AllocatedKVObject:
    """A projected Backend object bound to one Worker-local KV block."""

    object: KVObject
    block_id: int


@dataclass(frozen=True, slots=True)
class KVMemorySlice:
    """Local address segments for one physical partition of a KV object."""

    addresses: tuple[int, ...]
    sizes: tuple[int, ...]


class KVObjectProjector:
    """Turn semantic regions and content hashes into Backend object identities."""

    def __init__(self, token_database: ChunkedTokenDatabase) -> None:
        self._token_database = token_database

    def project(
        self,
        region: KVRegion,
        block_hashes: Sequence[BlockHash | str],
    ) -> tuple[KVObjectProjection, ...]:
        projections = []
        hashes = list(block_hashes)
        for selection in region.chunk_selections:
            group_id = selection.group_id
            block_size = self._token_database.get_block_size(group_id)
            aligned_start_token = region.token_range.start_token // block_size * block_size
            logical_block_count = min(
                len(get_block_hashes(hashes, block_size, self._token_database.hash_block_size)),
                cdiv(region.token_range.end_token, block_size),
            )
            objects = tuple(
                KVObject(start // block_size, TokenRange(start, end), key, content_hash)
                for start, end, key, content_hash in self._token_database.process_token_key_strings(
                    region.token_range.end_token,
                    hashes,
                    mask_num=aligned_start_token,
                    kv_cache_group_id=group_id,
                    chunk_filter=partial(selection.includes, block_size=block_size),
                )
            )
            projections.append(KVObjectProjection(group_id, logical_block_count, objects))
        return tuple(projections)


def bind_local_blocks(
    projection: KVObjectProjection,
    block_ids: Sequence[int],
    *,
    skip_null_blocks: bool = False,
) -> tuple[AllocatedKVObject, ...]:
    """Bind projected objects to the suffix of blocks allocated for this request."""

    block_id_offset = max(projection.logical_block_count - len(block_ids), 0)
    allocated_objects = []
    for kv_object in projection.objects:
        local_block_index = kv_object.block_index - block_id_offset
        if 0 <= local_block_index < len(block_ids):
            block_id = block_ids[local_block_index]
            if not skip_null_blocks or block_id > 0:
                allocated_objects.append(AllocatedKVObject(kv_object, block_id))
    return tuple(allocated_objects)


class EffectiveTPKeyProjector:
    """Project one base Backend key into the effective TP rank namespace."""

    def __init__(self, tp_rank: int, key_slices_per_rank: int) -> None:
        self.tp_rank = tp_rank
        self._key_slices_per_rank = key_slices_per_rank

    def project(self, base_key: str) -> tuple[str, ...]:
        marker = "@head_or_tp_rank:"
        marker_start = base_key.find(marker)
        if marker_start < 0:
            return (base_key,) * self._key_slices_per_rank

        value_start = marker_start + len(marker)
        value_end = base_key.find("@", value_start)
        if value_end < 0:
            value_end = len(base_key)
        keys = []
        for slice_index in range(self._key_slices_per_rank):
            effective_rank = self.tp_rank * self._key_slices_per_rank + slice_index
            keys.append(f"{base_key[:value_start]}{effective_rank}{base_key[value_end:]}")
        return tuple(keys)


class StridedKVMemoryProjector:
    """Project one local KV block into its effective-TP head slices."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        group_id: int,
        block_size: int,
        slices_per_block: int,
    ) -> None:
        self._token_database = token_database
        self._group_id = group_id
        self._block_size = block_size
        self._slices_per_block = slices_per_block

    def project(self, block_id: int, token_count: int) -> tuple[KVMemorySlice, ...]:
        group_addresses = self._token_database.group_kv_caches_base_addr[self._group_id]
        group_block_lengths = self._token_database.group_block_len[self._group_id]
        group_block_strides = self._token_database.group_block_stride.get(self._group_id)
        slice_size = group_block_lengths[0] // self._block_size // self._slices_per_block
        slices = []
        for slice_index in range(self._slices_per_block):
            addresses = []
            sizes = []
            head_offset = slice_index * slice_size
            for index, base_address in enumerate(group_addresses):
                block_length = group_block_lengths[index]
                block_stride = group_block_strides[index] if group_block_strides else block_length
                bytes_per_token = block_length // self._block_size
                block_address = base_address + block_id * block_stride
                for token_index in range(token_count):
                    addresses.append(block_address + token_index * bytes_per_token + head_offset)
                    sizes.append(slice_size)
            slices.append(KVMemorySlice(tuple(addresses), tuple(sizes)))
        return tuple(slices)
