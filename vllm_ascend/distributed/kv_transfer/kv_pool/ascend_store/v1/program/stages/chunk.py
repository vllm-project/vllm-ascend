"""Project selected token regions into semantic KV chunks."""

from __future__ import annotations

from functools import partial

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    get_block_hashes,
)

from ...coordinates import TokenRange
from ...protocol.lookup import TailKeyBoundary
from ...protocol.transfer import CheckpointStoreCommand
from ..representation import KVChunk, KVChunkBatch
from ..spec.topology import KVPoolGroupTopology
from .reachability import GroupSelection, KVSelection


class SemanticChunkProjection:
    """Materialize selected token regions as content-identified KV chunks."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        groups: tuple[KVPoolGroupTopology, ...],
        fine_grained_lookup: bool,
    ) -> None:
        self._token_database = token_database
        self.group_ids = tuple(group.group_id for group in groups)
        self._groups = {group.group_id: group for group in groups}
        self._fine_grained_lookup = fine_grained_lookup

    def project(self, selection: KVSelection) -> tuple[KVChunkBatch, ...]:
        self._validate_groups(selection)
        return tuple(self._project_group(selection, group) for group in selection.groups)

    def project_lookup(self, selection: KVSelection) -> tuple[KVChunkBatch, ...]:
        """Project every remote identity that may prove the next reachable prefix."""

        self._validate_groups(selection)
        if not self._fine_grained_lookup:
            return tuple(self._project_group(selection, group) for group in selection.groups)
        return tuple(self._project_fine_group(selection, group) for group in selection.groups)

    def project_load(
        self,
        selection: KVSelection,
        tail_key_boundaries: tuple[TailKeyBoundary, ...],
    ) -> tuple[KVChunkBatch, ...]:
        """Project reusable chunks while preserving each tail object's observed identity."""

        self._validate_groups(selection)
        boundaries_by_group = {boundary.group_id: boundary.boundary_token for boundary in tail_key_boundaries}
        if len(boundaries_by_group) != len(tail_key_boundaries):
            raise ValueError("Load contains duplicate tail-key boundaries for one cache group")
        unknown_groups = set(boundaries_by_group).difference(self.group_ids)
        if unknown_groups:
            raise ValueError(f"Load tail-key boundaries contain unknown cache groups {sorted(unknown_groups)}")
        return tuple(
            self._project_load_group(selection, group, boundaries_by_group.get(group.group_id))
            for group in selection.groups
        )

    def _validate_groups(self, selection: KVSelection) -> None:
        selection_group_ids = tuple(group.group_id for group in selection.groups)
        if selection_group_ids != self.group_ids:
            raise ValueError(f"KV selection groups {selection_group_ids} do not match compiled groups {self.group_ids}")

    def _project_group(self, selection: KVSelection, group_selection: GroupSelection) -> KVChunkBatch:
        group = self._groups[group_selection.group_id]
        hashes = list(selection.block_hashes)
        aligned_start = selection.token_range.start_token // group.block_size * group.block_size
        logical_block_count = min(
            len(get_block_hashes(hashes, group.block_size, self._token_database.hash_block_size)),
            cdiv(selection.token_range.end_token, group.block_size),
        )
        records = self._token_database.process_token_key_strings(
            selection.token_range.end_token,
            hashes,
            mask_num=aligned_start,
            kv_cache_group_id=group.group_id,
            chunk_filter=partial(group_selection.includes, block_size=group.block_size),
        )
        chunks = tuple(self._project_record(group, record) for record in records)
        return KVChunkBatch(group.group_id, logical_block_count, chunks)

    def _project_fine_group(self, selection: KVSelection, group_selection: GroupSelection) -> KVChunkBatch:
        group = self._groups[group_selection.group_id]
        hash_block_size = self._token_database.hash_block_size
        first_hash_index = selection.token_range.start_token // hash_block_size
        last_hash_index = min(len(selection.block_hashes), selection.token_range.end_token // hash_block_size)
        chunks = []
        for hash_index in range(first_hash_index, last_hash_index):
            boundary_token = (hash_index + 1) * hash_block_size
            block_index = cdiv(boundary_token, group.block_size) - 1
            block_start = block_index * group.block_size
            if group_selection.includes(block_start, group.block_size):
                chunks.append(
                    KVChunk(
                        group.group_id,
                        block_index,
                        TokenRange(block_start, boundary_token),
                        selection.block_hashes[hash_index],
                    )
                )
        logical_block_count = min(
            cdiv(selection.token_range.end_token, group.block_size),
            cdiv(last_hash_index * hash_block_size, group.block_size),
        )
        return KVChunkBatch(group.group_id, logical_block_count, tuple(chunks))

    def _project_load_group(
        self,
        selection: KVSelection,
        group_selection: GroupSelection,
        boundary_token: int | None,
    ) -> KVChunkBatch:
        batch = self._project_group(selection, group_selection)
        if boundary_token is None:
            return batch
        group = self._groups[group_selection.group_id]
        load_end = selection.token_range.end_token
        if load_end <= 0 or cdiv(boundary_token, group.block_size) != cdiv(load_end, group.block_size):
            raise ValueError(
                f"Tail-key boundary {boundary_token} does not identify cache group {group.group_id}'s Load tail"
            )
        block_index = cdiv(load_end, group.block_size) - 1
        tail = KVChunk(
            group.group_id,
            block_index,
            TokenRange(block_index * group.block_size, load_end),
            self._boundary_hash(boundary_token, selection.block_hashes),
        )
        chunks = tuple(chunk for chunk in batch.chunks if chunk.block_index != block_index) + (tail,)
        return KVChunkBatch(group.group_id, max(batch.logical_block_count, block_index + 1), chunks)

    def _boundary_hash(
        self,
        boundary_token: int,
        block_hashes: tuple[BlockHash | str, ...],
    ) -> BlockHash | str:
        hash_block_size = self._token_database.hash_block_size
        if boundary_token <= 0 or boundary_token % hash_block_size:
            raise ValueError(f"Tail-key boundary {boundary_token} is not aligned to the hash block size")
        hash_index = boundary_token // hash_block_size - 1
        if hash_index >= len(block_hashes):
            raise ValueError(f"Tail-key boundary {boundary_token} has no corresponding block hash")
        return block_hashes[hash_index]

    @staticmethod
    def _project_record(group: KVPoolGroupTopology, record: tuple[int, int, str, BlockHash | str]) -> KVChunk:
        start_token, end_token, _, content_hash = record
        return KVChunk(
            group.group_id,
            start_token // group.block_size,
            TokenRange(start_token, end_token),
            content_hash,
        )


class CheckpointChunkProjection:
    """Project explicit Mamba state checkpoints and their companion KV chunks."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        groups: tuple[KVPoolGroupTopology, ...],
        checkpoint_group_ids: frozenset[int],
    ) -> None:
        self._token_database = token_database
        self._groups = {group.group_id: group for group in groups}
        self._checkpoint_group_ids = checkpoint_group_ids
        self._companion_groups = tuple(group for group in groups if group.group_id not in checkpoint_group_ids)

    def project(self, command: CheckpointStoreCommand) -> tuple[KVChunkBatch, ...]:
        boundary_tokens = {source.boundary_token for source in command.sources}
        if len(boundary_tokens) != 1:
            raise ValueError("State checkpoints for one request must share one token boundary")
        boundary_token = next(iter(boundary_tokens))
        batches = tuple(
            self._project_checkpoint(source.group_id, source.boundary_token, command.block_hashes)
            for source in command.sources
        )
        if not any(boundary_token % self._groups[source.group_id].block_size for source in command.sources):
            return batches
        companions = tuple(
            self._project_companion_group(
                group,
                boundary_token,
                command.published_store_end_token,
                command.block_hashes,
            )
            for group in self._companion_groups
        )
        return batches + companions

    def _project_checkpoint(
        self,
        group_id: int,
        boundary_token: int,
        block_hashes: tuple[BlockHash, ...],
    ) -> KVChunkBatch:
        group = self._groups.get(group_id)
        if group is None:
            raise ValueError(f"State checkpoint belongs to non-transferable cache group {group_id}")
        if group_id not in self._checkpoint_group_ids:
            raise ValueError(f"Cache group {group_id} does not use Mamba align state")
        block_index = cdiv(boundary_token, group.block_size) - 1
        chunk = KVChunk(
            group_id,
            block_index,
            TokenRange(block_index * group.block_size, boundary_token),
            self._boundary_hash(boundary_token, block_hashes),
        )
        return KVChunkBatch(group_id, block_index + 1, (chunk,))

    def _project_companion_group(
        self,
        group: KVPoolGroupTopology,
        boundary_token: int,
        published_store_end_token: int,
        block_hashes: tuple[BlockHash, ...],
    ) -> KVChunkBatch:
        last_block_index = cdiv(boundary_token, group.block_size) - 1
        first_block_index = min(published_store_end_token // group.block_size, last_block_index)
        chunks = tuple(
            KVChunk(
                group.group_id,
                block_index,
                TokenRange(block_index * group.block_size, min((block_index + 1) * group.block_size, boundary_token)),
                self._boundary_hash(min((block_index + 1) * group.block_size, boundary_token), block_hashes),
            )
            for block_index in range(first_block_index, last_block_index + 1)
        )
        return KVChunkBatch(group.group_id, last_block_index + 1, chunks)

    def _boundary_hash(self, boundary_token: int, block_hashes: tuple[BlockHash, ...]) -> BlockHash:
        if boundary_token <= 0 or boundary_token % self._token_database.hash_block_size:
            raise ValueError(f"Boundary token {boundary_token} is not aligned to the hash block size")
        hash_index = boundary_token // self._token_database.hash_block_size - 1
        if hash_index >= len(block_hashes):
            raise ValueError(f"Boundary token {boundary_token} has no corresponding block hash")
        return block_hashes[hash_index]
