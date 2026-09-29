"""Project KV selections into semantic chunks and canonical remote identities."""

from __future__ import annotations

from functools import partial

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    get_block_hashes,
)

from ...coordinates import TokenRange
from ..representation import KVChunk, KVChunkBatch, RemoteObjectKey, RemoteObjectKeyBatch
from ..spec.topology import KVPoolGroupTopology, KVPoolTopology
from .reachability import GroupSelection, KVSelection


class KVChunkProjection:
    """Materialize selected token ranges as canonical keyed KV chunks."""

    def __init__(self, token_database: ChunkedTokenDatabase, topology: KVPoolTopology) -> None:
        self._token_database = token_database
        groups_by_id = {group.group_id: group for group in topology.groups}
        try:
            groups = tuple(groups_by_id[group_id] for group_id in topology.transfer_group_ids)
        except KeyError as error:
            raise ValueError(f"Unknown transferable KV cache group {error.args[0]}") from error
        self.group_ids = tuple(group.group_id for group in groups)
        self._groups = {group.group_id: group for group in groups}

    def project(
        self,
        selection: KVSelection,
    ) -> tuple[tuple[KVChunkBatch, ...], tuple[RemoteObjectKeyBatch, ...]]:
        selection_group_ids = tuple(group.group_id for group in selection.groups)
        if selection_group_ids != self.group_ids:
            raise ValueError(f"KV selection groups {selection_group_ids} do not match compiled groups {self.group_ids}")
        projected_groups = tuple(self._project_group(selection, group) for group in selection.groups)
        return (
            tuple(chunks for chunks, _ in projected_groups),
            tuple(object_keys for _, object_keys in projected_groups),
        )

    def _project_group(
        self,
        selection: KVSelection,
        group_selection: GroupSelection,
    ) -> tuple[KVChunkBatch, RemoteObjectKeyBatch]:
        group = self._groups[group_selection.group_id]
        hashes = list(selection.block_hashes)
        aligned_start = selection.token_range.start_token // group.block_size * group.block_size
        logical_block_count = min(
            len(get_block_hashes(hashes, group.block_size, self._token_database.hash_block_size)),
            cdiv(selection.token_range.end_token, group.block_size),
        )
        key_records = tuple(
            self._token_database.process_token_key_strings(
                selection.token_range.end_token,
                hashes,
                mask_num=aligned_start,
                kv_cache_group_id=group.group_id,
                chunk_filter=partial(group_selection.includes, block_size=group.block_size),
            )
        )
        projected = tuple(self._project_record(group, record) for record in key_records)
        chunks = tuple(chunk for chunk, _ in projected)
        object_keys = tuple(object_key for _, object_key in projected)
        return (
            KVChunkBatch(group.group_id, logical_block_count, chunks),
            RemoteObjectKeyBatch(group.group_id, object_keys),
        )

    @staticmethod
    def _project_record(
        group: KVPoolGroupTopology,
        record: tuple[int, int, str, BlockHash | str],
    ) -> tuple[KVChunk, RemoteObjectKey]:
        start_token, end_token, base_key, content_hash = record
        chunk = KVChunk(
            group.group_id,
            start_token // group.block_size,
            TokenRange(start_token, end_token),
            content_hash,
        )
        return chunk, RemoteObjectKey(chunk, base_key)
