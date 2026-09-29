"""Project semantic chunks into canonical and physical Backend objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import PoolKey, block_hash_to_str

from ..representation import (
    KVChunk,
    KVChunkBatch,
    PhysicalCoordinate,
    RemoteKVObject,
    RemoteObjectBatch,
    RemoteObjectKey,
    RemoteObjectKeyBatch,
    TransferLayoutBatch,
)
from ..spec.topology import KVPoolGroupTopology, KVPoolTopology


@dataclass(frozen=True, slots=True)
class _RemoteRepresentation:
    coordinate: PhysicalCoordinate
    rank_values: tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class _RemoteKeyTemplate:
    fragments: tuple[str, str, str, str]
    rank_order: tuple[int, int, int]

    @classmethod
    def compile(cls, base_key: str) -> _RemoteKeyTemplate:
        spans = sorted(
            (
                (*_required_rank_span(base_key, "dcp"), 0),
                (*_required_rank_span(base_key, "head_or_tp_rank"), 1),
                (*_required_rank_span(base_key, "pp_rank"), 2),
            )
        )
        first, second, third = spans
        fragments = (
            base_key[: first[0]],
            base_key[first[1] : second[0]],
            base_key[second[1] : third[0]],
            base_key[third[1] :],
        )
        return cls(fragments, (first[2], second[2], third[2]))

    def render(self, rank_values: tuple[str, str, str]) -> str:
        first, second, third = self.rank_order
        return "".join(
            (
                self.fragments[0],
                rank_values[first],
                self.fragments[1],
                rank_values[second],
                self.fragments[2],
                rank_values[third],
                self.fragments[3],
            )
        )


def project_remote_identities(
    batches: tuple[KVChunkBatch, ...],
    groups_by_id: Mapping[int, KVPoolGroupTopology],
    transfer_group_ids: tuple[int, ...],
) -> tuple[RemoteObjectKeyBatch, ...]:
    """Project semantic chunks into canonical Backend identities."""

    group_ids = tuple(dict.fromkeys(batch.group_id for batch in batches))
    if any(group_id not in transfer_group_ids for group_id in group_ids):
        raise ValueError(f"KV chunk groups {group_ids} are not contained in compiled groups {transfer_group_ids}")
    projected = []
    for batch in batches:
        metadata = groups_by_id[batch.group_id].key_metadata
        keys = tuple(
            RemoteObjectKey(chunk, PoolKey(metadata, block_hash_to_str(chunk.content_hash)).to_string())
            for chunk in batch.chunks
        )
        projected.append(RemoteObjectKeyBatch(batch.group_id, keys))
    return tuple(projected)


class RemoteObjectProjection:
    """Materialize canonical identities as Backend objects at compiled physical coordinates."""

    def __init__(self, topology: KVPoolTopology, lookup_rank_counts: Mapping[int, int]) -> None:
        groups_by_id = {group.group_id: group for group in topology.groups}
        groups = tuple(groups_by_id[group_id] for group_id in topology.transfer_group_ids)
        self.group_ids = tuple(group.group_id for group in groups)
        self._base_rank_values = {
            group.group_id: (
                str(group.key_metadata.dcp_rank),
                str(group.key_metadata.head_or_tp_rank),
                str(group.key_metadata.pp_rank),
            )
            for group in groups
        }
        self._representations_by_group = {
            group.group_id: tuple(
                _RemoteRepresentation(
                    PhysicalCoordinate(pp_rank=pp_rank, dcp_rank=dcp_rank, head_rank=head_rank),
                    (str(dcp_rank), str(head_rank), str(pp_rank)),
                )
                for pp_rank in range(topology.pp_size)
                for dcp_rank in range(topology.dcp_size)
                for head_rank in range(lookup_rank_counts[group.group_id])
            )
            for group in groups
        }

    def project_lookup(self, batches: tuple[RemoteObjectKeyBatch, ...]) -> tuple[RemoteObjectBatch, ...]:
        group_ids = tuple(batch.group_id for batch in batches)
        if group_ids != self.group_ids:
            raise ValueError(f"Remote object key groups {group_ids} do not match compiled groups {self.group_ids}")
        return tuple(self._project_lookup_batch(batch) for batch in batches)

    def project_transfer(
        self,
        layout_batch: TransferLayoutBatch,
        object_keys: RemoteObjectKeyBatch,
    ) -> RemoteObjectBatch:
        if layout_batch.group_id != object_keys.group_id:
            raise ValueError(
                f"Transfer layout group {layout_batch.group_id} does not match remote object key group "
                f"{object_keys.group_id}"
            )
        templates = {item.chunk: _RemoteKeyTemplate.compile(item.base_key) for item in object_keys.keys}
        projected: dict[tuple[KVChunk, PhysicalCoordinate], RemoteKVObject] = {}
        remote_objects = []
        for layout in layout_batch.layouts:
            chunk = layout.local_region.region.chunk
            coordinate = layout.remote_layout.coordinate
            try:
                template = templates[chunk]
            except KeyError as error:
                raise ValueError(f"KV transfer layout has no remote object key: {chunk}") from error
            object_identity = chunk, coordinate
            remote_object = projected.get(object_identity)
            if remote_object is None:
                remote_object = RemoteKVObject(
                    chunk,
                    template.render(self._rank_values(layout_batch.group_id, coordinate)),
                    coordinate,
                )
                projected[object_identity] = remote_object
            remote_objects.append(remote_object)
        return RemoteObjectBatch(layout_batch.group_id, tuple(remote_objects))

    def _project_lookup_batch(self, batch: RemoteObjectKeyBatch) -> RemoteObjectBatch:
        representations = self._representations_by_group[batch.group_id]
        if len(representations) == 1:
            representation = representations[0]
            if representation.rank_values == self._base_rank_values[batch.group_id]:
                objects = tuple(
                    RemoteKVObject(item.chunk, item.base_key, representation.coordinate) for item in batch.keys
                )
                return RemoteObjectBatch(batch.group_id, objects)

        templates = tuple(_RemoteKeyTemplate.compile(item.base_key) for item in batch.keys)
        objects = tuple(
            RemoteKVObject(item.chunk, template.render(representation.rank_values), representation.coordinate)
            for representation in representations
            for item, template in zip(batch.keys, templates, strict=True)
        )
        return RemoteObjectBatch(batch.group_id, objects)

    def _rank_values(self, group_id: int, coordinate: PhysicalCoordinate) -> tuple[str, str, str]:
        dcp_rank, head_rank, pp_rank = self._base_rank_values[group_id]
        if coordinate.dcp_rank is not None:
            dcp_rank = str(coordinate.dcp_rank)
        if coordinate.head_rank is not None:
            head_rank = str(coordinate.head_rank)
        if coordinate.effective_tp_rank is not None:
            head_rank = str(coordinate.effective_tp_rank)
        if coordinate.pp_rank is not None:
            pp_rank = str(coordinate.pp_rank)
        if coordinate.consumer_pp_slice is not None:
            pp_rank = str(coordinate.consumer_pp_slice)
        return dcp_rank, head_rank, pp_rank


def _required_rank_span(key: str, field: str) -> tuple[int, int]:
    marker = f"@{field}:"
    value_start = key.index(marker) + len(marker)
    value_end = key.index("@", value_start)
    return value_start, value_end
