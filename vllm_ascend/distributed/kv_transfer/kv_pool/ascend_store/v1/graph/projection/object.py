"""Project canonical remote keys into Backend-visible object representations."""

from __future__ import annotations

from dataclasses import dataclass

from ..elements import PhysicalCoordinate, RemoteKVObject, RemoteObjectBatch, RemoteObjectKeyBatch
from ..topology import KVPoolTopology


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


class RemoteObjectProjection:
    """Expand canonical Backend keys into every representation required by Lookup."""

    def __init__(self, topology: KVPoolTopology) -> None:
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
        self._representations = tuple(
            _RemoteRepresentation(
                PhysicalCoordinate(pp_rank=pp_rank, dcp_rank=dcp_rank, head_rank=head_rank),
                (str(dcp_rank), str(head_rank), str(pp_rank)),
            )
            for pp_rank in range(topology.pp_size)
            for dcp_rank in range(topology.dcp_size)
            for head_rank in range(topology.tp_partition.key_rank_count)
        )

    def project(self, batches: tuple[RemoteObjectKeyBatch, ...]) -> tuple[RemoteObjectBatch, ...]:
        group_ids = tuple(batch.group_id for batch in batches)
        if group_ids != self.group_ids:
            raise ValueError(f"Remote object key groups {group_ids} do not match compiled groups {self.group_ids}")
        return tuple(self._project_batch(batch) for batch in batches)

    def _project_batch(self, batch: RemoteObjectKeyBatch) -> RemoteObjectBatch:
        if len(self._representations) == 1:
            representation = self._representations[0]
            if representation.rank_values == self._base_rank_values[batch.group_id]:
                objects = tuple(
                    RemoteKVObject(item.chunk, item.base_key, representation.coordinate) for item in batch.keys
                )
                return RemoteObjectBatch(batch.group_id, objects)

        templates = tuple(_RemoteKeyTemplate.compile(item.base_key) for item in batch.keys)
        objects = tuple(
            RemoteKVObject(item.chunk, template.render(representation.rank_values), representation.coordinate)
            for representation in self._representations
            for item, template in zip(batch.keys, templates, strict=True)
        )
        return RemoteObjectBatch(batch.group_id, objects)


def replace_key_rank(key: str, field: str, rank: int) -> str:
    """Replace one rank component in an AscendStore Backend key."""

    marker = f"@{field}:"
    marker_start = key.find(marker)
    if marker_start < 0:
        return key
    value_start = marker_start + len(marker)
    value_end = key.find("@", value_start)
    if value_end < 0:
        value_end = len(key)
    return f"{key[:value_start]}{rank}{key[value_end:]}"


def _required_rank_span(key: str, field: str) -> tuple[int, int]:
    marker = f"@{field}:"
    value_start = key.index(marker) + len(marker)
    value_end = key.index("@", value_start)
    return value_start, value_end
