"""Build executable Load tasks from Worker-local cache layout."""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass
from typing import Protocol

from ...protocol.transfer import LoadRequest
from ..projection import (
    ContiguousKVBindingProjector,
    KVBinding,
    KVObjectProjection,
    StridedKVBindingProjector,
    bind_local_blocks,
)


def _circular_shift(values: list, offset: int) -> list:
    if not values or offset == 0:
        return values
    return values[offset:] + values[:offset]


@dataclass(frozen=True, slots=True)
class LoadTask:
    """A fully resolved Load operation ready for execution."""

    request_id: str
    bindings: tuple[KVBinding, ...]


class LoadTaskBuilder(Protocol):
    """Build one Backend-ready task from an approved Load request."""

    def build(self, request: LoadRequest, projections: tuple[KVObjectProjection, ...]) -> LoadTask: ...


class ContiguousLoadTaskBuilder:
    """Compile contiguous object-memory bindings into one Load task."""

    def __init__(
        self,
        binding_projector: ContiguousKVBindingProjector,
        tp_rank: int,
        align_state_group_ids: Set[int] = frozenset(),
    ) -> None:
        self._binding_projector = binding_projector
        self.tp_rank = tp_rank
        self._align_state_group_ids = align_state_group_ids

    def build(self, request: LoadRequest, projections: tuple[KVObjectProjection, ...]) -> LoadTask:
        bindings = []
        for projection in projections:
            group_id = projection.group_id
            group_block_ids = request.block_ids_by_group[group_id]
            allocated_objects = bind_local_blocks(
                projection,
                group_block_ids,
                skip_null_blocks=group_id in self._align_state_group_ids,
            )
            bindings.extend(self._binding_projector.project(group_id, group_block_ids, allocated_objects))
        bindings = _circular_shift(bindings, self.tp_rank % len(bindings)) if bindings else []
        return LoadTask(request.request_id, tuple(bindings))


class StridedLoadTaskBuilder:
    """Compile effective-TP object-memory bindings into one Load task."""

    def __init__(
        self,
        binding_projector: StridedKVBindingProjector,
        tp_rank: int,
    ) -> None:
        self._binding_projector = binding_projector
        self._tp_rank = tp_rank

    def build(self, request: LoadRequest, projections: tuple[KVObjectProjection, ...]) -> LoadTask:
        if len(projections) != 1:
            raise ValueError("Strided Load requires one cache-group projection")
        projection = projections[0]
        group_id = projection.group_id
        block_ids = request.block_ids_by_group[group_id]
        allocated_objects = bind_local_blocks(projection, block_ids)
        bindings = list(self._binding_projector.project(group_id, allocated_objects))
        bindings = _circular_shift(bindings, self._tp_rank % len(bindings)) if bindings else []
        return LoadTask(request.request_id, tuple(bindings))
