"""Carry transfer evidence and group common completion outcomes by request."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...infrastructure.protocol.messages import StoreCommand
from .batch import KVTransferBatch, TransferSource


@dataclass(frozen=True, slots=True)
class TransferEvidence:
    source: TransferSource
    result_code: int | None
    source_release_confirmed: bool | None = None


@dataclass(frozen=True, slots=True)
class StoreEvidence:
    transfer_evidence: tuple[TransferEvidence, ...]
    succeeded: bool
    source_release_confirmed: bool
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class LoadCompletion:
    """A completed Load for one request; only failed or unknown transfers carry provenance."""

    request_id: str
    failures: tuple[TransferEvidence, ...]


@dataclass(frozen=True, slots=True)
class StoreCompletion:
    request_id: str
    evidence: StoreEvidence
    store_job_id: int | None = None


def _batch_sources(batch: KVTransferBatch, layer_id: int | None) -> tuple:
    return tuple(
        source
        for group in batch.groups
        if layer_id is None or layer_id in group.physical_layer_ids
        for source in (
            group.source(index, layer_id)
            for index in (
                range(group.object_count)
                if group.selected_objects is None
                else np.flatnonzero(group.selected_objects).tolist()
            )
        )
    )


def _load_completions(
    batch: KVTransferBatch,
    failures: tuple[TransferEvidence, ...],
) -> tuple[LoadCompletion, ...]:
    if not failures:
        return tuple(LoadCompletion(request_id, ()) for request_id in batch.request_ids)
    by_request: list[list[TransferEvidence]] = [[] for _ in batch.request_ids]
    for item in failures:
        by_request[item.source.request_index].append(item)
    return tuple(
        LoadCompletion(request_id, tuple(items))
        for request_id, items in zip(batch.request_ids, by_request, strict=True)
    )


def _failed_load_completions(
    batch: KVTransferBatch,
    layer_id: int | None,
) -> tuple[LoadCompletion, ...]:
    evidence = tuple(TransferEvidence(source, None) for source in _batch_sources(batch, layer_id))
    return _load_completions(batch, evidence)


def _store_command_completions(
    commands: tuple[StoreCommand, ...], error: Exception | None = None
) -> tuple[StoreCompletion, ...]:
    evidence = StoreEvidence((), error is None, True, error)
    return tuple(StoreCompletion(command.request_id, evidence, command.store_job_id) for command in commands)
