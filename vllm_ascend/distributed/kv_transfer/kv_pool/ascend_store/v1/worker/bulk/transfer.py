"""Bulk writer selection, Backend arguments, and positional result interpretation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...infrastructure.configuration.topology import KVPoolTopology
from ...infrastructure.mapping import BulkMapping
from ..transfer.batch import KVTransferBatch
from ..transfer.completion import LoadCompletion, StoreCompletion, StoreEvidence, TransferEvidence, _load_completions
from ..transfer.rows import StoreTransferRecords


@dataclass(frozen=True, slots=True)
class BulkBackendArguments:
    """Key-aligned local ranges prepared by the concrete Bulk route."""

    keys: list[str]
    addresses: list[list[int]]
    sizes: list[list[int]]


def materialize_bulk_arguments(mappings: dict[int, BulkMapping], batch: KVTransferBatch) -> BulkBackendArguments:
    """Select full local payloads and produce the Bulk Backend's final lists."""
    keys, addresses, lengths = [], [], []
    for group in batch.groups:
        record_count = group.transfer_record_count
        if not record_count:
            continue
        masks = (
            None
            if group.transfer_candidate_mask is None
            else group.transfer_candidate_mask.reshape(len(group.keys_by_partition), record_count)
        )
        for partition_index, partition in enumerate(mappings[group.group_id].local_layout.partitions):
            indices = None
            if masks is not None:
                selected_count = np.count_nonzero(masks[partition_index])
                if not selected_count:
                    continue
                if selected_count != record_count:
                    indices = np.flatnonzero(masks[partition_index])
            ids = group.local_block_ids if indices is None else group.local_block_ids[indices]
            partition_keys = group.keys_by_partition[partition_index]
            keys.extend(partition_keys if indices is None else (partition_keys[i] for i in indices))
            segments = partition.local_segments
            addresses.extend(segments.addresses(ids).tolist())
            sizes = np.broadcast_to(
                segments.segment_lengths_bytes, segments.block_zero_segment_addresses.shape
            ).tolist()
            lengths.extend(sizes.copy() for _ in ids)
    return BulkBackendArguments(keys, addresses, lengths)


def select_store_writer_records(
    topology: KVPoolTopology,
    rows: StoreTransferRecords,
    *,
    tp_mismatch: bool,
    align_state: bool = False,
) -> StoreTransferRecords:
    """Partition rows only among ranks proven to publish the same key and bytes."""

    if tp_mismatch or align_state:
        replica_count = 1
    elif topology.dcp_size > 1:
        # The Adapter rejects topologies where one (head, dcp) key still has
        # multiple writers or where DCP spans PCP. Every remaining local key is
        # therefore unique and must publish the complete row set.
        replica_count = 1
    else:
        replica_count = topology.put_step
    shard_rank = topology.pcp_rank * replica_count + topology.tp_rank % replica_count
    shard_count = topology.pcp_size * replica_count
    if shard_count <= 1:
        return rows
    counts, hashes, local_block_ids = rows
    selection = slice(shard_rank, len(counts), shard_count)
    return counts[selection], hashes[selection], local_block_ids[selection]


def _bulk_load_completions(batch: KVTransferBatch, result_codes: tuple[int, ...]) -> tuple[LoadCompletion, ...]:
    """Interpret positional Bulk results without allocating provenance for successful objects."""

    if all(code == 0 for code in result_codes):
        return _load_completions(batch, ())
    objects = (
        (group, index)
        for group in batch.groups
        for index in (
            range(group.transfer_candidate_count)
            if group.transfer_candidate_mask is None
            else np.flatnonzero(group.transfer_candidate_mask).tolist()
        )
    )
    failures = tuple(
        TransferEvidence(group.source(index, None), code)
        for (group, index), code in zip(objects, result_codes, strict=True)
        if code != 0
    )
    return _load_completions(batch, failures)


def _empty_store_completions(batch: KVTransferBatch) -> tuple[StoreCompletion, ...]:
    job_ids = batch.store_job_ids or (None,) * len(batch.request_ids)
    return tuple(
        StoreCompletion(request_id, StoreEvidence((), True, True), store_job_id)
        for request_id, store_job_id in zip(batch.request_ids, job_ids, strict=True)
    )


def _store_completions(
    batch: KVTransferBatch,
    evidence: tuple[TransferEvidence, ...],
    error: Exception | None = None,
    *,
    force_failed: bool = False,
) -> tuple[StoreCompletion, ...]:
    by_request: list[list[TransferEvidence]] = [[] for _ in batch.request_ids]
    for item in evidence:
        by_request[item.source.request_index].append(item)
    job_ids = batch.store_job_ids or (None,) * len(batch.request_ids)
    completions = []
    for request_id, store_job_id, items in zip(batch.request_ids, job_ids, by_request, strict=True):
        item_tuple = tuple(items)
        if not item_tuple:
            completions.append(
                StoreCompletion(
                    request_id,
                    StoreEvidence((), True, True),
                    store_job_id,
                )
            )
            continue
        succeeded = (
            not force_failed
            and error is None
            and all(
                item.result_code == 0 or (item.result_code is None and item.source_release_confirmed is True)
                for item in item_tuple
            )
        )
        source_released = all(item.source_release_confirmed is True for item in item_tuple)
        completions.append(
            StoreCompletion(
                request_id,
                StoreEvidence(item_tuple, succeeded, source_released, error),
                store_job_id,
            )
        )
    return tuple(completions)


def _failed_store_completions(
    batch: KVTransferBatch, evidence: tuple[TransferEvidence, ...], error: Exception
) -> tuple[StoreCompletion, ...]:
    by_request: list[list[TransferEvidence]] = [[] for _ in batch.request_ids]
    for item in evidence:
        by_request[item.source.request_index].append(item)
    job_ids = batch.store_job_ids or (None,) * len(batch.request_ids)
    return tuple(
        StoreCompletion(
            request_id,
            StoreEvidence(
                tuple(items), succeeded=not items, source_release_confirmed=True, error=error if items else None
            ),
            store_job_id,
        )
        for request_id, store_job_id, items in zip(batch.request_ids, job_ids, by_request, strict=True)
    )
