"""Prepare per-layer Backend transfers and cleanup; delegate timing to LayerwiseExecutor."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from functools import partial
from typing import Any, cast

import numpy as np

from ...backend import GVABackend, GVAObjectSessions, KeyRangeBackend, LayerwiseAccessKind, ObjectSessions
from ...infrastructure.executor import LayerwiseExecutor
from ...infrastructure.mapping import PartitionId, local_addresses
from ...protocol.results import LoadResult, StoreResult
from ..data.batch import LayerTransferPlan, LoadBatch, StoreBatch
from ..data.context import TransferContext
from ..data.selection import CachePartition, LoadSelection, StoreSelection

MAX_GVA_ADDRESS = (1 << 64) - 1


class LayerwiseTransfer:
    """Both services share this instance; execution timing belongs to LayerwiseExecutor.

    Writes fill caller-owned memory before consumption. Reads consume it after
    the caller's source-ready boundary. IO completion callbacks run on their
    direction's executor after all submitted layer tasks have become terminal.
    """

    def __init__(
        self,
        context: TransferContext,
        backend: KeyRangeBackend | GVABackend,
        access: LayerwiseAccessKind,
        backend_sessions: ObjectSessions,
        executor: LayerwiseExecutor,
    ) -> None:
        self.context, self.backend, self.backend_sessions = context, backend, backend_sessions
        self.executor = executor
        self.shared_objects = access is LayerwiseAccessKind.GVA
        self._write_batch: LoadBatch | None = None
        self._read_batch: StoreBatch | None = None
        self._previous_read: StoreBatch | None = None
        self.copy: Callable[..., int | tuple[int, ...]]
        if access is LayerwiseAccessKind.GVA:
            gva, regions = cast(GVABackend, backend), cast(GVAObjectSessions, backend_sessions)
            self._load_gva, self._store_gva = gva.copy_from_gva, gva.copy_to_gva
            self._load_regions, self._store_regions = regions.load_regions, regions.store_regions
            self.copy = self._copy_gva
        else:
            ranges = cast(KeyRangeBackend, backend)
            self._load_ranges, self._store_ranges = ranges.copy_key_range_load, ranges.copy_key_range_store
            self.copy = self._copy_ranges
        self.access = access
        self._geometry: dict[int, dict[tuple[int, PartitionId], tuple[slice, Any, Any]]] = {}
        for group_id, partitions in context.local.items():
            for partition in partitions:
                for layer_id, span in context.layers[group_id][partition[0]].items():
                    if span.start != span.stop:
                        self._geometry.setdefault(layer_id, {})[group_id, partition] = (
                            span,
                            np.asarray(context.sizes[group_id][partition][span], dtype=np.uint64)
                            if self.shared_objects
                            else list(context.sizes[group_id][partition][span]),
                            context.remote[group_id][partition[0]]["offsets"][span]
                            if self.shared_objects
                            else context.remote[group_id][partition[0]]["offsets"][span].tolist(),
                        )

        self._memberships: dict[tuple[tuple[int, PartitionId], ...], list[int]] = {}
        for layer_id, geometry in self._geometry.items():
            self._memberships.setdefault(tuple(geometry), []).append(layer_id)

    @property
    def read_enabled(self) -> bool:
        return self.executor.read_enabled

    def begin_step(self) -> Future[tuple[StoreResult, ...]] | None:
        self.executor.begin_step()
        return self._previous_read.completion if self._previous_read is not None else None

    def submit_write(
        self,
        batch: LoadBatch,
        finish: Callable[[LoadBatch], LoadResult],
        check_ready: Callable[[LoadBatch], None],
    ) -> None:
        if self._write_batch is not None:
            raise RuntimeError("Layerwise write already started for this step")
        batch.completion = Future()
        self._write_batch = batch
        if not batch.selection.keys:
            batch.completion.set_result(finish(batch))
            return
        self.executor.start_write(
            partial(self._open_write, batch, check_ready),
            partial(self._write_layer, batch=batch, check_ready=check_ready),
            partial(self._finish_write, batch, finish),
        )

    def _open_write(self, batch: LoadBatch, check_ready: Callable[[LoadBatch], None]) -> tuple[int, ...]:
        batch.plan, codes, _ = self.open(batch.selection, to_remote=False)
        batch.codes = codes
        if any(batch.codes):
            check_ready(batch)
        return tuple(sorted(batch.plan.layers))

    def wait_for_layer(self, layer_id: int, record_source_ready: Callable[[], Any]) -> None:
        self.executor.wait_for_layer(
            layer_id, record_source_ready, read_at_compute_start=self.context.read_at_compute_start
        )

    def _write_layer(self, layer_id: int, batch: LoadBatch, check_ready: Callable[[LoadBatch], None]) -> None:
        assert batch.plan is not None
        if batch.error is not None:
            raise batch.error
        try:
            codes = self.copy(
                batch.plan,
                layer_id,
                to_remote=False,
                on_start=lambda: setattr(batch, "destination_writes_stopped", False),
            )
            batch.destination_writes_stopped = True
            failed = False
            for index, code in batch.plan.failures(layer_id, codes):
                failed = True
                if batch.codes[index] == 0:
                    batch.codes[index] = code
            batch.copied_layers.add(layer_id)
            if failed:
                check_ready(batch)
        except BaseException as error:
            batch.error = error
            raise

    def _finish_write(
        self, batch: LoadBatch, finish: Callable[[LoadBatch], LoadResult], error: BaseException | None
    ) -> None:
        if batch.completion.done():
            return
        batch.error = batch.error or error
        try:
            if batch.destination_writes_stopped:
                self.backend_sessions.finish_load()
            if batch.error is not None:
                raise batch.error
            assert batch.plan is not None
            if missing := batch.plan.layers.keys() - batch.copied_layers:
                raise RuntimeError(f"Layerwise Load did not reach physical layers {sorted(missing)}")
            batch.completion.set_result(finish(batch))
        except BaseException as error:
            batch.completion.set_exception(error)
            raise

    def submit_read(
        self,
        batch: StoreBatch,
        prepare: Callable[[], StoreSelection],
        finish: Callable[[StoreBatch], tuple[StoreResult, ...]],
    ) -> None:
        if self._read_batch is not None:
            raise RuntimeError("Layerwise read already started for this step")
        batch.completion = Future()
        self.executor.start_read(
            partial(self._open_read, batch, prepare),
            partial(self._read_layer, batch=batch),
            lambda layer_id: batch.plan is not None and batch.remaining_objects > 0 and layer_id in batch.plan.layers,
            lambda: batch.plan.layers.keys() if batch.plan is not None else self._geometry.keys(),
            partial(self._finish_read, batch=batch, finish=finish),
        )
        self._read_batch = batch
        self._previous_read = None

    def _open_read(self, batch: StoreBatch, prepare: Callable[[], StoreSelection]) -> None:
        batch.selection = prepare()
        batch.plan, codes, batch.opened_indices = self.open(
            batch.selection, to_remote=True, keep_regions=batch.needs_shared_confirmation
        )
        batch.codes = cast("list[int | None]", codes)
        batch.identities = batch.plan.regions
        batch.remaining_objects = len(batch.opened_indices)

    def read_layer(self, layer_id: int, record_source_ready: Callable[[], Any]) -> None:
        self.executor.read_layer(layer_id, record_source_ready)

    def read_all_layers(self, record_source_ready: Callable[[], Any]) -> None:
        self.executor.read_all_layers(record_source_ready)

    def _read_layer(self, source_ready: Any, layer_id: int, batch: StoreBatch) -> None:
        if batch.plan is None or batch.remaining_objects == 0 or layer_id not in batch.plan.layers:
            return
        started = False

        def mark_started() -> None:
            nonlocal started
            started = True

        try:
            codes = self.copy(
                batch.plan,
                layer_id,
                to_remote=True,
                source_ready=source_ready,
                on_start=mark_started,
            )
        except Exception as error:
            batch.error = batch.error or error
            for index in batch.plan.result_indices(layer_id):
                if started:
                    batch.unsafe_objects.add(index)
                if batch.codes[index] == 0:
                    batch.remaining_objects -= 1
                batch.codes[index] = None
            raise
        try:
            for index, code in batch.plan.failures(layer_id, codes):
                if batch.codes[index] == 0:
                    batch.remaining_objects -= 1
                    batch.codes[index] = code
        except ValueError as error:
            batch.contract_error = error
            for index in batch.plan.result_indices(layer_id):
                if batch.codes[index] == 0:
                    batch.remaining_objects -= 1
                batch.codes[index] = None
            raise

    def seal_reads(self) -> None:
        try:
            self.executor.seal_reads()
        except BaseException as error:
            if self._read_batch is not None and not self._read_batch.completion.done():
                self._read_batch.completion.set_exception(error)
            raise

    def _finish_read(
        self,
        opening_error: BaseException | None,
        layer_errors: dict[int, BaseException],
        submitted_layers: frozenset[int],
        *,
        batch: StoreBatch,
        finish: Callable[[StoreBatch], tuple[StoreResult, ...]],
    ) -> None:
        try:
            if opening_error is not None:
                if not isinstance(opening_error, Exception):
                    raise opening_error
                batch.open_failed, batch.error = True, opening_error
            for error in layer_errors.values():
                if not isinstance(error, Exception):
                    raise error
                batch.error = batch.error or error
            if batch.plan is not None and batch.remaining_objects:
                for layer_id in batch.plan.layers.keys() - submitted_layers:
                    batch.error = batch.error or RuntimeError(
                        f"Layerwise Store did not reach physical layer {layer_id}"
                    )
                    for index in batch.plan.result_indices(layer_id):
                        if batch.codes[index] == 0:
                            batch.remaining_objects -= 1
                            batch.codes[index] = None
            batch.completion.set_result(finish(batch))
        except BaseException as error:
            batch.completion.set_exception(error)
            raise

    def finish_step(self, record_source_ready: Callable[[], Any]) -> Future[tuple[StoreResult, ...]] | None:
        self.seal_reads()
        # The framework's per-layer save boundary requires business result observation.
        return self._read_batch.completion if self._read_batch is not None else None

    def check_read_release(self, batch: StoreBatch) -> None:
        if self._read_batch is batch and not self.executor.reads_sealed:
            raise RuntimeError("Active Layerwise Store must be sealed before releasing its source")

    def end_step(self, *, wait: bool) -> None:
        if not self.executor.active:
            return
        try:
            self.executor.end_step(wait=wait)
        except BaseException as error:
            if self.executor.active and self._write_batch is not None and not self._write_batch.completion.done():
                self._write_batch.completion.set_exception(error)
            if (
                self._read_batch is not None
                and not self._read_batch.completion.done()
                and not self.executor.reads_sealed
            ):
                self._read_batch.completion.set_exception(error)
            raise
        finally:
            if not self.executor.active:
                if self._read_batch is not None:
                    self._previous_read = self._read_batch
                self._write_batch, self._read_batch = None, None

    def commit_read(self, batch: StoreBatch, indices: list[int]) -> None:
        assert batch.selection is not None
        keys = [batch.selection.keys[index] for index in indices]
        try:
            codes: tuple[int | None, ...] = self.backend_sessions.commit_store(keys)
        except Exception as error:
            batch.error = batch.error or error
            codes = (None,) * len(keys)
        for index, code in zip(indices, codes, strict=True):
            batch.codes[index] = code

    def revoke_read(self, batch: StoreBatch, indices: list[int] | None) -> None:
        if indices is None:
            keys = list(self.backend_sessions.pending_store_keys)
        else:
            assert batch.selection is not None
            keys = [batch.selection.keys[index] for index in indices]
        if keys:
            try:
                self.backend_sessions.revoke_store(keys)
            except Exception as error:
                batch.error = batch.error or error

    def open(
        self, selection: LoadSelection | StoreSelection, *, to_remote: bool, keep_regions: bool = False
    ) -> tuple[LayerTransferPlan, list[int], set[int]]:
        """Open identities first, then prepare only successful local execution positions."""
        plan = LayerTransferPlan()
        if not selection.keys:
            return plan, [], set()
        sizes = (
            selection.sizes
            if isinstance(selection, LoadSelection)
            else [self.context.object_sizes[partition] for partition, _, _ in selection.sources]
        )
        start = self.backend_sessions.start_store if to_remote else self.backend_sessions.start_load
        opened = start(selection.keys, sizes)
        if len(opened) != len(selection.keys):
            raise ValueError("Backend open result count does not match its keys")
        opened_indices = [index for index, code in enumerate(opened) if code == 0]
        successful = set(opened_indices)

        # Opening results add new information. Project them onto the fixed ownership once.
        execution_objects: dict[CachePartition, list[int]] = {}
        if isinstance(selection, LoadSelection):
            codes = [0] * selection.target_count
            for coordinate, local in selection.iter_local():
                positions, indices = [], []
                for position, index in enumerate(local.object_indices):
                    code = opened[index]
                    if code != 0:
                        codes[local.start + position] = code
                    else:
                        positions.append(position)
                        indices.append(index)
                if not positions:
                    continue
                execution_objects[coordinate] = indices
                plan.result_slots[coordinate] = tuple(local.start + position for position in positions)
                plan.block_ids[coordinate] = np.fromiter(
                    (local.block_ids[position] for position in positions), dtype=np.uint64
                )
        else:
            codes = list(opened)
            for index, (partition, _, _) in enumerate(selection.sources):
                if index in successful:
                    execution_objects.setdefault(partition, []).append(index)
            for coordinate, indices in execution_objects.items():
                plan.result_slots[coordinate] = tuple(indices)
                plan.block_ids[coordinate] = np.fromiter(
                    (selection.sources[index][1] for index in indices), dtype=np.uint64
                )
        if not successful:
            return plan, codes, successful

        # Resolve each successful object once in opening order, including heterogeneous sizes.
        # Publication identities are needed only when StoreService requests shared confirmation.
        if self.shared_objects:
            indices = opened_indices
            region_keys = tuple(selection.keys[index] for index in indices)
            region_sizes = tuple(sizes[index] for index in indices)
            resolve = self._store_regions if to_remote else self._load_regions
            regions = resolve(region_keys, region_sizes)
            bases = [0] * len(selection.keys)
            numeric = True
            for index, region in zip(indices, regions, strict=True):
                bases[index] = region.address
                numeric &= region.address >= 0 and region.address + region.size <= MAX_GVA_ADDRESS
            if keep_regions:
                plan.regions.update(zip(region_keys, regions, strict=True))
            for coordinate, indices in execution_objects.items():
                # Object arithmetic remains the fallback when uint64 cannot preserve the address extent.
                plan.object_bases[coordinate] = np.fromiter(
                    (bases[index] for index in indices), dtype=np.uint64 if numeric else object
                )

        layouts: dict[tuple[CachePartition, ...], int] = {}
        for coordinates, layer_ids in self._memberships.items():
            membership = tuple(coordinate for coordinate in coordinates if coordinate in execution_objects)
            if not membership:
                continue
            previous = layouts.get(membership)
            if previous is None:
                previous = layer_ids[0]
                layouts[membership] = previous
                if not self.shared_objects:
                    # These are already identified objects: this projection only assigns native layer slots.
                    layer_objects = sorted(
                        {index for coordinate in membership for index in execution_objects[coordinate]}
                    )
                    plan.range_keys[previous] = [selection.keys[index] for index in layer_objects]
                    slots = {index: slot for slot, index in enumerate(layer_objects)}
                    routes: list[int] | None = None
                    position = 0
                    for coordinate in membership:
                        for index in execution_objects[coordinate]:
                            slot = slots[index]
                            if routes is None and slot != position:
                                routes = list(range(position))
                            if routes is not None:
                                routes.append(slot)
                            position += 1
                    plan.key_merges[previous] = None if routes is None else tuple(routes)
                    plan.range_indices[previous] = tuple(
                        slot for coordinate in membership for slot in plan.result_slots[coordinate]
                    )
            for layer_id in layer_ids:
                plan.layers[layer_id] = membership
                if not self.shared_objects:
                    plan.range_indices[layer_id] = plan.range_indices[previous]
                    plan.range_keys[layer_id] = plan.range_keys[previous]
                    plan.key_merges[layer_id] = plan.key_merges[previous]
        if not self.shared_objects:
            plan.result_slots.clear()
        return plan, codes, successful

    def _copy_ranges(
        self,
        plan: LayerTransferPlan,
        layer_id: int,
        *,
        to_remote: bool,
        source_ready: Any = None,
        on_start: Callable[[], None] | None = None,
    ) -> tuple[int, ...]:
        keys = plan.range_keys[layer_id]
        merges = plan.key_merges[layer_id]
        addresses: list[list[int]] = [[] for _ in keys] if merges is not None else []
        sizes: list[list[int]] = [[] for _ in keys] if merges is not None else []
        offsets: list[list[int]] = [[] for _ in keys] if merges is not None else []
        position = 0
        for coordinate in plan.layers[layer_id]:
            group_id, partition = coordinate
            block_ids = plan.block_ids[coordinate]
            span, lengths, remote_offsets = self._geometry[layer_id][coordinate]
            rows = local_addresses(self.context.local[group_id][partition], block_ids, span).tolist()
            if merges is None:
                addresses.extend(rows)
                # Fixed native input rows are read-only and shared by every block in a partition.
                sizes.extend([lengths] * len(block_ids))
                offsets.extend([remote_offsets] * len(block_ids))
            else:
                for row in rows:
                    index = merges[position]
                    addresses[index].extend(row)
                    sizes[index].extend(lengths)
                    offsets[index].extend(remote_offsets)
                    position += 1
        if source_ready is not None:
            source_ready.synchronize()
        if on_start is not None:
            on_start()
        copy = self._store_ranges if to_remote else self._load_ranges
        return copy(keys, addresses, sizes, offsets)

    def _copy_gva(
        self,
        plan: LayerTransferPlan,
        layer_id: int,
        *,
        to_remote: bool,
        source_ready: Any = None,
        on_start: Callable[[], None] | None = None,
    ) -> int:
        remote: list[int] = []
        local: list[int] = []
        sizes: list[int] = []
        for coordinate in plan.layers[layer_id]:
            group_id, partition = coordinate
            block_ids = plan.block_ids[coordinate]
            span, lengths, offsets = self._geometry[layer_id][coordinate]
            addresses = local_addresses(self.context.local[group_id][partition], block_ids, span)
            remote.extend((plan.object_bases[coordinate][:, None] + offsets).ravel().tolist())
            local.extend(addresses.ravel().tolist())
            sizes.extend(np.broadcast_to(lengths, addresses.shape).ravel().tolist())
        if source_ready is not None:
            source_ready.synchronize()
        if on_start is not None:
            on_start()
        copy = self._store_gva if to_remote else self._load_gva
        return copy(remote, local, sizes)
