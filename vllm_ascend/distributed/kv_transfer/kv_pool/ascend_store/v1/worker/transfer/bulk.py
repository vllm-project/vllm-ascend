"""Prepare whole-batch Backend transfers over caller-owned memory."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from functools import partial
from itertools import groupby
from operator import itemgetter
from time import perf_counter
from typing import Any

from ...backend.interfaces import KVStoreBackend
from ...infrastructure.executor import BulkExecutor
from ...infrastructure.mapping import local_addresses
from ...protocol.outcomes import LoadResult, StoreResult
from ..data.batch import LoadBatch, StoreBatch
from ..data.context import TransferContext
from ..data.selection import LoadSelection, StoreSelection


class BulkTransfer:
    def __init__(
        self,
        context: TransferContext,
        backend: KVStoreBackend,
        executor: BulkExecutor,
        *,
        write_async: bool = False,
    ) -> None:
        self.context, self.backend = context, backend
        self.executor = executor
        self._write_async = write_async
        self.shared_objects = False
        self._read_batch: StoreBatch | None = None

    @property
    def read_enabled(self) -> bool:
        return self.executor.read_enabled

    def begin_step(self) -> Future[tuple[StoreResult, ...]] | None:
        return None

    def submit_write(
        self,
        batch: LoadBatch,
        finish: Callable[[LoadBatch], LoadResult],
        check_ready: Callable[[LoadBatch], None],
    ) -> None:
        if not batch.selection.keys:
            batch.completion = Future()
            batch.completion.set_result(finish(batch))
        else:
            batch.completion = self.executor.submit_write(partial(self._write, batch, finish, check_ready))
            if not self._write_async:
                batch.completion.result()

    def _write(
        self, batch: LoadBatch, finish: Callable[[LoadBatch], LoadResult], check_ready: Callable[[LoadBatch], None]
    ) -> LoadResult:
        keys, addresses, sizes = self.arguments(batch.selection)
        batch.destination_writes_stopped = False
        started = perf_counter()
        codes = self.backend.load(keys, addresses, sizes)
        batch.transfer_seconds = perf_counter() - started
        batch.destination_writes_stopped = True
        if len(codes) != batch.selection.target_count:
            raise ValueError("Backend Load result count does not match its parameters")
        batch.codes = list(codes)
        if any(batch.codes):
            check_ready(batch)
        return finish(batch)

    def submit_read(
        self,
        batch: StoreBatch,
        prepare: Callable[[], StoreSelection],
        finish: Callable[[StoreBatch], tuple[StoreResult, ...]],
    ) -> None:
        self.executor.check_read_ready(self._needs_source())
        self._read_batch = batch
        try:
            batch.completion = self.executor.submit_read(prepare, partial(self._read, batch, finish=finish))
        except BaseException:
            if self._read_batch is batch:
                self._read_batch = None
            raise

    def _read(
        self,
        batch: StoreBatch,
        prepare: Callable[[], StoreSelection],
        ready: Callable[[], Any],
        finish: Callable[[StoreBatch], tuple[StoreResult, ...]],
    ) -> tuple[StoreResult, ...]:
        try:
            batch.selection = prepare()
            if batch.invalid_requests:
                batch.invalidate_sources(tuple(batch.invalid_requests))
        except Exception as error:
            batch.open_failed, batch.error = True, error
        if batch.selection is not None:
            if batch.selection.keys:
                batch.codes = [None] * len(batch.selection.keys)
                started = False
                try:
                    # Build addresses before waiting for the producer's source boundary.
                    keys, addresses, sizes = self.arguments(batch.selection)
                    source_ready = ready()
                    source_ready.synchronize()
                    indices = (
                        [index for index in range(len(keys)) if index not in batch.invalid_objects]
                        if batch.invalid_objects
                        else range(len(keys))
                    )
                    if len(indices) != len(keys):
                        keys = [keys[index] for index in indices]
                        addresses = [addresses[index] for index in indices]
                        sizes = [sizes[index] for index in indices]
                    if keys:
                        started = True
                        codes = self.backend.store(keys, addresses, sizes)
                        started = False
                    else:
                        codes = ()
                    if len(codes) != len(indices):
                        batch.contract_error = ValueError("Backend Store result count does not match its parameters")
                    else:
                        for index, code in zip(indices, codes, strict=True):
                            batch.codes[index] = code
                except Exception as error:
                    if started:
                        batch.unsafe_objects.update(indices)
                    batch.error = error
        return finish(batch)

    def read_all_layers(self, record_source_ready: Callable[[], Any]) -> None:
        self.executor.record_read_ready(self._needs_source(), record_source_ready)

    def wait_for_layer(self, layer_id: int, record_source_ready: Callable[[], Any]) -> None:
        pass

    def read_layer(self, layer_id: int, record_source_ready: Callable[[], Any]) -> None:
        pass

    def finish_step(self, record_source_ready: Callable[[], Any]) -> Future[tuple[StoreResult, ...]] | None:
        self.read_all_layers(record_source_ready)
        return None

    def _needs_source(self) -> bool:
        batch = self._read_batch
        return batch is not None and not batch.open_failed and (batch.selection is None or bool(batch.selection.keys))

    def seal_reads(self) -> None:
        self.executor.check_read_ready(self._needs_source())

    def check_read_release(self, batch: StoreBatch) -> None:
        if self._read_batch is batch:
            self.seal_reads()

    def end_step(self, *, wait: bool) -> None:
        self.executor.end_step()
        self._read_batch = None

    def commit_read(self, batch: StoreBatch, indices: list[int]) -> None:
        # Whole-object publication already completed in backend.store().
        pass

    def revoke_read(self, batch: StoreBatch, indices: list[int] | None) -> None:
        pass

    def arguments(
        self, selection: LoadSelection | StoreSelection
    ) -> tuple[list[str], list[list[int]], list[list[int]]]:
        addresses: list[list[int]] = []
        sizes: list[list[int]] = []
        if isinstance(selection, StoreSelection):
            # Service preserves partition order; native parameters and compact result slots follow it directly.
            for (group_id, partition), sources in groupby(selection.sources, key=itemgetter(0)):
                block_ids = [block_id for _, block_id, _ in sources]
                addresses.extend(local_addresses(self.context.local[group_id][partition], block_ids).tolist())
                sizes.extend([list(self.context.sizes[group_id][partition])] * len(block_ids))
            return selection.keys, addresses, sizes

        keys: list[str] = []
        for (group_id, partition), local in selection.iter_local():
            keys.extend(selection.keys[index] for index in local.object_indices)
            addresses.extend(local_addresses(self.context.local[group_id][partition], local.block_ids).tolist())
            sizes.extend([list(self.context.sizes[group_id][partition])] * len(local.block_ids))
        return keys, addresses, sizes
