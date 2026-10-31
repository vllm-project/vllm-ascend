"""Prepare Load batches and interpret transport completion by request and block."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from queue import Empty, SimpleQueue

from ....metrics import AscendStoreKVConnectorStats
from ...infrastructure.coordinator import Coordinator
from ...protocol.messages import LoadCommand
from ...protocol.results import DrainResult, LoadFailureLocation, LoadResult
from ..data.batch import LoadBatch
from ..data.context import TransferContext
from ..data.keys import KeyIndex
from ..data.selection import LoadSelection
from ..transfer.bulk import BulkTransfer
from ..transfer.layerwise import LayerwiseTransfer


class LoadService:
    """Own selection, request admission and business completion over a composed transport."""

    def __init__(
        self,
        context: TransferContext,
        coordinator: Coordinator,
        transfer: BulkTransfer | LayerwiseTransfer,
        *,
        require_complete_load: bool = False,
        on_failure: Callable[[frozenset[str]], None] | None = None,
    ) -> None:
        self.context, self.coordinator = context, coordinator
        self.transfer = transfer
        self._require_complete_load = require_complete_load
        self._on_failure = on_failure
        self._accepting = True
        self._pending: deque[LoadBatch] = deque()
        self._request_ids: set[str] = set()
        self._fatal_error: Exception | None = None
        self._timings: SimpleQueue[tuple[float, int]] = SimpleQueue()

    def submit(self, commands: tuple[LoadCommand, ...]) -> None:
        self.check_available()
        if not self._accepting:
            raise RuntimeError("Load no longer accepts submissions")
        if not commands:
            return
        request_ids = tuple(command.request_id for command in commands)
        if len(set(request_ids)) != len(commands):
            raise ValueError("Load commands contain duplicate request IDs")
        if overlap := self._request_ids.intersection(request_ids):
            raise RuntimeError(f"Request already has a pending asynchronous Load: {sorted(overlap)}")
        batch = LoadBatch(request_ids, self._prepare(commands))
        self._request_ids.update(request_ids)
        self._pending.append(batch)
        try:
            self.transfer.submit_write(batch, self._complete, self._check_ready)
        except BaseException as error:
            # Keep ownership after startup failure so drain still checks destination safety.
            if not hasattr(batch, "completion"):
                batch.completion = Future()
            if not batch.completion.done():
                batch.completion.set_exception(error)
            if isinstance(error, Exception):
                self._fatal_error = error
            raise

    def collect(self) -> LoadResult:
        self.check_available()
        completed: set[str] = set()
        failed_requests: set[str] = set()
        failed_blocks: set[int] = set()
        locations: dict[LoadFailureLocation, None] = {}
        while self._pending:
            batch = self._pending[0]
            if not batch.completion.done():
                break
            result = self._observe(batch)
            self._pending.popleft()
            completed.update(result.completed_request_ids)
            failed_requests.update(result.failed_request_ids)
            failed_blocks.update(result.failed_block_ids)
            locations.update(dict.fromkeys(result.failed_locations))
        self._request_ids.difference_update(completed)
        return LoadResult(frozenset(completed), frozenset(failed_requests), frozenset(failed_blocks), tuple(locations))

    def stop_accepting(self) -> None:
        self._accepting = False

    def take_stats(self) -> AscendStoreKVConnectorStats | None:
        stats = AscendStoreKVConnectorStats()
        while True:
            try:
                duration, count = self._timings.get_nowait()
            except Empty:
                return None if stats.is_empty() else stats
            stats.record_operation("load_get", duration, count)

    def drain(self) -> DrainResult:
        """Wait for every submitted task without consuming normal completion notifications."""
        self.stop_accepting()
        error: BaseException | None = self._fatal_error
        for batch in self._pending:
            try:
                self._observe(batch)
            except BaseException as caught:
                error = error or caught
        return DrainResult(self.safe_to_release, error)

    def check_available(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError("Worker cannot continue after a fatal Load failure") from self._fatal_error

    @property
    def pending(self) -> bool:
        return bool(self._pending)

    @property
    def safe_to_release(self) -> bool:
        return all(batch.completion.done() and batch.destination_writes_stopped for batch in self._pending)

    # ================================================
    # Object selection and business results
    # ================================================

    def _complete(self, batch: LoadBatch) -> LoadResult:
        if batch.transfer_seconds > 0:
            self._timings.put((batch.transfer_seconds, len(batch.selection.keys)))
        failures = [value for _, value in sorted(batch.failures.items(), key=lambda item: (item[1][0], item[0]))]
        return LoadResult(
            frozenset(batch.request_ids),
            frozenset(batch.request_ids[request_index] for request_index, _ in failures),
            frozenset(location.block_id for _, location in failures),
            tuple(dict.fromkeys(location for _, location in failures)),
        )

    def _check_ready(self, batch: LoadBatch, failed_indices: set[int] | None = None) -> None:
        """Interpret each failed destination once, then enforce the caller's consumption policy."""
        failed_requests: set[str] = set()
        failed_locations: set[tuple[int, int]] = set()
        selection = batch.selection
        if len(batch.codes) != selection.target_count:
            raise ValueError("Backend Load result count does not match its parameters")
        indices = iter(
            sorted(failed_indices)
            if failed_indices is not None
            else [index for index, code in enumerate(batch.codes) if code != 0]
        )
        index = next(indices, None)
        for (group_id, _), local in selection.iter_local():
            while index is not None and index < local.start + len(local.block_ids):
                position = index - local.start
                if position < 0:
                    raise ValueError("Load result has no destination ownership")
                if index not in batch.failures:
                    request_index, block_id = local.request_indices[position], local.block_ids[position]
                    batch.failures[index] = (request_index, LoadFailureLocation(group_id, block_id))
                    failed_requests.add(batch.request_ids[request_index])
                    failed_locations.add((group_id, block_id))
                index = next(indices, None)
            if index is None:
                break
        if index is not None:
            raise ValueError("Load result has no destination ownership")
        if failed_requests and self._on_failure is not None:
            self._on_failure(frozenset(failed_requests))
        if failed_requests and self._require_complete_load:
            raise RuntimeError(
                f"Hybrid KV Load failed for requests {sorted(failed_requests)}; "
                f"cache group/block failures {sorted(failed_locations)}"
            )

    def _prepare(self, commands: tuple[LoadCommand, ...]) -> LoadSelection:
        context = self.context
        selection = LoadSelection(context.partition_order)
        key_index = KeyIndex(keys=selection.keys)
        for request_index, command in enumerate(commands):
            if command.start_token < 0 or command.end_token < command.start_token:
                raise ValueError("Load range requires 0 <= start_token <= end_token")
            tails = {item.group_id: item.boundary_token for item in command.tail_key_boundaries}
            if len(tails) != len(command.tail_key_boundaries) or tails.keys() - context.groups.keys():
                raise ValueError("Load contains duplicate or unknown tail-key cache groups")
            if (
                command.start_token == command.end_token
                or not command.block_hashes
                or not any(command.block_ids_by_group)
            ):
                continue
            masks = self.coordinator.select_for_load(command.block_hashes, command.end_token)
            for group_id, mask in zip(context.groups, masks, strict=True):
                if mask == () or not command.block_ids_by_group[group_id]:
                    continue
                minimum_block_id = 1 if group_id in context.state_group_ids else 0
                blocks = context.block_hash_mappings[group_id].pair_block_ids(
                    command.block_hashes,
                    command.block_ids_by_group[group_id],
                    start_token=command.start_token,
                    end_token=command.end_token,
                    tail_boundary=tails.get(group_id),
                )
                for logical_index, block_hash, block_id in blocks:
                    if mask is not None and (logical_index >= len(mask) or not mask[logical_index]):
                        continue
                    if block_id < minimum_block_id:
                        continue
                    for partition, parts in context.key_parts[group_id].items():
                        selection.add_target(
                            key_index.identify(parts, block_hash),
                            partition,
                            block_id,
                            request_index,
                            context.object_sizes[partition],
                        )
        if selection.keys:
            selection.bind_results()
        return selection

    def _observe(self, batch: LoadBatch) -> LoadResult:
        try:
            return batch.completion.result()
        except Exception as error:
            self._fatal_error = self._fatal_error or error
            raise
