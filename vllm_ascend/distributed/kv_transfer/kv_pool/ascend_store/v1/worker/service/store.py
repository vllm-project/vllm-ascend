"""Store missing objects, publish complete writes and report source reuse independently."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from functools import partial
from queue import Empty, SimpleQueue

from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from ...infrastructure.collective import CollectiveGroup, GroupCompletion
from ...infrastructure.coordinator import Coordinator, checkpoint_companion_indices
from ...infrastructure.mapping import BlockPair
from ...protocol.messages import CheckpointStoreCommand, StoreCommand
from ...protocol.outcomes import DrainResult, StoreResult
from ..data.batch import StoreBatch
from ..data.context import TransferContext
from ..data.selection import StoreSelection
from ..transfer.bulk import BulkTransfer
from ..transfer.layerwise import LayerwiseTransfer


class StoreService:
    """Own object selection, admission, publication decisions and source-release reporting."""

    def __init__(
        self,
        context: TransferContext,
        coordinator: Coordinator,
        transfer: BulkTransfer | LayerwiseTransfer,
        *,
        requires_exists: bool,
        collective: CollectiveGroup | None = None,
    ) -> None:
        self.context, self.coordinator = context, coordinator
        self.transfer = transfer
        self._requires_exists = requires_exists
        self._accepting = True
        self._pending: deque[StoreBatch] = deque()
        self._latest: dict[str, StoreBatch] = {}
        self._released: set[int] = set()
        self._fatal_error: Exception | None = None
        self._events: SimpleQueue[BlockStored] = SimpleQueue()
        topology = context.topology
        self._writers: dict[int, slice] = {}
        self._completion_group: GroupCompletion[str, object] | None = None
        if transfer.shared_objects:
            writers = tuple(min(peers) for peers in topology.context_replica_groups)
            if transfer.read_enabled and not writers:
                raise ValueError("GVA Store requires resolved context replica groups")
            selected = slice(None) if topology.global_rank in writers else slice(0, 0)
            self._writers = dict.fromkeys(context.groups, selected)
            if transfer.read_enabled and topology.config.cp_partition.object_shard_count > 1:
                if collective is None:
                    raise ValueError("Shared GVA Store requires a collective group")
                self._completion_group = GroupCompletion(writers, collective)
        else:
            for group_id, group in context.groups.items():
                replicas = (
                    1
                    if (
                        topology.config.tp_partition.tp_mismatch
                        or group.uses_align_state
                        or topology.config.cp_partition.dcp_size > 1
                    )
                    else topology.config.put_step
                )
                rank = topology.pcp_rank * replicas + topology.tp_rank % replicas
                count = topology.config.cp_partition.pcp_size * replicas
                self._writers[group_id] = slice(rank, None, count) if count > 1 else slice(None)

        self._active_groups = frozenset(
            group_id
            for group_id in context.groups
            if group_id not in context.state_group_ids and self._writers[group_id].stop != 0
        )

    def submit(self, commands: tuple[StoreCommand, ...]) -> bool:
        """Register source ownership before launching preparation on the Store queue."""
        self.check_available()
        if not self._accepting:
            raise RuntimeError("Store no longer accepts submissions")
        if not self.transfer.read_enabled or not commands:
            return False
        batch = StoreBatch(commands, needs_shared_confirmation=self._completion_group is not None)
        try:
            self._pending.append(batch)
            try:
                self.transfer.submit_read(batch, partial(self._prepare, commands), self._complete)
            except BaseException:
                self._pending.pop()
                raise
            self._latest.update((request_id, batch) for request_id in batch.request_ids)
            return True
        except Exception as error:
            self._fatal_error = error
            raise

    def fence(self, released_request_ids: frozenset[str] | None = None) -> tuple[StoreResult, ...]:
        """None seals/waits for all jobs; an empty set only collects ready jobs.

        A nonempty set waits through the latest job for each listed request.
        Active layer producers must be sealed before their sources can be released.
        """
        self.check_available()
        if released_request_ids is None:
            self.transfer.seal_reads()
        elif released_request_ids:
            for request_id in released_request_ids:
                if batch := self._latest.get(request_id):
                    self.transfer.check_read_release(batch)
        required = (
            set()
            if released_request_ids is None
            else {self._latest[request_id] for request_id in released_request_ids if request_id in self._latest}
        )
        collected: list[StoreResult] = []
        try:
            while self._pending:
                batch = self._pending[0]
                if released_request_ids is not None and not required and not batch.completion.done():
                    break
                collected.extend(self._consume(batch))
                self._pending.popleft()
                required.discard(batch)
        except Exception as error:
            self._fatal_error = error
            raise
        return tuple(collected)

    def stop_accepting(self) -> None:
        self._accepting = False

    def invalidate_requests(self, request_ids: frozenset[str]) -> None:
        """Invalidate only the current sources selected for requests whose Load failed."""
        for request_id in request_ids:
            if batch := self._latest.get(request_id):
                batch.invalidate_sources((request_id,))

    def drain(self) -> DrainResult:
        """Observe every sealed job; retain jobs without confirmed safe source release."""
        self.stop_accepting()
        error: BaseException | None = self._fatal_error
        retained: deque[StoreBatch] = deque()
        while self._pending:
            batch = self._pending.popleft()
            try:
                self._consume(batch)
            except BaseException as caught:
                retained.append(batch)
                if self._fatal_error is None and isinstance(caught, Exception):
                    self._fatal_error = caught
                error = error or caught
        self._pending = retained
        return DrainResult(not retained, error or self._fatal_error)

    def _consume(self, batch: StoreBatch) -> tuple[StoreResult, ...]:
        completions = batch.completion.result()
        self._accept(completions, batch)
        for request_id in batch.request_ids:
            if self._latest.get(request_id) is batch:
                del self._latest[request_id]
        return completions

    def take_released_jobs(self) -> set[int]:
        released, self._released = self._released, set()
        return released

    def take_events(self) -> list[BlockStored]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except Empty:
                return events

    def check_available(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError("Worker cannot continue after a fatal Store failure") from self._fatal_error

    @property
    def pending(self) -> bool:
        return bool(self._pending)

    # ================================================
    # Publication and business results
    # ================================================

    def _complete(self, batch: StoreBatch) -> tuple[StoreResult, ...]:
        selection = batch.selection
        if selection is not None and selection.keys:
            if batch.open_failed:
                batch.codes = [None] * len(selection.keys)
            if len(batch.codes) != len(selection.keys):
                raise ValueError("Backend Store result count does not match its parameters")
            for index in tuple(batch.invalid_objects):
                batch.codes[index] = None
        commit = [
            index for index in batch.opened_indices if batch.codes[index] == 0 and index not in batch.unsafe_objects
        ]
        # Every participant joins, including empty selections and failed preparation.
        commit = self._confirm_publication(batch, commit)
        if commit:
            self.transfer.commit_read(batch, commit)
        if batch.open_failed:
            self.transfer.revoke_read(batch, None)
        else:
            revoke = [
                index for index in batch.opened_indices if batch.codes[index] != 0 and index not in batch.unsafe_objects
            ]
            if revoke:
                self.transfer.revoke_read(batch, revoke)
        if batch.contract_error is not None:
            raise batch.contract_error
        if selection is None:
            return tuple(
                StoreResult(command.request_id, command.store_job_id, succeeded=False, error=batch.error)
                for command in batch.commands
            )
        if selection.events:
            successful = {
                key
                for index, key in enumerate(selection.keys)
                if batch.codes[index] == 0 and index not in batch.unsafe_objects
            }
            for event, keys, owner in selection.events:
                if (not keys or batch.commands[owner].request_id not in batch.invalid_requests) and all(
                    key in successful for key in keys
                ):
                    self._events.put(event)
        return self._results(batch)

    def _confirm_publication(self, batch: StoreBatch, commit: list[int]) -> list[int]:
        if self._completion_group is None:
            return commit
        selection = batch.selection
        try:
            local_completed = (
                {
                    selection.keys[index]: batch.identities[selection.keys[index]]
                    for index in commit
                    if selection.keys[index] in batch.identities
                }
                if selection is not None
                else {}
            )
            completed = self._completion_group.resolve(local_completed)
        except Exception as error:
            batch.fatal_error = error
            batch.error = batch.error or error
            completed = set()
        confirmed = []
        for index in commit:
            assert selection is not None
            if selection.keys[index] in completed:
                confirmed.append(index)
            else:
                batch.error = batch.error or RuntimeError("Shared GVA object has an incomplete context shard")
                batch.codes[index] = None
        return confirmed

    # ================================================
    # Object selection and business results
    # ================================================

    def _prepare(self, commands: tuple[StoreCommand, ...]) -> StoreSelection:
        selection = self._candidates(commands)
        if not selection.keys or not self._requires_exists:
            return selection
        present = self.transfer.backend.exists(selection.keys)
        if len(present) != len(selection.keys):
            raise ValueError("Backend exists result count does not match its parameters")
        # No identities or ownership change when every candidate is missing.
        if not any(present):
            return selection
        missing = StoreSelection()
        for key, source, hit in zip(selection.keys, selection.sources, present, strict=True):
            if not hit:
                missing.keys.append(key)
                missing.sources.append(source)
        if selection.events:
            present_keys = {key for key, hit in zip(selection.keys, present, strict=True) if hit}
            missing.events = [
                (event, tuple(key for key in keys if key not in present_keys), owner)
                for event, keys, owner in selection.events
            ]
        return missing

    def _candidates(self, commands: tuple[StoreCommand, ...]) -> StoreSelection:
        context = self.context
        selection = StoreSelection()
        seen_keys: set[str] = set()
        for request_index, command in enumerate(commands):
            if isinstance(command, CheckpointStoreCommand):
                self._checkpoint_candidates(command, request_index, selection, seen_keys)
                continue
            if command.start_token < 0 or command.end_token < command.start_token:
                raise ValueError("Store range requires 0 <= start_token <= end_token")
            if (
                not self._active_groups
                or command.end_token <= command.start_token
                or not command.block_hashes
                or not any(command.block_ids_by_group)
            ):
                continue
            masks = self.coordinator.select_for_store(command.start_token, command.end_token, command.num_prompt_tokens)
            for group_id, mask in zip(context.groups, masks, strict=True):
                if group_id not in self._active_groups or mask == () or not command.block_ids_by_group[group_id]:
                    continue
                block_mapping = context.block_hash_mappings[group_id]
                mask_start = max(command.start_token // block_mapping.block_size, 0)
                blocks = block_mapping.pair_block_ids(
                    command.block_hashes,
                    command.block_ids_by_group[group_id],
                    start_token=command.start_token,
                    end_token=command.end_token,
                )
                self._add_candidates(request_index, group_id, blocks, selection, seen_keys, command, mask, mask_start)
        # Preserve public per-partition result order while keeping each selected source unchanged.
        if len(context.partition_order) > 1 and selection.keys:
            ordered = sorted(
                zip(selection.keys, selection.sources, strict=True),
                key=lambda item: context.partition_order[item[1][0]],
            )
            selection.keys = [key for key, _ in ordered]
            selection.sources = [source for _, source in ordered]
        return selection

    def _add_candidates(
        self,
        request_index: int,
        group_id: int,
        blocks: Iterable[BlockPair],
        selection: StoreSelection,
        seen_keys: set[str],
        command: StoreCommand,
        mask: tuple[bool, ...] | None = None,
        mask_start: int = 0,
    ) -> None:
        writer = self._writers[group_id]
        if writer.stop == 0:
            return
        skip, stride = writer.start or 0, writer.step or 1
        partitions = self.context.key_parts[group_id].items()
        for logical_index, block_hash, block_id in blocks:
            if block_id < 0 or (
                mask is not None and (logical_index - mask_start >= len(mask) or not mask[logical_index - mask_start])
            ):
                continue
            # Writer positions count valid, unmasked candidates, before identity deduplication.
            if skip:
                skip -= 1
                continue
            skip = stride - 1
            value = block_hash.hex() if isinstance(block_hash, bytes) else block_hash
            first = len(selection.keys)
            for partition, (prefix, suffix) in partitions:
                key = prefix + value + suffix
                if key not in seen_keys:
                    seen_keys.add(key)
                    selection.keys.append(key)
                    selection.sources.append((partition, block_id, request_index))
            if command.event_tokens is not None and len(selection.keys) != first:
                token_start, token_ids = command.event_tokens
                mapping = self.context.block_hash_mappings[group_id]
                start = logical_index * mapping.block_size
                end = min(start + mapping.block_size, token_start + len(token_ids))
                parent = mapping.hash_at_boundary(start, command.block_hashes) if start else None
                event = BlockStored(
                    block_hashes=[maybe_convert_block_hash(block_hash)],
                    parent_block_hash=maybe_convert_block_hash(parent) if parent is not None else None,
                    token_ids=list(token_ids[start - token_start : end - token_start]),
                    block_size=mapping.block_size,
                    lora_id=None,
                    medium="cpu",
                    lora_name=None,
                )
                selection.events.append((event, tuple(selection.keys[first:]), request_index))

    def _checkpoint_candidates(
        self, command: CheckpointStoreCommand, request_index: int, selection: StoreSelection, seen_keys: set[str]
    ) -> None:
        context = self.context
        groups = tuple(source.group_id for source in command.sources)
        if not context.state_group_ids or len(set(groups)) != len(groups) or set(groups) - context.state_group_ids:
            raise ValueError("State checkpoint Store requires unique align-state cache-group sources")
        boundaries = {source.boundary_token for source in command.sources}
        if len(boundaries) != 1:
            raise ValueError("State checkpoints for one request must share one token boundary")
        boundary = next(iter(boundaries))
        value = context.block_hash_mappings[groups[0]].hash_at_boundary(boundary, command.block_hashes)
        partial = False
        for source in command.sources:
            if source.block_id <= 0:
                continue
            block_mapping = context.block_hash_mappings[source.group_id]
            block = (-(-boundary // block_mapping.block_size) - 1, value, source.block_id)
            self._add_candidates(request_index, source.group_id, (block,), selection, seen_keys, command)
            partial |= bool(boundary % block_mapping.block_size)
        if partial:
            for group_id, block_mapping in context.block_hash_mappings.items():
                if group_id in context.state_group_ids:
                    continue
                block_ids = command.block_ids_by_group[group_id]
                block_size = block_mapping.block_size
                offset = max(-(-boundary // block_size) - len(block_ids), 0)
                blocks = (
                    (
                        index + offset,
                        block_mapping.hash_at_boundary(
                            min((index + offset + 1) * block_size, boundary), command.block_hashes
                        ),
                        block_ids[index],
                    )
                    for index in checkpoint_companion_indices(
                        boundary, command.store_planned_end_token, block_size, len(block_ids)
                    )
                    if block_ids[index] >= 0
                )
                self._add_candidates(request_index, group_id, blocks, selection, seen_keys, command)

    def _results(self, batch: StoreBatch) -> tuple[StoreResult, ...]:
        assert batch.selection is not None
        if not batch.selection.keys:
            return tuple(StoreResult(command.request_id, command.store_job_id) for command in batch.commands)
        keys_by_request: list[list[str]] = [[] for _ in batch.commands]
        codes_by_request: list[list[int | None]] = [[] for _ in batch.commands]
        succeeded = [True] * len(batch.commands)
        unsafe_requests: set[int] = set()
        for index, (_, _, request_index) in enumerate(batch.selection.sources):
            keys_by_request[request_index].append(batch.selection.keys[index])
            code = batch.codes[index]
            codes_by_request[request_index].append(code)
            if code != 0:
                succeeded[request_index] = False
            if index in batch.unsafe_objects:
                unsafe_requests.add(request_index)
        return tuple(
            StoreResult(
                command.request_id,
                command.store_job_id,
                tuple(keys_by_request[index]),
                tuple(codes_by_request[index]),
                succeeded=succeeded[index],
                source_release_confirmed=index not in unsafe_requests,
                error=(
                    RuntimeError("Store source depends on a failed Load")
                    if command.request_id in batch.invalid_requests and not succeeded[index]
                    else batch.error
                    if not succeeded[index]
                    else None
                ),
            )
            for index, command in enumerate(batch.commands)
        )

    def _accept(self, results: tuple[StoreResult, ...], batch: StoreBatch) -> None:
        if batch.fatal_error is not None:
            self._fatal_error = self._fatal_error or batch.fatal_error
        seen: set[int] = set()
        unsafe: StoreResult | None = None
        for result in results:
            job_id = result.store_job_id
            if job_id is not None:
                seen.add(job_id)
                if result.source_release_confirmed and job_id not in batch.reported_jobs:
                    self._released.add(job_id)
                    batch.reported_jobs.add(job_id)
            if result.error is not None or not result.succeeded:
                logger.error(
                    "KV cache Store failed for request %s (job %s); result codes=%s, error=%s",
                    result.request_id,
                    job_id,
                    result.result_codes,
                    result.error,
                )
            if not result.source_release_confirmed and unsafe is None:
                unsafe = result
        # Record every confirmed release before surfacing another request's unsafe source.
        if unsafe is not None:
            raise RuntimeError(f"Store source release is unknown for request {unsafe.request_id}") from unsafe.error
        if missing := batch.job_ids - seen:
            raise RuntimeError(f"Store source release is missing for jobs {sorted(missing)}")
