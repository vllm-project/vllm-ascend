"""Plan KV movement from Scheduler request progress."""

from __future__ import annotations

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash

from ..coordinates import TokenRange
from ..protocol.transfer import (
    CheckpointStoreCommand,
    KVTransferStep,
    LoadCommand,
    LoadCommandBatch,
    RangeStoreCommand,
    StateCheckpointSource,
    StoreCommand,
    StoreCommandBatch,
)
from .availability import ExternalPrefixPlan, LookupQuery, RemoteAvailabilityProbe
from .progress import LoadCandidate, LoadPublication, RequestSnapshot
from .spec import TransferPlanningSpec
from .step import (
    ScheduledRequest,
    ScheduledRequestKind,
    StateCheckpointHandoff,
    TransferPlanningStep,
)


class TransferPlanner:
    """Track request coordinates and publish immutable Worker transfer commands."""

    def __init__(
        self,
        spec: TransferPlanningSpec,
        availability_probe: RemoteAvailabilityProbe,
        load_publication: LoadPublication,
        *,
        store_enabled: bool,
        save_decode_cache: bool,
    ) -> None:
        self._spec = spec
        self._availability_probe = availability_probe
        self._load_publication = load_publication
        self._store_enabled = store_enabled
        self._save_decode_cache = save_decode_cache
        self._pending_loads: dict[str, LoadCandidate] = {}
        self.request_progress: dict[str, RequestSnapshot] = {}

    def lookup(self, query: LookupQuery) -> ExternalPrefixPlan:
        availability = self._availability_probe.query(query)
        if availability is None:
            return ExternalPrefixPlan(0, False)
        candidate = LoadCandidate(
            availability.load_range,
            availability.matched_end_token,
            availability.tail_key_boundaries,
        )
        self._pending_loads[query.request_id] = candidate
        return ExternalPrefixPlan(
            availability.matched_end_token - query.local_cached_tokens,
            self._load_publication.is_deferred,
        )

    def confirm_allocation(
        self,
        request_id: str,
        block_ids_by_group: tuple[tuple[int, ...], ...],
        block_hashes: tuple[BlockHash, ...],
        num_prompt_tokens: int,
        allocated_external_tokens: int,
    ) -> None:
        candidate = self._pending_loads.get(request_id)
        if candidate is None or allocated_external_tokens == 0:
            return
        expected_tokens = candidate.matched_end_token - candidate.load_range.start_token
        assert allocated_external_tokens == expected_tokens, (
            f"Mismatch in number of tokens: {allocated_external_tokens} vs "
            f"{candidate.matched_end_token} - {candidate.load_range.start_token} for request {request_id}"
        )
        self._pending_loads.pop(request_id)
        self._load_publication.confirm(request_id, candidate)
        self.request_progress[request_id] = RequestSnapshot(
            request_id,
            candidate.load_range.end_token,
            block_ids_by_group,
            block_hashes,
            num_prompt_tokens,
        )

    def build_step(self, planning_step: TransferPlanningStep) -> KVTransferStep:
        self._discard_finished_and_preempted(planning_step)
        load_commands, planned_store_commands = self._plan_scheduled_requests(planning_step.scheduled_requests)
        store_commands: list[StoreCommand] = list(planned_store_commands)
        load_commands.extend(self._publish_allocation_ready_load_commands())
        store_commands.extend(self._plan_checkpoint_stores(planning_step.checkpoint_handoffs))
        return KVTransferStep(
            LoadCommandBatch(tuple(load_commands)),
            StoreCommandBatch(tuple(store_commands)),
        )

    def _plan_checkpoint_stores(self, handoffs: tuple[StateCheckpointHandoff, ...]) -> list[CheckpointStoreCommand]:
        if not self._store_enabled:
            return []

        commands = []
        for handoff in handoffs:
            snapshot = self.request_progress.get(handoff.request_id)
            if snapshot is None:
                continue
            entries_by_boundary: dict[int, list[tuple[int, int]]] = {}
            for source in handoff.sources:
                if (
                    source.group_id in self._spec.transfer_group_ids
                    and source.block_id > 0
                    and 0 < source.boundary_token <= snapshot.request_token_len
                ):
                    entries_by_boundary.setdefault(source.boundary_token, []).append((source.group_id, source.block_id))
            for boundary_token, boundary_entries in sorted(entries_by_boundary.items()):
                commands.append(
                    CheckpointStoreCommand(
                        handoff.request_id,
                        snapshot.block_ids_by_group,
                        handoff.block_hashes,
                        snapshot.published_store_end_token,
                        tuple(
                            StateCheckpointSource(group_id, block_id, boundary_token)
                            for group_id, block_id in boundary_entries
                        ),
                    )
                )
        return commands

    def close(self) -> None:
        self._availability_probe.close()

    def _discard_finished_and_preempted(self, planning_step: TransferPlanningStep) -> None:
        for request_id in planning_step.finished_request_ids:
            self._discard_request(request_id)
        for request_id in planning_step.preempted_request_ids:
            self._discard_request(request_id)

    def _discard_request(self, request_id: str) -> None:
        self.request_progress.pop(request_id, None)
        self._pending_loads.pop(request_id, None)
        self._load_publication.discard(request_id)

    def _plan_scheduled_requests(
        self, scheduled_requests: tuple[ScheduledRequest, ...]
    ) -> tuple[list[LoadCommand], list[RangeStoreCommand]]:
        load_commands: list[LoadCommand] = []
        store_commands: list[RangeStoreCommand] = []
        for scheduled_request in scheduled_requests:
            if scheduled_request.kind is not ScheduledRequestKind.NEW and not self._store_enabled:
                continue
            if scheduled_request.kind is ScheduledRequestKind.NEW:
                commands = self._plan_new_request(scheduled_request)
            elif scheduled_request.kind is ScheduledRequestKind.RESUMED:
                commands = self._plan_resumed_request(scheduled_request)
            else:
                commands = self._plan_running_request(scheduled_request)
            self._append_commands(commands, load_commands, store_commands)
        return load_commands, store_commands

    def _publish_allocation_ready_load_commands(self) -> list[LoadCommand]:
        commands = []
        for request_id, candidate in self._load_publication.take_ready():
            snapshot = self.request_progress.get(request_id)
            if snapshot is None:
                raise ValueError(f"Request {request_id} is ready for asynchronous Load without allocated blocks")
            load_command, store_command = self._plan_commands(snapshot, candidate)
            if load_command is None or store_command is not None:
                raise ValueError(f"Request {request_id} did not produce an asynchronous Load command")
            commands.append(load_command)
        return commands

    def _plan_new_request(self, request: ScheduledRequest) -> tuple[LoadCommand | None, RangeStoreCommand | None]:
        if request.block_ids_by_group is None:
            raise ValueError(f"New request {request.request_id} has no allocated blocks")
        scheduled_end_token = request.num_computed_tokens + request.num_scheduled_tokens
        snapshot = RequestSnapshot(
            request.request_id,
            scheduled_end_token,
            request.block_ids_by_group,
            request.block_hashes,
            request.num_prompt_tokens,
            committable_end_token=min(scheduled_end_token, request.current_token_count),
        )
        self.request_progress[request.request_id] = snapshot
        return self._plan_commands(snapshot, self._take_scheduled_load(request.request_id))

    def _plan_resumed_request(self, request: ScheduledRequest) -> tuple[LoadCommand | None, RangeStoreCommand | None]:
        if request.block_ids_by_group is None:
            raise ValueError(f"Resumed request {request.request_id} has no replacement blocks")
        scheduled_end_token = request.num_computed_tokens + request.num_scheduled_tokens
        snapshot = RequestSnapshot(
            request.request_id,
            scheduled_end_token,
            request.block_ids_by_group,
            request.block_hashes,
            request.num_prompt_tokens,
            committable_end_token=min(scheduled_end_token, request.current_token_count),
        )
        self.request_progress[request.request_id] = snapshot
        return self._plan_commands(snapshot, self._take_scheduled_load(request.request_id))

    def _plan_running_request(self, request: ScheduledRequest) -> tuple[LoadCommand | None, RangeStoreCommand | None]:
        snapshot = self.request_progress.get(request.request_id)
        if snapshot is None:
            raise ValueError(f"Request {request.request_id} has no progress snapshot")
        if request.num_computed_tokens >= snapshot.num_prompt_tokens and not self._save_decode_cache:
            return None, None
        scheduled_end_token = request.num_computed_tokens + request.num_scheduled_tokens
        snapshot = snapshot.with_position(
            scheduled_end_token,
            request.block_ids_by_group,
            request.block_hashes,
            min(scheduled_end_token, request.current_token_count),
        )
        self.request_progress[request.request_id] = snapshot
        return self._plan_commands(snapshot, None)

    def _take_scheduled_load(self, request_id: str) -> LoadCandidate | None:
        self._pending_loads.pop(request_id, None)
        return self._load_publication.take_scheduled(request_id)

    def _plan_commands(
        self, snapshot: RequestSnapshot, candidate: LoadCandidate | None
    ) -> tuple[LoadCommand | None, RangeStoreCommand | None]:
        if candidate is not None:
            return (
                LoadCommand(
                    snapshot.request_id,
                    candidate.load_range,
                    snapshot.block_ids_by_group,
                    snapshot.block_hashes,
                    candidate.tail_key_boundaries,
                ),
                None,
            )
        transfer_end = self._resolve_transfer_end(snapshot.store_end_token, len(snapshot.block_hashes))
        store_range = TokenRange(snapshot.published_store_end_token, transfer_end)
        if not self._should_store(store_range):
            return None, None
        store_command = RangeStoreCommand(
            snapshot.request_id,
            store_range,
            snapshot.block_ids_by_group,
            snapshot.block_hashes,
            snapshot.num_prompt_tokens,
        )
        self.request_progress[snapshot.request_id] = snapshot.with_published_store_end(store_range.end_token)
        return None, store_command

    def _resolve_transfer_end(self, target_token_len: int, num_block_hashes: int) -> int:
        granularity = self._spec.cache_transfer_granularity
        transfer_end = target_token_len
        if self._spec.discard_partial_chunks:
            transfer_end = target_token_len // granularity * granularity
        hashes_per_block = granularity // self._spec.hash_block_size
        full_blocks = target_token_len // granularity
        available_full_blocks = num_block_hashes // hashes_per_block
        missing_boundary_hash = (
            target_token_len > 0 and target_token_len % granularity == 0 and full_blocks > available_full_blocks
        )
        return available_full_blocks * granularity if missing_boundary_hash else transfer_end

    def _should_store(self, store_range: TokenRange) -> bool:
        chunk_boundary = (
            cdiv(store_range.start_token + 1, self._spec.cache_transfer_granularity)
            * self._spec.cache_transfer_granularity
            if self._spec.discard_partial_chunks
            else 0
        )
        return self._store_enabled and store_range.end_token >= chunk_boundary

    @staticmethod
    def _append_commands(
        commands: tuple[LoadCommand | None, RangeStoreCommand | None],
        load_commands: list[LoadCommand],
        store_commands: list[RangeStoreCommand],
    ) -> None:
        load_command, store_command = commands
        if load_command is not None:
            load_commands.append(load_command)
        if store_command is not None:
            store_commands.append(store_command)
