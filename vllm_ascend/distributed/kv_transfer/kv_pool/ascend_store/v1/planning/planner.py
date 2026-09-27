"""Plan KV movement from Scheduler request progress."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.utils.math_utils import cdiv

from ..graph.coordinates import TokenRange
from ..protocol.transfer import KVTransferStep, LoadCommand, LoadCommandBatch, StoreCommand, StoreCommandBatch
from .availability import ExternalPrefixPlan, LookupQuery, RemoteAvailabilityProbe
from .progress import LoadCandidate, LoadPublication, RequestSnapshot
from .spec import TransferPlanningSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.request import Request


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
        self.requests: dict[str, Request] = {}
        self.preempted_request_ids: set[str] = set()

    def lookup(self, query: LookupQuery) -> ExternalPrefixPlan:
        availability = self._availability_probe.query(query)
        if availability is None:
            return ExternalPrefixPlan(0, False)
        candidate = LoadCandidate(availability.load_range, availability.matched_end_token)
        self._pending_loads[query.request_id] = candidate
        return ExternalPrefixPlan(
            availability.matched_end_token - query.local_cached_tokens,
            self._load_publication.is_deferred,
        )

    def confirm_allocation(
        self, request: Request, blocks: tuple[list[int], ...], allocated_external_tokens: int
    ) -> None:
        request_id = request.request_id
        self.requests[request_id] = request
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
            tuple(tuple(group_block_ids) for group_block_ids in blocks),
            tuple(request.block_hashes),
            len(request.prompt_token_ids),
        )

    def build_step(self, scheduler_output: SchedulerOutput) -> KVTransferStep:
        self._discard_finished_and_preempted(scheduler_output)
        load_commands, store_commands = self._plan_scheduled_requests(scheduler_output)
        load_commands.extend(self._publish_allocation_ready_load_commands())
        return KVTransferStep(
            LoadCommandBatch(tuple(load_commands)),
            StoreCommandBatch(tuple(store_commands)),
        )

    def close(self) -> None:
        self._availability_probe.close()

    def _discard_finished_and_preempted(self, scheduler_output: SchedulerOutput) -> None:
        for request_id in scheduler_output.finished_req_ids:
            self._discard_request(request_id)
        for request_id in scheduler_output.preempted_req_ids:
            self.preempted_request_ids.add(request_id)
            self.request_progress.pop(request_id, None)
            self.requests.pop(request_id, None)
            self._pending_loads.pop(request_id, None)
            self._load_publication.discard(request_id)

    def _discard_request(self, request_id: str) -> None:
        self.request_progress.pop(request_id, None)
        self.requests.pop(request_id, None)
        self.preempted_request_ids.discard(request_id)
        self._pending_loads.pop(request_id, None)
        self._load_publication.discard(request_id)

    def _plan_scheduled_requests(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[LoadCommand], list[StoreCommand]]:
        load_commands: list[LoadCommand] = []
        store_commands: list[StoreCommand] = []
        for scheduled_request in scheduler_output.scheduled_new_reqs:
            self._append_commands(
                self._plan_new_request(scheduled_request, scheduler_output),
                load_commands,
                store_commands,
            )
        if not self._store_enabled:
            return load_commands, store_commands
        cached = scheduler_output.scheduled_cached_reqs
        for index, request_id in enumerate(cached.req_ids):
            new_block_ids = cached.new_block_ids[index]
            if request_id in self.preempted_request_ids:
                if not new_block_ids:
                    continue
                commands = self._plan_resumed_request(request_id, new_block_ids, scheduler_output)
            else:
                commands = self._plan_running_request(request_id, new_block_ids, scheduler_output)
            self._append_commands(commands, load_commands, store_commands)
        return load_commands, store_commands

    def _publish_allocation_ready_load_commands(self) -> list[LoadCommand]:
        commands = []
        for request_id, candidate in self._load_publication.take_ready():
            request = self.requests.get(request_id)
            snapshot = self.request_progress.get(request_id)
            if request is None or snapshot is None:
                raise ValueError(f"Request {request_id} is ready for asynchronous Load without allocated blocks")
            load_command, store_command = self._plan_commands(snapshot, candidate)
            if load_command is None or store_command is not None:
                raise ValueError(f"Request {request_id} did not produce an asynchronous Load command")
            commands.append(load_command)
        return commands

    def _plan_new_request(
        self, scheduled_request: NewRequestData, scheduler_output: SchedulerOutput
    ) -> tuple[LoadCommand | None, StoreCommand | None]:
        request_id = scheduled_request.req_id
        candidate = self._take_scheduled_load(request_id)
        request = self.requests.get(request_id)
        if request is None:
            raise ValueError(f"Request {request_id} is not in request progress, but it is scheduled as a new request")
        snapshot = RequestSnapshot(
            request_id,
            scheduled_request.num_computed_tokens + scheduler_output.num_scheduled_tokens[request_id],
            tuple(tuple(group_block_ids) for group_block_ids in scheduled_request.block_ids),
            tuple(request.block_hashes),
            len(request.prompt_token_ids),
        )
        self.request_progress[request_id] = snapshot
        return self._plan_commands(snapshot, candidate)

    def _plan_resumed_request(
        self, request_id: str, new_block_ids: tuple[list[int], ...], scheduler_output: SchedulerOutput
    ) -> tuple[LoadCommand | None, StoreCommand | None]:
        self.preempted_request_ids.discard(request_id)
        candidate = self._take_scheduled_load(request_id)
        request = self.requests.get(request_id)
        if request is None:
            raise ValueError(f"Request {request_id} is not in request progress, but it is scheduled after preemption")
        snapshot = RequestSnapshot(
            request_id,
            request.num_computed_tokens + scheduler_output.num_scheduled_tokens[request_id],
            tuple(tuple(group_block_ids) for group_block_ids in new_block_ids),
            tuple(request.block_hashes),
            len(request.prompt_token_ids),
        )
        self.request_progress[request_id] = snapshot
        return self._plan_commands(snapshot, candidate)

    def _plan_running_request(
        self, request_id: str, new_block_ids: tuple[list[int], ...] | None, scheduler_output: SchedulerOutput
    ) -> tuple[LoadCommand | None, StoreCommand | None]:
        request = self.requests.get(request_id)
        is_decoding = request is not None and request.num_computed_tokens >= request.num_prompt_tokens
        if is_decoding and not self._save_decode_cache:
            return None, None
        snapshot = self.request_progress.get(request_id)
        if snapshot is None:
            raise ValueError(f"Request {request_id} has no progress snapshot")
        if request is None:
            raise ValueError(f"Request {request_id} has no active vLLM request")
        snapshot = snapshot.advance(
            scheduler_output.num_scheduled_tokens[request_id],
            new_block_ids,
            request.block_hashes,
        )
        self.request_progress[request_id] = snapshot
        return self._plan_commands(snapshot, None)

    def _take_scheduled_load(self, request_id: str) -> LoadCandidate | None:
        self._pending_loads.pop(request_id, None)
        return self._load_publication.take_scheduled(request_id)

    def _plan_commands(
        self, snapshot: RequestSnapshot, candidate: LoadCandidate | None
    ) -> tuple[LoadCommand | None, StoreCommand | None]:
        if candidate is not None:
            return (
                LoadCommand(
                    snapshot.request_id,
                    candidate.load_range,
                    snapshot.block_ids_by_group,
                    snapshot.block_hashes,
                ),
                None,
            )
        transfer_end = self._resolve_transfer_end(snapshot.request_token_len, len(snapshot.block_hashes))
        store_range = TokenRange(snapshot.published_store_end_token, transfer_end)
        if not self._should_store(store_range):
            return None, None
        store_command = StoreCommand(
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
        commands: tuple[LoadCommand | None, StoreCommand | None],
        load_commands: list[LoadCommand],
        store_commands: list[StoreCommand],
    ) -> None:
        load_command, store_command = commands
        if load_command is not None:
            load_commands.append(load_command)
        if store_command is not None:
            store_commands.append(store_command)
