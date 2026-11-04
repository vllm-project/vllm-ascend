"""Scheduler Store planning and ownership of sources read by asynchronous jobs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, TypeVar, cast

from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from ..infrastructure.configuration import KVPoolConfig
from ..infrastructure.coordinator import checkpoint_companion_indices, store_block_range, store_group_mask
from ..protocol.messages import (
    CheckpointStoreCommand,
    RangeStoreCommand,
    StateCheckpointSource,
    StoreCommand,
    StoreCommandBatch,
)
from .state import RequestProgress

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.request import Request

StoreCommandT = TypeVar("StoreCommandT", bound=StoreCommand)


class StorePlanner:
    """Derive commands and tentative progress from explicit request facts.

    The Scheduler protects sources and adopts the progress only when the whole
    step is ready to publish. This class neither holds leases nor confirms writes.
    """

    def __init__(self, config: KVPoolConfig) -> None:
        self._config = config
        self._event_block_sizes = (
            tuple(sorted({config.group_specs[group_id].block_size for group_id in config.transfer_group_ids}))
            if config.enable_kv_events
            else ()
        )

    # ====================================
    # Range Commands and Planning Progress
    # ====================================

    def plan_range(
        self, request: Request, progress: RequestProgress
    ) -> tuple[RequestProgress, RangeStoreCommand | None]:
        if not self._config.store_enabled:
            return progress, None
        granularity = self._config.cache_transfer_granularity
        available_end = min(progress.store_end_token, len(progress.block_hashes) * self._config.hash_block_size)
        transfer_end = available_end // granularity * granularity
        store_start = progress.store_planned_end_token
        if not self._config.store_prefill_enabled:
            store_start = max(store_start, progress.prefill_end_token)
        if transfer_end <= store_start or transfer_end < cdiv(store_start + 1, granularity) * granularity:
            return progress, None

        command = RangeStoreCommand(
            progress.request_id,
            store_start,
            transfer_end,
            progress.block_ids_by_group,
            progress.block_hashes,
            progress.prefill_end_token,
            event_tokens=self._event_tokens(request, store_start, transfer_end),
        )
        return progress.with_store_planned_end(transfer_end), command

    # ====================================
    # Ready Checkpoints and Event Snapshots
    # ====================================

    def plan_checkpoints(
        self, request: Request, progress: RequestProgress, entries: Sequence[tuple[int, int, int]]
    ) -> list[CheckpointStoreCommand]:
        if not self._config.store_enabled:
            return []
        entries_by_boundary: dict[int, list[StateCheckpointSource]] = {}
        for group_id, block_id, boundary_token in entries:
            if (
                group_id in self._config.transfer_group_ids
                and group_id in self._config.align_state_group_ids
                and block_id > 0
                and 0 < boundary_token <= progress.request_token_len
            ):
                entries_by_boundary.setdefault(boundary_token, []).append(
                    StateCheckpointSource(group_id, block_id, boundary_token)
                )
        return [
            CheckpointStoreCommand(
                progress.request_id,
                progress.block_ids_by_group,
                progress.block_hashes,
                progress.store_planned_end_token,
                tuple(sources),
                event_tokens=self._event_tokens(
                    request, min(progress.store_planned_end_token, boundary_token - 1), boundary_token
                ),
            )
            for boundary_token, sources in sorted(entries_by_boundary.items())
        ]

    def plan_finished_checkpoint(
        self,
        request: Request,
        progress: RequestProgress,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: Sequence[tuple[int, int, int]],
    ) -> CheckpointStoreCommand | None:
        if not self._config.store_enabled or not partial_tail_offloads or not any(block_ids):
            return None
        boundaries = {boundary for _, _, boundary in partial_tail_offloads}
        if len(boundaries) != 1:
            raise ValueError("Finished state checkpoints must share one token boundary")
        boundary = next(iter(boundaries))
        if boundary <= 0 or boundary > progress.prefill_end_token:
            return None
        if (
            boundary % self._config.hash_block_size
            or boundary > len(request.block_hashes) * self._config.hash_block_size
        ):
            return None
        sources = tuple(
            StateCheckpointSource(group_id, block_id, boundary)
            for group_id, block_id, boundary in partial_tail_offloads
            if group_id in self._config.transfer_group_ids
            and group_id in self._config.align_state_group_ids
            and block_id > 0
        )
        if len(sources) != len(partial_tail_offloads):
            return None
        return CheckpointStoreCommand(
            request.request_id,
            tuple(tuple(group) for group in block_ids),
            tuple(request.block_hashes),
            progress.store_planned_end_token,
            sources,
            event_tokens=self._event_tokens(request, min(progress.store_planned_end_token, boundary - 1), boundary),
        )

    def _event_tokens(self, request: Request, start_token: int, end_token: int) -> tuple[int, tuple[int, ...]] | None:
        if not self._event_block_sizes:
            return None
        start = min(start_token // block_size * block_size for block_size in self._event_block_sizes)
        return start, tuple(request.all_token_ids[start:end_token])


class StoreSourceLeases:
    """Pin only sources that can be freed before the production reuse fence."""

    def __init__(self, config: KVPoolConfig) -> None:
        self._config = config
        self._expected_worker_count = config.expected_worker_count
        self._block_pool: BlockPool | None = None
        self._next_job_id = 0
        self._leases: dict[int, tuple[tuple[int, ...], int, str]] = {}
        store_group_specs = (
            tuple((group_id, config.group_specs[group_id]) for group_id in config.transfer_group_ids)
            if config.store_enabled and not config.use_layerwise
            else ()
        )
        self._group_block_sizes = {group_id: spec.block_size for group_id, spec in store_group_specs}
        effective_specs = tuple(
            (group_id, next(iter(spec.kv_cache_specs.values())) if isinstance(spec, UniformTypeKVCacheSpecs) else spec)
            for group_id, spec in store_group_specs
        )
        eagle_specs = [spec for group_id, spec in store_group_specs if group_id in config.eagle_group_ids]
        eagle_specs = [
            next(iter(spec.kv_cache_specs.values())) if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            for spec in eagle_specs
        ]
        self._recycling_groups = []
        for group_id, spec in effective_specs:
            if group_id in config.align_state_group_ids:
                continue
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            if manager_cls is None:
                raise ValueError(f"No vLLM KV cache manager is registered for {type(spec).__name__}")
            # Full-attention managers retain every page until request release.
            # R-SWA and custom managers may inherit FullAttentionSpec but recycle
            # pages: use the registered manager's reclamation behavior, not its spec name.
            retains_request_blocks = manager_cls is FullAttentionManager or (
                issubclass(manager_cls, FullAttentionManager)
                and manager_cls.remove_skipped_blocks is FullAttentionManager.remove_skipped_blocks
                and manager_cls.get_num_skipped_tokens is FullAttentionManager.get_num_skipped_tokens
            )
            if not retains_request_blocks:
                use_eagle = spec in eagle_specs or (config.use_eagle and not eagle_specs)
                self._recycling_groups.append((group_id, spec, manager_cls, use_eagle))

    def bind_block_pool(self, block_pool: BlockPool) -> None:
        self._block_pool = block_pool

    def acquire(self, command: StoreCommandT) -> StoreCommandT:
        if command.store_job_id is not None:
            return command
        # Ordinary Bulk and Layerwise keep the original command: no job ID,
        # replacement object, block-table scan, or release acknowledgement.
        if not self._recycling_groups and not isinstance(command, CheckpointStoreCommand):
            return command
        block_ids = self._source_block_ids(command)
        if not block_ids:
            return command
        if self._block_pool is None:
            raise RuntimeError("GPU block pool must be bound before a Store command is published")
        store_job_id = self._next_job_id
        self._next_job_id += 1
        self._block_pool.touch([self._block_pool.blocks[block_id] for block_id in block_ids])
        self._leases[store_job_id] = (block_ids, self._expected_worker_count, command.request_id)
        return cast(StoreCommandT, replace(command, store_job_id=store_job_id))

    def acquire_step(self, commands: StoreCommandBatch) -> StoreCommandBatch:
        """Protect a prepared step, rolling back only newly acquired, unpublished jobs.

        Finish-time checkpoints already have leases before vLLM releases their
        sources. Their job IDs predate this call and remain owned on failure.
        """
        if not commands.source_pending_commands and not commands.source_ready_commands:
            return commands
        first_job_id = self._next_job_id
        try:
            return StoreCommandBatch(
                tuple(self.acquire(command) for command in commands.source_pending_commands),
                tuple(self.acquire(command) for command in commands.source_ready_commands),
            )
        except BaseException:
            for job_id in range(first_job_id, self._next_job_id):
                lease = self._leases.get(job_id)
                if lease is not None:
                    assert self._block_pool is not None
                    block_ids, _, _ = lease
                    self._block_pool.free_blocks(self._block_pool.blocks[block_id] for block_id in reversed(block_ids))
                    del self._leases[job_id]
            raise

    def release(self, released_store_jobs: dict[int, int]) -> None:
        for store_job_id, count in released_store_jobs.items():
            lease = self._leases.get(store_job_id)
            if lease is None:
                continue
            block_ids, remaining, request_id = lease
            remaining -= count
            if remaining < 0:
                raise RuntimeError(f"Store job {store_job_id} reported source release by too many Workers")
            if remaining:
                self._leases[store_job_id] = (block_ids, remaining, request_id)
                continue
            if self._block_pool is None:
                raise RuntimeError("GPU block pool is unavailable while Store source leases are active")
            del self._leases[store_job_id]
            self._block_pool.free_blocks(self._block_pool.blocks[block_id] for block_id in reversed(block_ids))

    def has_pending(self) -> bool:
        return bool(self._leases)

    def has_pending_block_frees(self) -> bool:
        """Report blocks held only by Store, on the allocation-failure slow path."""

        if self._block_pool is None or not self._leases:
            return False
        # Each job touches a block once, independently of how many Workers still owe its acknowledgement.
        source_refs = Counter(block_id for block_ids, _, _ in self._leases.values() for block_id in block_ids)
        return any(self._block_pool.blocks[block_id].ref_cnt == count for block_id, count in source_refs.items())

    def pending_job_ids(self) -> tuple[int, ...]:
        return tuple(self._leases)

    def pending_counts(self) -> tuple[int, int]:
        return (
            len({request_id for _, _, request_id in self._leases.values()}),
            len({block for blocks, _, _ in self._leases.values() for block in blocks}),
        )

    def _source_block_ids(self, command: StoreCommand) -> tuple[int, ...]:
        source_ids: list[int] = []
        if isinstance(command, CheckpointStoreCommand):
            source_ids.extend(source.block_id for source in command.sources)
            # Finished-request checkpoints are published after upstream releases
            # the request. Protect their companion pages even for full attention.
            needs_companions = any(
                source.block_id > 0 and source.boundary_token % self._group_block_sizes[source.group_id]
                for source in command.sources
            )
            if needs_companions:
                boundary = command.sources[0].boundary_token
                for group_id, block_size in self._group_block_sizes.items():
                    if group_id in self._config.align_state_group_ids:
                        continue
                    block_ids = command.block_ids_by_group[group_id]
                    indices = checkpoint_companion_indices(
                        boundary, command.store_planned_end_token, block_size, len(block_ids)
                    )
                    source_ids.extend(block_ids[index] for index in indices)
        else:
            for group_id, spec, manager_cls, use_eagle in self._recycling_groups:
                block_ids = command.block_ids_by_group[group_id]
                hash_count = len(command.block_hashes) // (spec.block_size // self._config.hash_block_size)
                indices = store_block_range(
                    command.start_token,
                    command.end_token,
                    spec.block_size,
                    hash_count,
                    len(block_ids),
                )
                if not indices:
                    continue
                block_offset = max(indices.stop - len(block_ids), 0)
                mask = store_group_mask(
                    spec,
                    manager_cls,
                    command.end_token,
                    command.num_prompt_tokens,
                    alignment_tokens=self._config.cache_transfer_granularity,
                    use_eagle=use_eagle,
                    retention_interval=self._config.retention_interval,
                    start_block=indices.start,
                )
                if mask is None:
                    source_ids.extend(block_ids[indices.start - block_offset : indices.stop - block_offset])
                else:
                    source_ids.extend(
                        block_ids[index - block_offset] for index, selected in zip(indices, mask) if selected
                    )
        return tuple(dict.fromkeys(block_id for block_id in source_ids if block_id > 0))
