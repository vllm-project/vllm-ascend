"""Scheduler ownership of block references read by asynchronous Store jobs."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, TypeVar, cast

from ..protocol.transfer import CheckpointStoreCommand, StoreCommand

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool

StoreCommandT = TypeVar("StoreCommandT", bound=StoreCommand)


class StoreSourceLeases:
    """Keep Store source blocks alive until every Worker stops reading them."""

    def __init__(
        self,
        transfer_group_ids: frozenset[int],
        align_state_group_ids: frozenset[int],
        expected_worker_count: int,
    ) -> None:
        self._transfer_group_ids = transfer_group_ids
        self._align_state_group_ids = align_state_group_ids
        self._expected_worker_count = expected_worker_count
        self._block_pool: BlockPool | None = None
        self._next_job_id = 0
        self._leases: dict[int, tuple[tuple[int, ...], int]] = {}

    def bind_block_pool(self, block_pool: BlockPool) -> None:
        self._block_pool = block_pool

    def acquire(self, command: StoreCommandT) -> StoreCommandT:
        if command.store_job_id is not None:
            return command
        store_job_id = self._next_job_id
        self._next_job_id += 1
        block_ids = self._source_block_ids(command)
        if block_ids:
            if self._block_pool is None:
                raise RuntimeError("GPU block pool must be bound before a Store command is published")
            self._block_pool.touch([self._block_pool.blocks[block_id] for block_id in block_ids])
            self._leases[store_job_id] = (block_ids, self._expected_worker_count)
        return cast(StoreCommandT, replace(command, store_job_id=store_job_id))

    def release(self, released_store_jobs: dict[int, int]) -> None:
        for store_job_id, count in released_store_jobs.items():
            lease = self._leases.get(store_job_id)
            if lease is None:
                continue
            block_ids, remaining = lease
            remaining -= count
            if remaining < 0:
                raise RuntimeError(f"Store job {store_job_id} reported source release by too many Workers")
            if remaining:
                self._leases[store_job_id] = (block_ids, remaining)
                continue
            if self._block_pool is None:
                raise RuntimeError("GPU block pool is unavailable while Store source leases are active")
            del self._leases[store_job_id]
            self._block_pool.free_blocks(self._block_pool.blocks[block_id] for block_id in reversed(block_ids))

    def has_pending(self) -> bool:
        return bool(self._leases)

    def pending_job_ids(self) -> tuple[int, ...]:
        return tuple(self._leases)

    def _source_block_ids(self, command: StoreCommand) -> tuple[int, ...]:
        source_ids: list[int] = []
        if isinstance(command, CheckpointStoreCommand):
            source_ids.extend(source.block_id for source in command.sources)
        for group_id, group_block_ids in enumerate(command.block_ids_by_group):
            if group_id in self._transfer_group_ids and group_id not in self._align_state_group_ids:
                source_ids.extend(group_block_ids)
        return tuple(dict.fromkeys(block_id for block_id in source_ids if block_id > 0))
