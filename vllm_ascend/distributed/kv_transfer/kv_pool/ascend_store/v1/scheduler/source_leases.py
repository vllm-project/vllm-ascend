"""Scheduler ownership of block references read by asynchronous Store jobs."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, TypeVar, cast

from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from ..infrastructure.configuration import KVPoolConfig
from ..infrastructure.coordinator import checkpoint_companion_indices, store_block_range, store_group_mask
from ..infrastructure.protocol.messages import CheckpointStoreCommand, StoreCommand

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool

StoreCommandT = TypeVar("StoreCommandT", bound=StoreCommand)


class StoreSourceLeases:
    """Pin only sources that can be freed before the production reuse fence."""

    def __init__(self, config: KVPoolConfig) -> None:
        self._config = config
        self._expected_worker_count = config.expected_worker_count
        self._block_pool: BlockPool | None = None
        self._next_job_id = 0
        self._leases: dict[int, tuple[tuple[int, ...], int]] = {}
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

    def has_pending_block_frees(self) -> bool:
        """Report blocks held only by Store, on the allocation-failure slow path."""

        if self._block_pool is None or not self._leases:
            return False
        # Each job touches a block once, independently of how many Workers still owe its acknowledgement.
        source_refs = Counter(block_id for block_ids, _ in self._leases.values() for block_id in block_ids)
        return any(self._block_pool.blocks[block_id].ref_cnt == count for block_id, count in source_refs.items())

    def pending_job_ids(self) -> tuple[int, ...]:
        return tuple(self._leases)

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
                        boundary, command.published_store_end_token, block_size, len(block_ids)
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
