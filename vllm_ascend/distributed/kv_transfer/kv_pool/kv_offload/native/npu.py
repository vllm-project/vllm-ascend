# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ascend specs for vLLM's native KV-cache offloading."""

from __future__ import annotations

from typing_extensions import override
from vllm.logger import logger
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingManager, OffloadingWorker
from vllm.v1.kv_offload.cpu import spec as cpu_spec
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec as _CPUOffloadingSpec
from vllm.v1.kv_offload.tiering import spec as tiering_spec
from vllm.v1.kv_offload.tiering.spec import (
    TieringOffloadingSpec as _TieringOffloadingSpec,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.cpu_npu import (
    NPUOffloadingWorker,
)


class _NPUWorkerMixin:
    """Replace only the accelerator worker and upstream platform gate."""

    _worker: NPUOffloadingWorker | None

    def create_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> NPUOffloadingWorker:
        raise NotImplementedError

    def get_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> OffloadingWorker:
        if self._worker is None:
            self._worker = self.create_worker(kv_caches)
        return self._worker


class NPUOffloadingSpec(_NPUWorkerMixin, _CPUOffloadingSpec):
    """Single CPU-tier offloading with pinned host tensors and NPU DMA."""

    @override
    def create_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> NPUOffloadingWorker:
        # Keep the single-tier path on PyTorch's pinned allocator. This spec
        # does not use a shared mmap, so upstream's _uses_shared_region() gate
        # keeps replicated_layout disabled.
        return NPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_chunks,
        )


class NPUTieringOffloadingSpec(_NPUWorkerMixin, _TieringOffloadingSpec):
    """CPU primary tier plus vLLM secondary tiers on Ascend NPU."""

    _manager: OffloadingManager | None

    def _uses_shared_region(self) -> bool:
        # Unlike the CPU-only NPU spec, tiering always connects every worker
        # and the scheduler-side primary tier through SharedOffloadRegion.
        # Advertising that fact lets upstream safely enable its single-copy
        # layout for configurations it has certified as byte-replicated
        # (currently pure MLA under the supported TP topology).
        return True

    def _create_shared_region(self, rank: int | None) -> SharedOffloadRegion:
        region_kwargs = dict(
            engine_id=self._engine_id,
            num_chunks=self.num_chunks,
            rank=rank,
            kv_bytes_per_chunk=self.kv_bytes_per_chunk,
            cpu_page_size=self.cpu_page_size_per_worker,
        )
        # The scheduler must use exactly the same backing files as workers.
        # v0.32 predates segmented mappings and sharded population.
        num_segments = getattr(cpu_spec, "REGION_SEGMENTS", getattr(cpu_spec, "_REGION_SEGMENTS", None))
        if num_segments is not None:
            region_kwargs.update(
                num_segments=num_segments,
                populate_shard=(
                    (self.config.parallel.rank, self.config.parallel.world_size)
                    if rank is not None and self.replicated_layout
                    else None
                ),
            )
        return SharedOffloadRegion(**region_kwargs)

    def get_manager(self) -> OffloadingManager:
        # Upstream hardcodes its single-file mapping inside get_manager.
        # Reuse its managers with the same segmented region as NPU workers,
        # without changing upstream module globals or secondary-tier logic.
        if self._manager is not None:
            return self._manager
        if int(self.extra_config.get("store_threshold", 0)) >= 2:
            raise ValueError("store_threshold is not supported for TieringOffloadingSpec")

        scheduler_mmap = None
        primary_tier = None
        secondary_tiers = []
        try:
            scheduler_mmap = self._create_shared_region(rank=None)
            self._scheduler_mmap = scheduler_mmap
            primary_kwargs = dict(
                num_chunks=self.num_chunks,
                cache_policy=self.eviction_policy,
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                mmap_region=scheduler_mmap,
            )
            if hasattr(self, "cpu_info"):
                primary_kwargs["info"] = self.cpu_info
            primary_tier = tiering_spec.CPUPrimaryTierOffloadingManager(**primary_kwargs)

            primary_kv_view = primary_tier.get_kv_memoryview()
            for i, tier_config in enumerate(self.secondary_tier_configs):
                tier = tiering_spec.SecondaryTierFactory.create_secondary_tier(tier_config, primary_kv_view, self)
                secondary_tiers.append(tier)
                logger.info("Created secondary tier #%d (%s)", i, tier.tier_type)

            manager_kwargs = dict(primary_tier=primary_tier, secondary_tiers=secondary_tiers)
            if hasattr(self, "tier_poll_interval_s"):
                manager_kwargs["tier_poll_interval_s"] = self.tier_poll_interval_s
            self._manager = tiering_spec.TieringOffloadingManager(**manager_kwargs)
        except Exception:
            for tier in reversed(secondary_tiers):
                try:
                    tier.shutdown()
                except Exception:
                    logger.exception("Failed to shut down secondary tier during initialization cleanup")
            if primary_tier is not None:
                try:
                    primary_tier.shutdown()
                except Exception:
                    logger.exception("Failed to shut down primary tier during initialization cleanup")
            elif scheduler_mmap is not None:
                try:
                    scheduler_mmap.cleanup()
                except Exception:
                    logger.exception("Failed to clean up scheduler mmap during initialization cleanup")
            self._scheduler_mmap = None
            raise

        logger.info(
            "Created TieringOffloadingManager with primary tier (%s, %s chunks) and %s secondary tier(s)",
            self.eviction_policy,
            self.num_chunks,
            len(secondary_tiers),
        )
        return self._manager

    @override
    def create_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> NPUOffloadingWorker:
        # Match the scheduler's rank-based layout, regardless of device selection.
        rank = 0 if self.replicated_layout else self.config.parallel.rank

        pin_shared_mmap = self.extra_config.get("pin_shared_mmap", True)
        if isinstance(pin_shared_mmap, str):
            pin_shared_mmap = pin_shared_mmap.lower() not in ("0", "false", "no")
        else:
            pin_shared_mmap = bool(pin_shared_mmap)

        try:
            worker_mmap = self._create_shared_region(rank)
        except Exception:
            # Release peers waiting for population even when this rank fails.
            try:
                cpu_spec._all_workers_barrier()
            except Exception:
                logger.warning("Failed to release peers after shared-region initialization failure", exc_info=True)
            raise
        try:
            # EngineCore creates the scheduler after workers initialize caches.
            # Passing this barrier into SharedOffloadRegion would unlink files
            # before the scheduler can open them. Synchronize after population
            # instead, and retain backing paths until the creator shuts down.
            cpu_spec._all_workers_barrier()
        except Exception:
            worker_mmap.cleanup()
            raise
        return NPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_chunks,
            mmap_region=worker_mmap,
            pin_shared_mmap=pin_shared_mmap,
            worker_slot_only=not self.replicated_layout,
        )
