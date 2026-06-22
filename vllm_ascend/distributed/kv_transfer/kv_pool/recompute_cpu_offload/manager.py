# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for recompute CPU offloading."""

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import SlidingWindowSpec, UniformTypeKVCacheSpecs
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.metadata import (
    RecomputeCPUOffloadMetadata,
    RecomputeCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class TransferMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]


@dataclass
class PreemptedRequestState:
    req_id: str
    cpu_block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    store_transfer_meta: TransferMeta
    store_event: int | None = None
    load_event: int | None = None
    load_transfer_meta: TransferMeta | None = None
    load_start_tokens: int = 0
    ready: bool = False
    finished: bool = False


class RecomputeCPUOffloadScheduler:
    """Preserve preempted requests' KV blocks in CPU memory.

    When offload prefix caching is enabled, full hashed blocks share CPU
    blocks. Otherwise every offloaded block is private to its request.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        enable_offload_prefix_caching: bool = True,
    ):
        assert kv_cache_config is not None
        self.vllm_config = vllm_config
        self.enable_offload_prefix_caching = enable_offload_prefix_caching
        self.cpu_kv_cache_config = self._derive_cpu_config(kv_cache_config, cpu_capacity_bytes)
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks
        self._group_is_sliding_window = self._get_group_is_sliding_window(kv_cache_config)
        self.enable_kv_cache_events = (
            vllm_config.kv_events_config is not None and vllm_config.kv_events_config.enable_kv_cache_events
        )

        logger.info(
            "RecomputeCPUOffloadScheduler: allocating %d CPU blocks (%.2f GB) for recompute offload, prefix caching=%s",
            self.num_cpu_blocks,
            cpu_capacity_bytes / (1024**3),
            self.enable_offload_prefix_caching,
        )

        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        assert dcp_world_size == 1 and pcp_world_size == 1
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            max_num_batched_tokens=(vllm_config.scheduler_config.max_num_batched_tokens),
            use_eagle=False,
            enable_caching=self.enable_offload_prefix_caching,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=vllm_config.cache_config.block_size,
        )
        self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool
        self._gpu_block_pool: BlockPool | None = None

        self._preempted_req_states: dict[str, PreemptedRequestState] = {}
        self._preempt_store_event_to_reqs: dict[int, list[str]] = {}
        self._preempt_store_event_to_blocks: dict[int, TransferMeta] = {}
        self._preempt_load_event_to_reqs: dict[int, list[str]] = {}

        # Hash blocks created before build_connector_meta() are shared by all
        # requests preempted in the same scheduling step.
        self._pending_hash_blocks: dict[BlockHashWithGroupId, KVCacheBlock] = {}

        self._load_event_counter = 0
        self._store_event_counter = 0
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = {}

    @staticmethod
    def _get_group_is_sliding_window(kv_cache_config: "KVCacheConfig") -> list[bool]:
        group_is_sliding_window: list[bool] = []
        for group in kv_cache_config.kv_cache_groups:
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
                group_is_sliding_window.append(
                    any(isinstance(spec, SlidingWindowSpec) for spec in group.kv_cache_spec.kv_cache_specs.values())
                )
            else:
                group_is_sliding_window.append(isinstance(group.kv_cache_spec, SlidingWindowSpec))
        return group_is_sliding_window

    @staticmethod
    def _derive_cpu_config(gpu_config: "KVCacheConfig", cpu_capacity_bytes: int) -> "KVCacheConfig":
        from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfigCls
        from vllm.v1.kv_cache_interface import KVCacheTensor

        assert gpu_config.kv_cache_tensors
        gpu_kv_cache_tensors = []
        for t in gpu_config.kv_cache_tensors:
            if t.shared_by:
                gpu_kv_cache_tensors.append(t)
        gpu_total_bytes = sum(t.size for t in gpu_kv_cache_tensors)
        num_gpu_blocks = gpu_config.num_blocks
        num_cpu_blocks = max(1, num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes)
        cpu_tensors = [
            KVCacheTensor(
                size=t.size // num_gpu_blocks * num_cpu_blocks,
                shared_by=list(t.shared_by),
            )
            for t in gpu_kv_cache_tensors
        ]
        return KVCacheConfigCls(
            num_blocks=num_cpu_blocks,
            kv_cache_tensors=cpu_tensors,
            kv_cache_groups=gpu_config.kv_cache_groups,
        )

    def _align_group_block_ids(
        self,
        group_idx: int,
        group_block_ids: list[int],
        logical_num_blocks: int,
    ) -> list[int]:
        if logical_num_blocks <= 0:
            return []
        aligned_group_block_ids = list(group_block_ids)
        if self._group_is_sliding_window[group_idx] and len(aligned_group_block_ids) < logical_num_blocks:
            aligned_group_block_ids = [0] * (
                logical_num_blocks - len(aligned_group_block_ids)
            ) + aligned_group_block_ids
        return aligned_group_block_ids[:logical_num_blocks]

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        self._gpu_block_pool = gpu_block_pool

    def has_preempted_request(self, req_id: str) -> bool:
        return req_id in self._preempted_req_states

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        state = self._preempted_req_states.get(request.request_id)
        if state is None:
            return 0, False
        if not state.ready:
            return None, False

        restorable_tokens = min(state.num_computed_tokens, request.num_tokens)
        hit_length = max(0, restorable_tokens - num_computed_tokens)
        if hit_length <= 0:
            self._cleanup_preempt_cache_request(request.request_id)
            return 0, False

        state.load_start_tokens = num_computed_tokens
        logger.debug(
            "Recompute offload cache hit for request %s: load_start=%d, load_tokens=%d, stored_tokens=%d.",
            request.request_id,
            num_computed_tokens,
            hit_length,
            state.num_computed_tokens,
        )
        return hit_length, True

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens <= 0:
            return
        prepared = self._prepare_preempt_load_after_alloc(
            request,
            blocks.get_block_ids(),
            num_external_tokens,
        )
        if not prepared:
            raise RuntimeError(
                "Failed to prepare recompute H2D load after KV block "
                f"allocation: req_id={request.request_id}, "
                f"num_external_tokens={num_external_tokens}"
            )

    def update_state_before_preempt(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
        num_computed_tokens: int,
    ) -> bool:
        if request.request_id in self._preempted_req_states:
            return True
        return self._create_preempt_state(
            request.request_id,
            block_ids,
            num_computed_tokens,
        )

    def _create_preempt_state(
        self,
        req_id: str,
        block_ids_by_group: tuple[list[int], ...],
        num_computed_tokens: int,
    ) -> bool:
        if num_computed_tokens <= 0 or self._gpu_block_pool is None:
            return False

        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups
        group_gpu_blocks: list[list[KVCacheBlock | None]] = []
        group_gpu_hashes: list[list[BlockHashWithGroupId | None]] = []
        missing_hashes: set[BlockHashWithGroupId] = set()
        num_unhashed = 0

        for g, group_gpu_ids in enumerate(block_ids_by_group):
            group_block_size = kv_cache_groups[g].kv_cache_spec.block_size
            logical_num_blocks = cdiv(num_computed_tokens, group_block_size)
            aligned_group_gpu_ids = self._align_group_block_ids(g, group_gpu_ids, logical_num_blocks)
            eviction_group_gpu_ids = self._align_group_block_ids(
                g,
                group_gpu_ids,
                max(logical_num_blocks, len(group_gpu_ids)),
            )
            gpu_blocks: list[KVCacheBlock | None] = []
            effective_hashes: list[BlockHashWithGroupId | None] = []

            for block_idx, block_id in enumerate(eviction_group_gpu_ids):
                if block_id <= 0:
                    continue
                gpu_block = self._gpu_block_pool.blocks[block_id]
                block_is_computed = (block_idx + 1) * group_block_size <= num_computed_tokens
                if not block_is_computed and gpu_block.block_hash is not None:
                    # allocate_slots() may assign a hash using tokens planned
                    # for this scheduling step. If the request is then
                    # preempted before forward, that block does not contain the
                    # hashed KV and must not remain in the GPU prefix cache.
                    self._gpu_block_pool._maybe_evict_cached_block(gpu_block)

            for block_idx, block_id in enumerate(aligned_group_gpu_ids):
                if block_id <= 0:
                    gpu_blocks.append(None)
                    effective_hashes.append(None)
                    continue

                gpu_block = self._gpu_block_pool.blocks[block_id]
                block_is_computed = (block_idx + 1) * group_block_size <= num_computed_tokens
                block_hash = gpu_block.block_hash if block_is_computed and self.enable_offload_prefix_caching else None
                gpu_blocks.append(gpu_block)
                effective_hashes.append(block_hash)
                if block_hash is None:
                    num_unhashed += 1
                elif (
                    self.cpu_block_pool.cached_block_hash_to_block.get_one_block(block_hash) is None
                    and block_hash not in self._pending_hash_blocks
                ):
                    missing_hashes.add(block_hash)
            group_gpu_blocks.append(gpu_blocks)
            group_gpu_hashes.append(effective_hashes)

        num_needed = num_unhashed + len(missing_hashes)
        if not any(any(gpu_block is not None for gpu_block in group) for group in group_gpu_blocks):
            return False
        if num_needed > self.cpu_block_pool.get_num_free_blocks():
            logger.warning(
                "Skip recompute offload for request %s: CPU cache has %d free blocks, but %d new blocks are required.",
                req_id,
                self.cpu_block_pool.get_num_free_blocks(),
                num_needed,
            )
            return False

        cpu_block_iter = iter(self.cpu_block_pool.get_new_blocks(num_needed))
        cpu_block_ids_by_group: list[list[int]] = []
        store_gpu_block_ids: list[int] = []
        store_cpu_block_ids: list[int] = []
        waiting_for_store = False

        for gpu_blocks, effective_hashes in zip(group_gpu_blocks, group_gpu_hashes):
            group_cpu_ids: list[int] = []
            for gpu_block, block_hash in zip(gpu_blocks, effective_hashes):
                if gpu_block is None:
                    group_cpu_ids.append(0)
                    continue

                cpu_block = None

                if block_hash is not None:
                    cpu_block = self.cpu_block_pool.cached_block_hash_to_block.get_one_block(block_hash)
                    if cpu_block is not None:
                        self.cpu_block_pool.touch([cpu_block])
                    else:
                        cpu_block = self._pending_hash_blocks.get(block_hash)
                        if cpu_block is not None:
                            self.cpu_block_pool.touch([cpu_block])
                            waiting_for_store = True
                        else:
                            cpu_block = next(cpu_block_iter)
                            cpu_block._block_hash = block_hash
                            self._pending_hash_blocks[block_hash] = cpu_block
                            store_gpu_block_ids.append(gpu_block.block_id)
                            store_cpu_block_ids.append(cpu_block.block_id)
                            waiting_for_store = True
                else:
                    cpu_block = next(cpu_block_iter)
                    store_gpu_block_ids.append(gpu_block.block_id)
                    store_cpu_block_ids.append(cpu_block.block_id)
                    waiting_for_store = True

                group_cpu_ids.append(cpu_block.block_id)
            cpu_block_ids_by_group.append(group_cpu_ids)

        store_transfer = TransferMeta(store_gpu_block_ids, store_cpu_block_ids)
        self._preempted_req_states[req_id] = PreemptedRequestState(
            req_id=req_id,
            cpu_block_ids=tuple(cpu_block_ids_by_group),
            num_computed_tokens=num_computed_tokens,
            store_transfer_meta=store_transfer,
            ready=not waiting_for_store,
        )
        logger.info(
            "Created recompute offload state for request %s: "
            "computed_tokens=%d, cpu_blocks=%d, store_blocks=%d, "
            "ready=%s.",
            req_id,
            num_computed_tokens,
            sum(len(ids) for ids in cpu_block_ids_by_group),
            len(store_cpu_block_ids),
            not waiting_for_store,
        )

        return True

    def _prepare_preempt_store_specs(
        self,
    ) -> tuple[list[int], list[int], list[str]]:
        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        req_ids: list[str] = []

        for req_id, state in self._preempted_req_states.items():
            if state.store_event is not None or state.ready:
                continue
            gpu_block_ids.extend(state.store_transfer_meta.gpu_block_ids)
            cpu_block_ids.extend(state.store_transfer_meta.cpu_block_ids)
            req_ids.append(req_id)
        return gpu_block_ids, cpu_block_ids, req_ids

    def _prepare_preempt_load_after_alloc(
        self,
        request: "Request",
        block_ids_by_group: tuple[list[int], ...],
        num_external_tokens: int,
    ) -> bool:
        state = self._preempted_req_states.get(request.request_id)
        if state is None or not state.ready:
            return False

        load_start_tokens = state.load_start_tokens
        load_end_tokens = min(
            load_start_tokens + num_external_tokens,
            state.num_computed_tokens,
        )
        if load_end_tokens <= load_start_tokens:
            return False

        if len(block_ids_by_group) != len(state.cpu_block_ids):
            raise RuntimeError(
                "Recompute H2D KV group count mismatch: "
                f"req_id={request.request_id}, "
                f"gpu_groups={len(block_ids_by_group)}, "
                f"cpu_groups={len(state.cpu_block_ids)}"
            )

        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        for g, group_cpu_ids in enumerate(state.cpu_block_ids):
            group_block_size = self.cpu_kv_cache_config.kv_cache_groups[g].kv_cache_spec.block_size
            start_block = load_start_tokens // group_block_size
            end_block = min(
                len(group_cpu_ids),
                len(
                    self._align_group_block_ids(
                        g,
                        block_ids_by_group[g],
                        max(
                            cdiv(load_end_tokens, group_block_size),
                            len(block_ids_by_group[g]),
                        ),
                    )
                ),
                cdiv(load_end_tokens, group_block_size),
            )
            if end_block == start_block:
                continue
            if end_block < start_block:
                raise RuntimeError(
                    "Recompute H2D produced an empty block range: "
                    f"req_id={request.request_id}, group={g}, "
                    f"start_block={start_block}, end_block={end_block}, "
                    f"gpu_blocks={len(block_ids_by_group[g])}, "
                    f"cpu_blocks={len(group_cpu_ids)}"
                )

            aligned_group_gpu_ids = self._align_group_block_ids(
                g,
                block_ids_by_group[g],
                end_block,
            )
            for block_idx in range(start_block, end_block):
                cpu_block_id = group_cpu_ids[block_idx]
                gpu_block_id = aligned_group_gpu_ids[block_idx]
                if cpu_block_id <= 0 or gpu_block_id <= 0:
                    continue
                cpu_block_ids.append(cpu_block_id)
                gpu_block_ids.append(gpu_block_id)

        if not cpu_block_ids or len(cpu_block_ids) != len(gpu_block_ids):
            raise RuntimeError(
                "Recompute H2D block mapping is incomplete: "
                f"req_id={request.request_id}, "
                f"gpu_blocks={len(gpu_block_ids)}, "
                f"cpu_blocks={len(cpu_block_ids)}"
            )

        assert self._gpu_block_pool is not None
        self._gpu_block_pool.touch([self._gpu_block_pool.blocks[block_id] for block_id in gpu_block_ids])
        state.load_transfer_meta = TransferMeta(gpu_block_ids, cpu_block_ids)
        logger.info(
            "Prepared recompute offload H2D load for request %s: tokens=[%d, %d), blocks=%d.",
            request.request_id,
            load_start_tokens,
            load_end_tokens,
            len(gpu_block_ids),
        )
        return True

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> RecomputeCPUOffloadMetadata:
        store_event = -1
        store_gpu, store_cpu, store_req_ids = self._prepare_preempt_store_specs()
        if store_gpu:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._preempt_store_event_to_blocks[store_event] = TransferMeta(store_gpu, store_cpu)
            self._preempt_store_event_to_reqs[store_event] = store_req_ids
            for req_id in store_req_ids:
                self._preempted_req_states[req_id].store_event = store_event
            self._pending_hash_blocks.clear()

        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_req_ids: list[str] = []
        for req_id, state in self._preempted_req_states.items():
            if state.load_transfer_meta is None or state.load_event is not None:
                continue
            load_gpu.extend(state.load_transfer_meta.gpu_block_ids)
            load_cpu.extend(state.load_transfer_meta.cpu_block_ids)
            load_req_ids.append(req_id)

        if load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            for req_id in load_req_ids:
                self._preempted_req_states[req_id].load_event = load_event
            self._preempt_load_event_to_reqs[load_event] = load_req_ids

        return RecomputeCPUOffloadMetadata(
            need_flush=bool(scheduler_output.preempted_req_ids),
            preempt_store_event=store_event,
            preempt_store_gpu_blocks=store_gpu,
            preempt_store_cpu_blocks=store_cpu,
            preempt_load_event=load_event,
            preempt_load_gpu_blocks=load_gpu,
            preempt_load_cpu_blocks=load_cpu,
            preempt_load_event_to_reqs=self._preempt_load_event_to_reqs,
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        for req_id in list(connector_output.finished_recving or []):
            if req_id in self._preempted_req_states:
                self._cleanup_preempt_load_request(req_id)

        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, RecomputeCPUOffloadWorkerMetadata):
            return
        for event_idx, count in meta.completed_store_events.items():
            total = self._store_event_pending_counts.get(event_idx, 0) + count
            if total >= self._expected_worker_count:
                self._store_event_pending_counts.pop(event_idx, None)
                self._process_preempt_store_event(event_idx)
            else:
                self._store_event_pending_counts[event_idx] = total

    def _process_preempt_store_event(self, event_idx: int) -> None:
        transfer = self._preempt_store_event_to_blocks.pop(event_idx)
        req_ids = self._preempt_store_event_to_reqs.pop(event_idx, [])

        for cpu_block_id in transfer.cpu_block_ids:
            cpu_block = self.cpu_block_pool.blocks[cpu_block_id]
            block_hash = cpu_block.block_hash
            if block_hash is None:
                continue
            cached_block = self.cpu_block_pool.cached_block_hash_to_block.get_one_block(block_hash)
            if cached_block is None:
                self.cpu_block_pool.cached_block_hash_to_block.insert(block_hash, cpu_block)
            elif cached_block.block_id != cpu_block.block_id:
                cpu_block.reset_hash()

        for req_id in req_ids:
            state = self._preempted_req_states.get(req_id)
            if state is not None:
                state.ready = True
                if state.finished:
                    self._cleanup_preempt_cache_request(req_id)

    def has_pending_transfers(self) -> bool:
        return bool(
            self._store_event_pending_counts
            or self._preempt_store_event_to_blocks
            or any(
                not state.ready or state.load_transfer_meta is not None for state in self._preempted_req_states.values()
            )
        )

    def reset_cache(self) -> bool:
        if self.has_pending_transfers():
            logger.warning(
                "Failed to reset recompute offload cache because transfers or request states are still pending."
            )
            return False
        for req_id in list(self._preempted_req_states):
            self._cleanup_preempt_cache_request(req_id)
        self._preempt_store_event_to_reqs.clear()
        self._preempt_store_event_to_blocks.clear()
        self._preempt_load_event_to_reqs.clear()
        self._pending_hash_blocks.clear()
        return self.cpu_block_pool.reset_prefix_cache()

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        state = self._preempted_req_states.get(request.request_id)
        if state is not None and state.load_event is None:
            if state.ready:
                self._cleanup_preempt_cache_request(request.request_id)
            else:
                state.finished = True
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids=[])

    def _cleanup_preempt_load_request(self, req_id: str) -> None:
        state = self._preempted_req_states.get(req_id)
        if state is None:
            return

        if state.load_event is not None:
            reqs = self._preempt_load_event_to_reqs.get(state.load_event)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._preempt_load_event_to_reqs.pop(state.load_event, None)

        if state.load_transfer_meta is not None:
            assert self._gpu_block_pool is not None
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[block_id] for block_id in state.load_transfer_meta.gpu_block_ids
            )
        self._cleanup_preempt_cache_request(req_id)

    def _cleanup_preempt_cache_request(self, req_id: str) -> None:
        state = self._preempted_req_states.pop(req_id, None)
        if state is None:
            return
        self.cpu_block_pool.free_blocks(
            self.cpu_block_pool.blocks[block_id]
            for group_cpu_ids in state.cpu_block_ids
            for block_id in group_cpu_ids
            if block_id > 0
        )

    def take_events(self) -> Iterable[KVCacheEvent]:
        return self.cpu_block_pool.take_events()
