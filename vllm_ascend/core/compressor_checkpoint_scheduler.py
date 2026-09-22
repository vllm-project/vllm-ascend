# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synchronous prefill checkpoint scheduling for private compressor rings."""

from dataclasses import dataclass, fields

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_ascend.core.compressor_checkpoint import TailCheckpointKey, TailRestorePlan, TailSavePlan
from vllm_ascend.core.compressor_checkpoint_coordinator import CompressorCheckpointCoordinator


@dataclass
class CompressorSchedulerOutput(SchedulerOutput):
    compressor_saves: tuple[TailSavePlan, ...] = ()
    compressor_restores: tuple[TailRestorePlan, ...] = ()


class CompressorCheckpointScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        coordinator = self.kv_cache_manager.coordinator
        if not isinstance(coordinator, CompressorCheckpointCoordinator):
            raise ValueError("CompressorCheckpointScheduler requires ring + prefix caching")
        if self.scheduler_config.async_scheduling or self.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Compressor checkpoints require synchronous scheduling and PP=1")
        self.checkpoint_coordinator = coordinator
        # Scope the allocation hook to this synchronous scheduler instance;
        # keep upstream admission, accounting and allocation unchanged.
        self.kv_cache_manager.allocate_slots = self._allocate_slots
        # Reuse the upstream pre-allocation boundary hook rather than copying
        # schedule(). This flag is only read at the two chunk-splitting sites.
        self.need_mamba_block_aligned_split = True
        # Cancellation must not recycle destination rings or KV still being
        # written by the current worker step. Reuse the upstream execution fence.
        self.defer_block_free = True

    def _allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0, new_computed_blocks=None, **kwargs):
        manager = self.kv_cache_manager
        coordinator = self.checkpoint_coordinator
        if kwargs.get("has_scheduled_reqs", True) and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            coordinator.admission_watermark = manager.watermark_blocks
        handle = coordinator._restore_handle(new_computed_blocks.blocks) if new_computed_blocks is not None else None
        try:
            allocated = KVCacheManager.allocate_slots(
                manager, request, num_new_tokens, num_new_computed_tokens, new_computed_blocks, **kwargs
            )
        finally:
            coordinator.admission_watermark = 0
        if allocated is None and handle is not None:
            # Drop only after upstream has rejected admission, when no stale
            # hit blocks can be allocated. The next schedule redoes lookup,
            # chunk splitting and token accounting as a miss/shorter hit.
            coordinator.checkpoints.discard(handle)
        return allocated

    def _mamba_block_aligned_split(
        self,
        request,
        num_new_tokens,
        num_new_local_computed_tokens=0,
        num_external_computed_tokens=0,
    ):
        assert num_external_computed_tokens == 0
        start = request.num_computed_tokens + num_new_local_computed_tokens
        if start >= request.num_prompt_tokens or not self.checkpoint_coordinator.checkpoints.can_reserve():
            return num_new_tokens
        alignment = self.checkpoint_coordinator.lcm_block_size
        boundary = (start // alignment + 1) * alignment
        end = min(start + num_new_tokens, request.num_prompt_tokens)
        for position in range(boundary, end + 1, alignment):
            key = TailCheckpointKey(request.block_hashes[position // self.hash_block_size - 1], position)
            if not self.checkpoint_coordinator.checkpoints.has_checkpoint(key):
                return position - start
        return num_new_tokens

    def schedule(self, throttle_prefills=False):
        if self.sched_step_seq != self.processed_step_seq:
            raise RuntimeError("Compressor checkpoint scheduling must wait for the previous worker step")
        output = super().schedule(throttle_prefills)
        coordinator = self.checkpoint_coordinator
        pool = coordinator.checkpoints
        restores = []
        saves = []
        for request_id, count in output.num_scheduled_tokens.items():
            request = self.requests[request_id]
            private_rings = {
                group_id: manager.req_to_blocks[request_id] for group_id, manager in coordinator.tail_managers.items()
            }
            if (handle := coordinator.pending_restores.pop(request_id, None)) is not None:
                restores.append(pool.restore_plan(handle, private_rings))
            # super().schedule has already advanced the optimistic endpoint.
            endpoint = request.num_computed_tokens
            if endpoint - count >= request.num_prompt_tokens or endpoint > request.num_prompt_tokens:
                continue
            if endpoint % coordinator.lcm_block_size:
                continue
            key = TailCheckpointKey(request.block_hashes[endpoint // self.hash_block_size - 1], endpoint)
            if (save := pool.reserve(key, private_rings)) is not None:
                saves.append(save)
        return CompressorSchedulerOutput(
            **{field.name: getattr(output, field.name) for field in fields(SchedulerOutput)},
            compressor_saves=tuple(saves),
            compressor_restores=tuple(restores),
        )

    def update_from_output(self, scheduler_output, model_runner_output):
        coordinator = self.checkpoint_coordinator
        # Worker completion includes all tail copies and a TP completion fence.
        # Cache ordinary groups only now, so they cannot be read before forward.
        for request_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(request_id)
            if request is not None and not request.is_finished():
                coordinator.cache_completed_blocks(request, request.num_computed_tokens)
        for save in scheduler_output.compressor_saves:
            coordinator.checkpoints.complete_save(save.handle)
        for restore in scheduler_output.compressor_restores:
            coordinator.checkpoints.release(restore.handle)
        return super().update_from_output(scheduler_output, model_runner_output)

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        if self.sched_step_seq != self.processed_step_seq:
            return False
        pool = self.checkpoint_coordinator.checkpoints
        blocks = self.kv_cache_manager.block_pool
        if not reset_running_requests and blocks.num_gpu_blocks - blocks.get_num_free_blocks() != pool.used_blocks + 1:
            return False
        pool.reset()
        return super().reset_prefix_cache(reset_running_requests, reset_connector)
