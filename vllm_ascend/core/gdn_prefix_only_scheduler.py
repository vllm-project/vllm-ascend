# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position
from vllm.v1.request import Request

from vllm_ascend.core.single_type_kv_cache_manager import AscendGDNCheckpointManager, AscendGDNPrefixOnlyManager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

    from vllm_ascend.ascend_config import AscendConfig


class _GDNWarmPrefillScheduler(Scheduler):
    """Recognize final warm chunks that split solely at a tail checkpoint."""

    _manager_type: type[AscendGDNPrefixOnlyManager | AscendGDNCheckpointManager]
    _preserve_checkpoint = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        managers = self.kv_cache_manager.coordinator.single_type_managers
        self._gdn_managers = tuple(manager for manager in managers if isinstance(manager, self._manager_type))
        if not self._gdn_managers or any(
            isinstance(manager, MambaManager)
            and (
                not isinstance(manager, self._manager_type)
                or manager.has_prefill_checkpoint_blocks != self._preserve_checkpoint
            )
            for manager in managers
        ):
            self._gdn_managers = ()
            if not self._preserve_checkpoint:
                raise ValueError(
                    "GDN prefix_only requires GDN state in every Mamba cache group and internal exports disabled."
                )

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        aligned_tokens = super()._mamba_block_aligned_split(
            request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens
        )
        if (
            not self._gdn_managers
            or not self.mamba_partial_cache_hit
            or aligned_tokens == num_new_tokens
            or num_external_computed_tokens
        ):
            return aligned_tokens

        start = request.num_computed_tokens + num_new_local_computed_tokens
        end = start + num_new_tokens
        block_size = self.cache_config.block_size
        # Preserve all physical-page, token-budget, replay and junction stops,
        # including those that the final tail stop may otherwise hide.
        if (
            end != request.num_prompt_tokens
            or request.num_tokens != end
            or start // block_size != (end - 1) // block_size
            or start < request.shared_prefix_boundary < end
            or any(manager.block_size != block_size for manager in self._gdn_managers)
        ):
            return aligned_tokens

        tail = get_mamba_prefill_checkpoint_position(
            request.num_prompt_tokens,
            self.hash_block_size,
            self.use_eagle_block_drop,
            num_hash_lookahead_tokens=self.num_hash_lookahead_tokens,
        )
        if not start < tail < end or start + aligned_tokens != tail:
            return aligned_tokens

        # Waiting requests expose the hit from this lookup before allocation.
        # Continued chunks use the manager's record of the accepted hit.
        if not num_new_local_computed_tokens and not all(
            manager.has_local_prefix_hit(request.request_id) for manager in self._gdn_managers
        ):
            return aligned_tokens
        if self._preserve_checkpoint:
            if start % self.hash_block_size or tail - start < self.hash_block_size:
                return aligned_tokens
            for manager in self._gdn_managers:
                alignment = manager.kv_cache_spec.prefill_checkpoint_alignment
                if alignment is None or (tail - start) % alignment:
                    return aligned_tokens
            for manager in self._gdn_managers:
                manager.checkpoint_plans[request.request_id] = (tail, end)
        return num_new_tokens


class GDNPrefixOnlyScheduler(_GDNWarmPrefillScheduler):
    """Elide tail publication for workloads that reuse a shared prefix."""

    _manager_type = AscendGDNPrefixOnlyManager


class AsyncGDNPrefixOnlyScheduler(GDNPrefixOnlyScheduler, AsyncScheduler):
    """Run the same tail policy with upstream async bookkeeping."""


class GDNCheckpointScheduler(_GDNWarmPrefillScheduler):
    """Hold each dedicated state page until the executing step returns."""

    _manager_type = AscendGDNCheckpointManager
    _preserve_checkpoint = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._inflight_checkpoints: dict[int, list[tuple[Request, AscendGDNCheckpointManager, int, KVCacheBlock]]] = {}

    def schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
        for manager in self._gdn_managers:
            manager.checkpoint_plans.clear()
        output = super().schedule(throttle_prefills=throttle_prefills)
        pages = []
        checkpoints: dict[str, list[tuple[int, int, int]]] = {}
        for manager in self._gdn_managers:
            for req_id, (position, block) in manager.checkpoint_pages.items():
                assert req_id in output.num_scheduled_tokens
                pages.append((self.requests[req_id], manager, position, block))
                checkpoints.setdefault(req_id, []).append((manager.kv_cache_group_id, position, block.block_id))
            manager.checkpoint_pages.clear()
        if pages:
            # SchedulerOutput is a pickled dataclass in the local executors.
            # Attach the step metadata without altering the upstream schema.
            output.mamba_prefill_checkpoints = checkpoints
            self._inflight_checkpoints[id(output)] = pages
        return output

    def update_from_output(self, scheduler_output: "SchedulerOutput", model_runner_output: "ModelRunnerOutput"):
        pages = self._inflight_checkpoints.pop(id(scheduler_output), ())
        completed_ids = set(model_runner_output.req_ids) if pages else ()
        try:
            for scheduled_request, manager, position, block in pages:
                req_id = scheduled_request.request_id
                request = self.requests.get(req_id)
                if (
                    request is scheduled_request
                    and not request.is_finished()
                    and not request.num_stale_output_tokens
                    and req_id in completed_ids
                ):
                    manager.block_pool.cache_partial_block(
                        request, block, position, manager.kv_cache_group_id, manager.block_size
                    )
        finally:
            for _, manager, _, block in pages:
                manager.block_pool.free_blocks([block])
        return super().update_from_output(scheduler_output, model_runner_output)


class AsyncGDNCheckpointScheduler(GDNCheckpointScheduler, AsyncScheduler):
    """Preserve checkpoints with the built-in asynchronous scheduler."""


def configure_gdn_checkpoint_scheduler(vllm_config: "VllmConfig", ascend_config: "AscendConfig") -> None:
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    if (
        not vllm_config.use_v2_model_runner
        or not cache.enable_prefix_caching
        or cache.mamba_cache_mode != "align"
        or vllm_config.kv_transfer_config is not None
        or vllm_config.speculative_config is not None
        or parallel.pipeline_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.use_ubatching
        or ascend_config.scheduler_config.enable_balance_scheduling
    ):
        return
    scheduler = vllm_config.scheduler_config
    scheduler_cls = scheduler.get_scheduler_cls()
    # Ascend may wrap the scheduler module's default Scheduler at import time.
    if scheduler.scheduler_cls is None or scheduler_cls in (Scheduler, AsyncScheduler):
        scheduler.scheduler_cls = AsyncGDNCheckpointScheduler if scheduler.async_scheduling else GDNCheckpointScheduler


def configure_gdn_prefix_only_scheduler(vllm_config: "VllmConfig", ascend_config: "AscendConfig") -> None:
    cache_config = vllm_config.cache_config
    if (
        not vllm_config.use_v2_model_runner
        or not cache_config.enable_prefix_caching
        or cache_config.mamba_cache_mode != "align"
        or vllm_config.kv_transfer_config is not None
        or vllm_config.speculative_config is not None
        or ascend_config.gdn_internal_prefill_checkpoints
    ):
        raise ValueError(
            "GDN prefix_only requires Model Runner V2, align-mode local prefix caching, "
            "and disabled speculative decoding and internal prefill checkpoints."
        )
    if ascend_config.scheduler_config.enable_balance_scheduling:
        raise ValueError("GDN prefix_only and balance scheduling are mutually exclusive.")

    scheduler_config = vllm_config.scheduler_config
    scheduler_cls = scheduler_config.get_scheduler_cls()
    use_async = bool(scheduler_config.async_scheduling)
    # A scheduler composed through inheritance can supply the same policy.
    if issubclass(scheduler_cls, GDNPrefixOnlyScheduler) and issubclass(scheduler_cls, AsyncScheduler) == use_async:
        return
    if scheduler_config.scheduler_cls is not None and scheduler_cls not in (Scheduler, AsyncScheduler):
        raise ValueError("GDN prefix_only requires the default scheduler or a compatible GDN prefix-only subclass.")
    scheduler_config.scheduler_cls = AsyncGDNPrefixOnlyScheduler if use_async else GDNPrefixOnlyScheduler
