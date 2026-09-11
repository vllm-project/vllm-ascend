# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from functools import wraps
from inspect import signature

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs


def _supports_reservation(spec) -> bool:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return all(_supports_reservation(child) for child in spec.kv_cache_specs.values())
    # Other cache managers can recycle blocks or allocate state using the
    # computed-token boundary rather than the lookahead-token boundary.
    return type(spec) is FullAttentionSpec


def reserve_generation_slots(manager, num_spec_tokens: int = 0) -> None:
    """Reserve real blocks through the existing lookahead allocation API.

    Computed tokens and scheduled work are unchanged. Full-attention blocks
    remain owned by the request until the normal finish/cancel path frees them.
    """
    if manager.enable_caching or any(
        not _supports_reservation(group.kv_cache_spec) for group in manager.kv_cache_config.kv_cache_groups
    ):
        raise ValueError("Batch-invariant scheduling requires full-attention KV caches with prefix caching disabled.")

    allocate = manager.allocate_slots
    allocate_signature = signature(allocate)

    @wraps(allocate)
    def allocate_slots(*args, **kwargs):
        bound = allocate_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        request = arguments["request"]
        total_computed = (
            request.num_computed_tokens
            + arguments["num_new_computed_tokens"]
            + arguments["num_external_computed_tokens"]
        )
        target = min(request.num_prompt_tokens + request.max_tokens + num_spec_tokens, manager.max_model_len)
        main_tokens = total_computed + arguments["num_new_tokens"]
        arguments["num_lookahead_tokens"] = max(arguments["num_lookahead_tokens"], target - main_tokens)

        if request.num_computed_tokens == 0:
            required = manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=target,
                new_computed_blocks=manager.empty_kv_cache_blocks.blocks,
                num_encoder_tokens=arguments["num_encoder_tokens"],
                total_computed_tokens=0,
                num_local_computed_tokens=0,
                num_tokens_main_model=target,
            )
            # The block pool always keeps one null block. Do not leave a
            # request that cannot fit even on an idle engine waiting forever.
            if required > manager.block_pool.num_gpu_blocks - 1:
                raise ValueError(
                    "Request exceeds total KV capacity with batch-invariant generation reservation; "
                    "reduce max_tokens/max_model_len or increase KV cache memory."
                )
        return allocate(*bound.args, **bound.kwargs)

    manager.allocate_slots = allocate_slots


class _BatchInvariantSchedulerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        reserve_generation_slots(self.kv_cache_manager, max(self.num_spec_tokens, self.num_lookahead_tokens))

    def _preempt_request(self, *args, **kwargs):
        # Fail closed if an explicit eviction path bypasses capacity admission.
        raise RuntimeError("Request preemption is not supported with VLLM_BATCH_INVARIANT=1.")

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        if reset_running_requests and self.running:
            return False
        return super().reset_prefix_cache(reset_running_requests, reset_connector)


class BatchInvariantScheduler(_BatchInvariantSchedulerMixin, Scheduler):
    """Full-prefill scheduling with generation KV reservation and no eviction."""


class BatchInvariantAsyncScheduler(_BatchInvariantSchedulerMixin, AsyncScheduler):
    """Async equivalent of BatchInvariantScheduler."""
