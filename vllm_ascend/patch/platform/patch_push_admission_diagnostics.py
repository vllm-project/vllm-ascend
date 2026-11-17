# SPDX-License-Identifier: Apache-2.0
"""Report the last KV block check when a Push producer cannot schedule its first chunk."""

from contextvars import ContextVar
from functools import wraps
from inspect import signature

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler

logger = init_logger(__name__)
_active_check: ContextVar[list[tuple[str, int]] | None] = ContextVar("push_admission_check", default=None)
_allocate_slots = KVCacheManager.allocate_slots
_count_blocks = KVCacheCoordinator.get_num_blocks_to_allocate
_count_signature = signature(_count_blocks)
_align_chunk = Scheduler._mamba_block_aligned_split


@wraps(_count_blocks)
def _count_with_diagnostics(self, *args, **kwargs):
    required = _count_blocks(self, *args, **kwargs)
    checks = _active_check.get()
    if checks is not None:
        arguments = _count_signature.bind(self, *args, **kwargs).arguments
        phase = "full_input" if arguments.get("apply_admission_cap") else "chunk"
        checks.append((phase, required))
    return required


@wraps(_allocate_slots)
def _allocate_with_diagnostics(self, request, num_new_tokens, *args, **kwargs):
    params = request.kv_transfer_params or {}
    if not params.get("do_remote_decode") or request.num_computed_tokens:
        return _allocate_slots(self, request, num_new_tokens, *args, **kwargs)

    checks: list[tuple[str, int]] = []
    token = _active_check.set(checks)
    try:
        result = _allocate_slots(self, request, num_new_tokens, *args, **kwargs)
    finally:
        _active_check.reset(token)
    if result is None:
        phase, required = checks[-1] if checks else ("before_block_count", None)
        if getattr(request, "_last_push_admission_failure", None) != phase:
            request._last_push_admission_failure = phase
            logger.warning(
                "Push P admission blocked request=%s phase=%s prompt_tokens=%d "
                "scheduled_tokens=%d counted_blocks=%s free_blocks=%d",
                request.request_id,
                phase,
                request.num_prompt_tokens,
                num_new_tokens,
                required,
                self.block_pool.get_num_free_blocks(),
            )
    return result


@wraps(_align_chunk)
def _align_with_diagnostics(self, request, num_new_tokens, *args, **kwargs):
    aligned = _align_chunk(self, request, num_new_tokens, *args, **kwargs)
    params = request.kv_transfer_params or {}
    if num_new_tokens > 0 and aligned == 0 and params.get("do_remote_decode"):
        if not getattr(request, "_push_alignment_logged", False):
            request._push_alignment_logged = True
            logger.warning(
                "Push P admission blocked request=%s phase=alignment prompt_tokens=%d scheduled_tokens=%d",
                request.request_id,
                request.num_prompt_tokens,
                num_new_tokens,
            )
    return aligned


Scheduler._mamba_block_aligned_split = _align_with_diagnostics
KVCacheCoordinator.get_num_blocks_to_allocate = _count_with_diagnostics
KVCacheManager.allocate_slots = _allocate_with_diagnostics
