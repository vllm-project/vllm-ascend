"""DSA sparse-cache scheduler hooks.

The DSA sparse-offload scheduler behavior is Ascend-specific, but it needs to
participate in vLLM's core scheduling decisions: slot allocation, prefill/decode
phase separation, and scheduler-output metadata.  Keep those adaptations here
instead of carrying a forked ``vllm.v1.core.sched.scheduler`` source file.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps
from typing import Any

import vllm.v1.core.sched.output as sched_output
import vllm.v1.core.sched.scheduler as scheduler_mod
from vllm.logger import init_logger
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

import vllm_ascend.patch.dsa_sparse.patch_scheduler_output  # noqa: F401
from vllm_ascend.dsa_sparse.dsa_block_hash_delta import (
    build_context_full_block_hash_delta,
)
from vllm_ascend.dsa_sparse.dsa_model_support import (
    is_dsa_sparse_runtime_enabled,
)
from vllm_ascend.dsa_sparse.dsa_sparse import DSASparseV1
from vllm_ascend.dsa_sparse.dsa_types import (
    INVALID_SLOT,
    DSASparseRole,
    ReqStage,
    max_safe_mtp_drafts_before_block_boundary,
)

logger = init_logger(__name__)


def _is_dsa_prefill_barrier_request(request: Request) -> bool:
    return (request.num_output_tokens == 0
            and request.num_computed_tokens < request.num_prompt_tokens)


def _has_dsa_running_prefill_barrier_work(
    running: Iterable[Request],
) -> bool:
    return any(_is_dsa_prefill_barrier_request(request) for request in running)


def _is_dsa_enabled(scheduler: Scheduler) -> bool:
    return getattr(scheduler, "dsa_scheduler_mgr", None) is not None


def _is_dsa_decode_request(self: Scheduler, request: Request) -> bool:
    return request.num_output_tokens > 0


def _has_dsa_prefill_work(self: Scheduler) -> bool:
    if not _is_dsa_enabled(self):
        return False
    return _has_dsa_running_prefill_barrier_work(self.running)


def _has_schedulable_dsa_waiting_prefill(
    self: Scheduler,
    token_budget: int,
) -> bool:
    """Return whether a waiting prefill can make progress this step.

    DSA avoids prefill/decode mixed forwards.  A waiting prefill should block
    running decode only when it can actually be admitted; otherwise continuous
    batching can stall with decode-ready requests in ``running`` and an
    unschedulable prefill at the head of ``waiting``.
    """
    if (not _is_dsa_enabled(self) or token_budget <= 0
            or len(self.running) >= self.max_num_running_reqs):
        return False

    request_queue = self._select_waiting_queue_for_scheduling()
    if request_queue is None:
        return False
    request = request_queue.peek_request()
    if not _is_dsa_prefill_barrier_request(request):
        return False
    if self._is_blocked_waiting_status(request.status):
        return False

    num_new_tokens = request.num_tokens - request.num_computed_tokens
    threshold = self.scheduler_config.long_prefill_token_threshold
    if 0 < threshold < num_new_tokens:
        num_new_tokens = threshold
    if (not self.scheduler_config.enable_chunked_prefill
            and num_new_tokens > token_budget):
        return False
    if min(num_new_tokens, token_budget) <= 0:
        return False
    return not (self.scheduler_reserve_full_isl
                and not self.kv_cache_manager.can_fit_full_sequence(request))


def _estimate_dsa_resident_slots(
    self: Scheduler,
    request: Request,
    num_new_tokens: int,
) -> int:
    if not _is_dsa_enabled(self):
        request.dsa_req_stage = (ReqStage.PREFILL
                                 if request.num_output_tokens == 0 else
                                 ReqStage.DENSE_DECODE)
        request.dsa_next_req_stage = request.dsa_req_stage
        request.dsa_resident_valid_seq_len = INVALID_SLOT
        request.dsa_sparse_budget_tokens = 0
        return INVALID_SLOT
    return self.dsa_scheduler_mgr.plan_decode_resident_slots(
        request,
        num_new_tokens=num_new_tokens,
    )


def _trim_dsa_mtp_drafts_at_block_boundaries(
    self: Scheduler,
) -> None:
    """Do not let an unverified MTP token complete an offloaded MLA block.

    A block is dumped to DRAM after attention.  Completing that block with an
    unverified draft would make rejection require a DRAM rollback.  Keep the
    guaranteed model token, but defer drafts at the boundary to the next
    scheduler step. vLLM already treats draft IDs as per-step proposals.
    """
    if not _is_dsa_enabled(self):
        return
    block_size = int(self.block_size)
    activation = int(
        self.dsa_scheduler_mgr._sparse_activation_tokens
    )
    for request in self.running:
        if (
            request.num_output_tokens <= 0
            or not request.spec_token_ids
            or request.has_encoder_inputs
        ):
            continue
        guaranteed_tokens = max(
            0,
            int(request.num_tokens) - int(request.num_computed_tokens),
        )
        if guaranteed_tokens <= 0:
            continue
        sparse_context = (
            ReqStage.coerce(request.dsa_req_stage).is_sparse_decode
            or int(request.num_computed_tokens) + guaranteed_tokens
            > activation
        )
        if not sparse_context:
            continue

        allowed_drafts = max_safe_mtp_drafts_before_block_boundary(
            num_computed_tokens=request.num_computed_tokens,
            guaranteed_tokens=guaranteed_tokens,
            block_size=block_size,
        )
        if len(request.spec_token_ids) > allowed_drafts:
            request.spec_token_ids = request.spec_token_ids[
                :allowed_drafts
            ]


def _record_dsa_request_state(
    request: Request,
    req_dsa_stage: dict[str, int],
    req_dsa_resident_valid_seq_len: dict[str, int],
    req_dsa_sparse_budget_tokens: dict[str, int],
    req_dsa_target_resident_budget_tokens: dict[str, int],
) -> None:
    req_dsa_stage[request.request_id] = int(
        ReqStage.coerce(getattr(request, "dsa_req_stage", ReqStage.PREFILL)))
    req_dsa_resident_valid_seq_len[request.request_id] = (
        request.dsa_resident_valid_seq_len)
    req_dsa_sparse_budget_tokens[request.request_id] = (
        request.dsa_sparse_budget_tokens)
    req_dsa_target_resident_budget_tokens[request.request_id] = (
        request.dsa_target_resident_budget_tokens)


def _check_dsa_block_ids_for_overflow(
    self: Scheduler,
    source: str,
    request: Request,
    block_ids: tuple[list[int], ...] | None,
) -> tuple[list[int], ...] | None:
    if not _is_dsa_enabled(self) or block_ids is None:
        return block_ids
    capacity = (self.max_model_len + self.block_size - 1) // self.block_size
    lengths = [len(group_block_ids) for group_block_ids in block_ids]
    if all(length <= capacity for length in lengths):
        return block_ids
    logger.warning(
        "[DSA scheduler block ids overflow] source=%s req_id=%s "
        "lengths=%s capacity=%s group_block_sizes=%s "
        "num_tokens=%s num_computed=%s prompt_tokens=%s output_tokens=%s "
        "dsa_resident_valid_seq_len=%s samples=%s",
        source,
        request.request_id,
        lengths,
        capacity,
        [g.kv_cache_spec.block_size for g in self.kv_cache_config.kv_cache_groups],
        request.num_tokens,
        request.num_computed_tokens,
        request.num_prompt_tokens,
        request.num_output_tokens,
        getattr(request, "dsa_resident_valid_seq_len", INVALID_SLOT),
        [{
            "head": group_block_ids[:8],
            "tail": group_block_ids[-8:],
        } for group_block_ids in block_ids],
    )
    return block_ids


def _has_ready_dsa_decode_work(self: Scheduler) -> bool:
    for request in self.running:
        if not self._is_dsa_decode_request(request):
            continue
        if (request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 -
                request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens):
            continue
        num_new_tokens = (request.num_tokens_with_spec +
                          request.num_output_placeholders -
                          request.num_computed_tokens)
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        num_new_tokens = min(num_new_tokens,
                             self.max_model_len - 1 -
                             request.num_computed_tokens)
        if num_new_tokens > 0:
            return True
    return False


def _install_dsa_allocate_slots_wrapper(self: Scheduler) -> None:
    kv_cache_manager = self.kv_cache_manager
    if getattr(kv_cache_manager, "_dsa_sparse_allocate_slots_patched", False):
        return

    original_allocate_slots = kv_cache_manager.allocate_slots

    @wraps(original_allocate_slots)
    def _dsa_allocate_slots(request: Request, num_new_tokens: int, *args: Any,
                            **kwargs: Any):
        if (not _is_dsa_enabled(self)
                or getattr(kv_cache_manager, "_dsa_sparse_inside_alloc", False)):
            return original_allocate_slots(request, num_new_tokens, *args, **kwargs)

        resident_valid_seq_len = self._estimate_dsa_resident_slots(
            request,
            num_new_tokens,
        )
        kv_cache_manager._dsa_sparse_inside_alloc = True
        try:
            return self.dsa_scheduler_mgr.dsa_alloc_slots_wrap(
                kv_cache_manager,
                request,
                resident_valid_seq_len,
                num_new_tokens,
                *args,
                **kwargs,
            )
        finally:
            kv_cache_manager._dsa_sparse_inside_alloc = False

    kv_cache_manager.allocate_slots = _dsa_allocate_slots
    kv_cache_manager._dsa_sparse_allocate_slots_patched = True


def _withhold_decode_running_for_prefill(self: Scheduler):
    withheld: list[tuple[int, Request]] = []
    kept: list[Request] = []
    for index, request in enumerate(self.running):
        if self._is_dsa_decode_request(request):
            withheld.append((index, request))
        else:
            kept.append(request)

    if not withheld:
        return None

    old_max_num_running_reqs = self.max_num_running_reqs
    self.running = kept
    self.max_num_running_reqs = max(0, old_max_num_running_reqs - len(withheld))

    def restore() -> None:
        restored = list(self.running)
        for index, request in withheld:
            if (request in restored or request.request_id not in self.requests
                    or request.status != RequestStatus.RUNNING):
                continue
            restored.insert(min(index, len(restored)), request)
        self.running = restored
        self.max_num_running_reqs = old_max_num_running_reqs

    return restore


def _withhold_waiting_for_decode(self: Scheduler):
    old_waiting = self.waiting
    old_skipped_waiting = self.skipped_waiting
    self.waiting = create_request_queue(self.policy)
    self.skipped_waiting = create_request_queue(self.policy)

    def restore() -> None:
        # Running-request preemption can enqueue into the temporary queues
        # even though waiting admission is disabled for this decode-only
        # step. Preserve those requests and put them ahead of the older
        # waiting work. Repeated prepend in reverse order retains FCFS order;
        # priority queues re-heapify through prepend_request().
        temporary_waiting = list(self.waiting)
        temporary_skipped_waiting = list(self.skipped_waiting)
        for request in reversed(temporary_waiting):
            old_waiting.prepend_request(request)
        for request in reversed(temporary_skipped_waiting):
            old_skipped_waiting.prepend_request(request)
        self.waiting = old_waiting
        self.skipped_waiting = old_skipped_waiting

    return restore


def _populate_dsa_scheduler_output(self: Scheduler, scheduler_output) -> None:
    if not _is_dsa_enabled(self):
        return

    req_dsa_stage: dict[str, int] = {}
    req_dsa_resident_valid_seq_len: dict[str, int] = {}
    req_dsa_sparse_budget_tokens: dict[str, int] = {}
    req_dsa_target_resident_budget_tokens: dict[str, int] = {}

    scheduled_req_ids = set(scheduler_output.num_scheduled_tokens)
    for req_id in scheduled_req_ids:
        request = self.requests.get(req_id)
        if request is None:
            continue
        _record_dsa_request_state(request, req_dsa_stage,
                                  req_dsa_resident_valid_seq_len,
                                  req_dsa_sparse_budget_tokens,
                                  req_dsa_target_resident_budget_tokens)

    scheduler_output.req_dsa_stage = req_dsa_stage
    scheduler_output.req_dsa_resident_valid_seq_len = (
        req_dsa_resident_valid_seq_len)
    scheduler_output.req_dsa_sparse_budget_tokens = req_dsa_sparse_budget_tokens
    scheduler_output.req_dsa_target_resident_budget_tokens = (
        req_dsa_target_resident_budget_tokens)

    for new_req_data in scheduler_output.scheduled_new_reqs:
        request = self.requests.get(new_req_data.req_id)
        if request is not None:
            # Patched NewRequestData.from_request() already materialized the
            # full snapshot. Keep this fallback for outputs constructed by a
            # different/older factory, without copying a long prompt ledger a
            # second time on the normal admission path.
            if new_req_data.block_hashes is None:
                new_req_data.block_hashes = list(request.block_hashes)
            self.dsa_sent_block_hash_counts[request.request_id] = len(
                request.block_hashes)

    cached_reqs_data = scheduler_output.scheduled_cached_reqs
    cached_reqs_data.block_hash_starts = []
    cached_reqs_data.block_hashes = []
    for req_id, block_ids in zip(cached_reqs_data.req_ids,
                                 cached_reqs_data.new_block_ids):
        request = self.requests.get(req_id)
        if request is None:
            cached_reqs_data.block_hash_starts.append(0)
            cached_reqs_data.block_hashes.append([])
            continue
        start, delta = build_context_full_block_hash_delta(
            request.block_hashes,
            self.dsa_sent_block_hash_counts.get(req_id, 0),
        )
        cached_reqs_data.block_hash_starts.append(start)
        cached_reqs_data.block_hashes.append(delta)
        self.dsa_sent_block_hash_counts[req_id] = len(request.block_hashes)
        self._check_dsa_block_ids_for_overflow("cached", request, block_ids)

    for new_req_data in scheduler_output.scheduled_new_reqs:
        request = self.requests.get(new_req_data.req_id)
        if request is not None:
            self._check_dsa_block_ids_for_overflow("new", request,
                                                   new_req_data.block_ids)


if not getattr(Scheduler, "_dsa_sparse_scheduler_patched", False):
    # Keep scheduler module globals aligned with the patched dataclasses even if
    # vLLM imported ``scheduler`` before patch_scheduler_output ran.
    scheduler_mod.NewRequestData = sched_output.NewRequestData
    scheduler_mod.CachedRequestData = sched_output.CachedRequestData
    scheduler_mod.SchedulerOutput = sched_output.SchedulerOutput
    # patch_balance_schedule imports NewRequestData by value at module load time,
    # before DSA patches are installed.  Re-bind its reference so the balanced
    # scheduler also produces DSA-extended NewRequestData with block_hashes.
    # Other scheduler variants (dynamic_batch, recompute, profiling_chunk)
    # have the same by-value import and must be re-bound too.
    import vllm_ascend.patch.platform.patch_balance_schedule as balance_mod
    balance_mod.NewRequestData = sched_output.NewRequestData
    balance_mod.SchedulerOutput = sched_output.SchedulerOutput
    import vllm_ascend.core.scheduler_dynamic_batch as dyn_batch_mod
    dyn_batch_mod.NewRequestData = sched_output.NewRequestData
    dyn_batch_mod.SchedulerOutput = sched_output.SchedulerOutput
    import vllm_ascend.core.recompute_scheduler as recompute_mod
    recompute_mod.NewRequestData = sched_output.NewRequestData
    recompute_mod.SchedulerOutput = sched_output.SchedulerOutput
    import vllm_ascend.core.scheduler_profiling_chunk as profiling_mod
    profiling_mod.NewRequestData = sched_output.NewRequestData
    profiling_mod.SchedulerOutput = sched_output.SchedulerOutput

    _original_init = Scheduler.__init__
    _original_schedule = Scheduler.schedule
    _original_preempt_request = Scheduler._preempt_request
    _original_update_from_output = Scheduler.update_from_output
    _original_add_request = Scheduler.add_request
    _original_free_request = Scheduler._free_request

    @wraps(_original_init)
    def _dsa_sparse_scheduler_init(self: Scheduler, *args: Any,
                                   **kwargs: Any) -> None:
        _original_init(self, *args, **kwargs)

        self.dsa_scheduler_mgr = None
        if is_dsa_sparse_runtime_enabled(self.vllm_config):
            self.dsa_scheduler_mgr = DSASparseV1(self.vllm_config,
                                                DSASparseRole.SCHEDULER)
            _install_dsa_allocate_slots_wrapper(self)
        self.dsa_prefill_full_released_req_ids: set[str] = set()
        # Cursor into each Request's append-only full-block hash ledger. New
        # requests send a complete snapshot; cached steps only send its suffix.
        self.dsa_sent_block_hash_counts: dict[str, int] = {}

    @wraps(_original_schedule)
    def _dsa_sparse_schedule(self: Scheduler):
        if not _is_dsa_enabled(self):
            return _original_schedule(self)

        _install_dsa_allocate_slots_wrapper(self)
        _trim_dsa_mtp_drafts_at_block_boundaries(self)
        restore = None
        token_budget = (0 if self._pause_state == PauseState.PAUSED_ALL else
                        self.max_num_scheduled_tokens)
        dsa_phase_barrier_active = (
            self._has_dsa_prefill_work()
            or self._has_schedulable_dsa_waiting_prefill(token_budget))
        if dsa_phase_barrier_active:
            restore = _withhold_decode_running_for_prefill(self)
        elif _has_ready_dsa_decode_work(self):
            restore = _withhold_waiting_for_decode(self)

        try:
            scheduler_output = _original_schedule(self)
        finally:
            if restore is not None:
                restore()

        if self.running:
            any_request_id = self.running[0].request_id
            scheduler_output.num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request_id))
        _populate_dsa_scheduler_output(self, scheduler_output)
        return scheduler_output

    @wraps(_original_preempt_request)
    def _dsa_sparse_preempt_request(self: Scheduler, request: Request,
                                    timestamp: float) -> None:
        _original_preempt_request(self, request, timestamp)
        if not _is_dsa_enabled(self):
            return
        self.dsa_prefill_full_released_req_ids.discard(request.request_id)
        request.dsa_req_stage = ReqStage.PREFILL
        request.dsa_next_req_stage = ReqStage.PREFILL
        request.dsa_resident_valid_seq_len = INVALID_SLOT
        request.dsa_sparse_budget_tokens = 0

    @wraps(_original_update_from_output)
    def _dsa_sparse_update_from_output(self: Scheduler, scheduler_output,
                                       model_runner_output):
        outputs = _original_update_from_output(self, scheduler_output,
                                               model_runner_output)
        if _is_dsa_enabled(self):
            for req_id in scheduler_output.num_scheduled_tokens:
                request = self.requests.get(req_id)
                if (request is not None and not request.is_finished()
                        and request.num_output_tokens == 0
                        and request.num_computed_tokens
                        >= request.num_prompt_tokens):
                    self._maybe_release_dsa_prefill_full_cache(request)
        return outputs

    @wraps(_original_add_request)
    def _dsa_sparse_add_request(self: Scheduler, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        _original_add_request(self, request)
        if existing is None:
            self.dsa_sent_block_hash_counts[request.request_id] = 0
            self.dsa_prefill_full_released_req_ids.discard(request.request_id)
            if _is_dsa_enabled(self):
                request.dsa_target_resident_budget_tokens = (
                    self.dsa_scheduler_mgr.request_begin(
                        request.request_id,
                        request.prompt_token_ids,
                    ))

    @wraps(_original_free_request)
    def _dsa_sparse_free_request(self: Scheduler,
                                 request: Request,
                                 delay_free_blocks: bool = False):
        if _is_dsa_enabled(self):
            self.dsa_scheduler_mgr.request_finished_in_scheduler(
                request.request_id)
            self.dsa_prefill_full_released_req_ids.discard(request.request_id)
            self.dsa_sent_block_hash_counts.pop(request.request_id, None)
        return _original_free_request(self, request, delay_free_blocks)

    def _maybe_release_dsa_prefill_full_cache(self: Scheduler,
                                              request: Request) -> None:
        if not _is_dsa_enabled(self):
            return
        request_id = request.request_id
        if request_id in self.dsa_prefill_full_released_req_ids:
            return
        if not self.dsa_scheduler_mgr.should_release_full_cache_after_prefill(
                request):
            return
        released = self.dsa_scheduler_mgr.release_prefill_full_cache_except_tail(
            self.kv_cache_manager, request)
        if released:
            self.dsa_prefill_full_released_req_ids.add(request_id)

    Scheduler._is_dsa_decode_request = _is_dsa_decode_request
    Scheduler._has_dsa_prefill_work = _has_dsa_prefill_work
    Scheduler._has_schedulable_dsa_waiting_prefill = (
        _has_schedulable_dsa_waiting_prefill)
    Scheduler._estimate_dsa_resident_slots = _estimate_dsa_resident_slots
    Scheduler._check_dsa_block_ids_for_overflow = (
        _check_dsa_block_ids_for_overflow)
    Scheduler.__init__ = _dsa_sparse_scheduler_init
    Scheduler.schedule = _dsa_sparse_schedule
    Scheduler._preempt_request = _dsa_sparse_preempt_request
    Scheduler.update_from_output = _dsa_sparse_update_from_output
    Scheduler.add_request = _dsa_sparse_add_request
    Scheduler._free_request = _dsa_sparse_free_request
    Scheduler._maybe_release_dsa_prefill_full_cache = (
        _maybe_release_dsa_prefill_full_cache)
    Scheduler._dsa_sparse_scheduler_patched = True
