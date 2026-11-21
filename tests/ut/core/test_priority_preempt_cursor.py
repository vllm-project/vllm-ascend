# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PRIORITY preemption must not skip the rest of the running queue.

The Ascend schedulers are copies of upstream vLLM's ``Scheduler.schedule()``.
When a running request cannot allocate its KV blocks under
``SchedulingPolicy.PRIORITY``, the lowest-priority request is preempted and
removed from ``self.running``.

``self.running.remove(victim)`` shifts every request behind the victim one slot
to the left, so the ``req_index`` loop cursor has to move back with it. The
copies only corrected the cursor when the victim had *already been scheduled*
in the current step (``victim in scheduled_running_reqs``).

A victim that was skipped earlier in the same step -- for example a request
with ``num_new_tokens == 0`` (deferred prefill, a decode-staggered request, or a
request that hit its max length) -- is **not** in ``scheduled_running_reqs``, so
the cursor was never moved back and every running request behind the victim was
silently skipped for the rest of that step: the requests keep their KV blocks but
produce no token, and the token budget of the step is silently under-used.

Upstream vLLM already carries the fix; this test pins the Ascend copies to it.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.models import ModelRegistry
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import (
    create_model_runner_output,
    create_request,
    create_vllm_config,
)
from vllm_ascend.core.dyntra_lb_scheduler import DyntraLBScheduler
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler

# The Ascend copies of upstream's ``Scheduler.schedule()`` that preempt running
# requests. All of them must remove the preemption victim by index and move the
# loop cursor back with it.
SCHEDULER_CLASSES = (DyntraLBScheduler, BalanceScheduler)


def _fake_model_info():
    return SimpleNamespace(
        architecture="OPTForCausalLM",
        is_text_generation_model=True,
        is_pooling_model=False,
        attn_type="decoder",
        default_seq_pooling_type=None,
        default_tok_pooling_type=None,
        score_type=None,
        supports_multimodal=False,
        supports_multimodal_raw_input_only=False,
        requires_raw_input_tokens=False,
        supports_multimodal_encoder_tp_data=False,
        supports_pp=True,
        has_inner_state=False,
        is_attention_free=False,
        is_hybrid=False,
        has_noops=False,
        supports_mamba_prefix_caching=False,
        supports_replayssm=False,
        supports_transcription=False,
        supports_transcription_only=False,
        supported_video_pruning_methods=(),
    )


def _make_config(
    scheduler_cls,
    *,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 2048,
    block_size: int = 16,
    num_blocks: int = 1000,
):
    """Build a config + scheduler for one of the Ascend scheduler copies."""
    with patch.object(
        ModelRegistry,
        "inspect_model_cls",
        return_value=(_fake_model_info(), "OPTForCausalLM"),
    ):
        vllm_config = create_vllm_config(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size,
        )
    vllm_config.kv_transfer_config = None
    vllm_config.scheduler_config.policy = "priority"
    # BalanceScheduler only runs its own (copied) schedule() when balance
    # scheduling is enabled; otherwise it delegates to the upstream scheduler.
    if scheduler_cls is BalanceScheduler:
        vllm_config.additional_config = {"enable_balance_scheduling": True}

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks

    scheduler = scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        block_size=block_size,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    # These fixtures assert legacy V1 scheduler output structures.
    scheduler.use_v2_model_runner = False
    return scheduler


def _make_running(scheduler_cls, block_size: int = 16):
    """Build a scheduler with three running requests ``[victim, trigger, tail]``."""
    scheduler = _make_config(scheduler_cls, block_size=block_size)
    requests = [
        create_request(
            request_id=i,
            num_tokens=4 * block_size,
            max_tokens=8,
            block_size=block_size,
        )
        for i in range(1, 4)
    ]
    for request in requests:
        scheduler.add_request(request)

    first = scheduler.schedule()
    scheduler.update_from_output(first, create_model_runner_output(requests))
    assert len(scheduler.running) == 3

    victim, trigger, tail = requests
    # The preemption victim is the running request with the numerically largest
    # priority value.
    victim.priority, trigger.priority, tail.priority = 9, 0, 1
    assert scheduler.running == [victim, trigger, tail]
    return scheduler, victim, trigger, tail


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_priority_preempt_keeps_running_queue_cursor(scheduler_cls):
    """A preemption must not skip the running requests behind the victim.

    Setup: three running requests ``[victim, trigger, tail]``. The victim is
    skipped in this step (``num_new_tokens == 0``), so it is *not* part of
    ``scheduled_running_reqs`` when the trigger cannot allocate its blocks and
    the victim gets preempted.

    Pre-fix, ``self.running.remove(victim)`` shifted ``[victim, trigger, tail]``
    to ``[trigger, tail]`` while ``req_index`` still pointed at index 1, so
    ``trigger`` (now at index 0) was skipped and ``tail`` was processed in its
    place. With the fix the cursor moves back with the victim and both
    ``trigger`` and ``tail`` are scheduled.
    """
    scheduler, victim, trigger, tail = _make_running(scheduler_cls)

    # Skip the victim for this step so it is not in ``scheduled_running_reqs``
    # when it gets preempted. This is exactly the case the old
    # ``self.running.remove(victim)`` handling got wrong: the cursor was only
    # corrected for victims that had already been scheduled this step.
    victim.num_computed_tokens = victim.num_tokens_with_spec + victim.num_output_placeholders

    # The trigger cannot allocate, so the victim is preempted to free blocks.
    original_allocate = scheduler.kv_cache_manager.allocate_slots
    failed = {"done": False}

    def allocate(request, num_new_tokens, **kwargs):
        if not failed["done"] and request.request_id == trigger.request_id:
            failed["done"] = True
            return None
        return original_allocate(request, num_new_tokens, **kwargs)

    scheduler.kv_cache_manager.allocate_slots = allocate
    try:
        output = scheduler.schedule()
    finally:
        scheduler.kv_cache_manager.allocate_slots = original_allocate

    assert victim.request_id in output.preempted_req_ids
    assert victim.status == RequestStatus.PREEMPTED
    # The requests behind the victim must still be scheduled in this step.
    assert trigger.request_id in output.num_scheduled_tokens, "the request behind the victim was skipped"
    assert tail.request_id in output.num_scheduled_tokens, "the request behind the victim was skipped"


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_priority_preempt_removes_victim_by_index(scheduler_cls):
    """The preemption victim must be removed by index, not by ``list.remove``.

    ``list.remove`` shifts every request behind the victim one slot to the
    left. The loop cursor has to move back with it, which is only possible when
    the victim's index is known -- hence the by-index removal that upstream
    vLLM uses.
    """
    scheduler, victim, trigger, tail = _make_running(scheduler_cls)

    original_allocate = scheduler.kv_cache_manager.allocate_slots
    failed = {"done": False}

    def allocate(request, num_new_tokens, **kwargs):
        if not failed["done"] and request.request_id == trigger.request_id:
            failed["done"] = True
            return None
        return original_allocate(request, num_new_tokens, **kwargs)

    scheduler.kv_cache_manager.allocate_slots = allocate
    try:
        scheduler.schedule()
    finally:
        scheduler.kv_cache_manager.allocate_slots = original_allocate

    # The victim is gone from the running queue, the other two are still there.
    running_ids = [req.request_id for req in scheduler.running]
    assert victim.request_id not in running_ids
    assert trigger.request_id in running_ids
    assert tail.request_id in running_ids


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_priority_preempt_of_scheduled_victim_keeps_cursor(scheduler_cls):
    """Preempting a victim that was already scheduled must keep the cursor.

    Here the victim is scheduled *before* the trigger in the same step, so it
    sits in front of the loop cursor when it gets preempted. Removing it from
    ``self.running`` shifts the trigger one slot to the left, so the cursor has
    to move back as well -- otherwise the trigger is skipped and the request
    behind it is processed in its place.
    """
    scheduler, victim, trigger, tail = _make_running(scheduler_cls)

    # The victim is scheduled first in this step: it is the head of the running
    # queue, so the loop cursor is already past it when the trigger fails to
    # allocate and the victim is preempted.
    assert scheduler.running[0] is victim

    original_allocate = scheduler.kv_cache_manager.allocate_slots
    failed = {"done": False}

    def allocate(request, num_new_tokens, **kwargs):
        if not failed["done"] and request.request_id == trigger.request_id:
            failed["done"] = True
            return None
        return original_allocate(request, num_new_tokens, **kwargs)

    scheduler.kv_cache_manager.allocate_slots = allocate
    try:
        output = scheduler.schedule()
    finally:
        scheduler.kv_cache_manager.allocate_slots = original_allocate

    # The trigger must still be scheduled in this step.
    assert trigger.request_id in output.num_scheduled_tokens, "the request behind the victim was skipped"
