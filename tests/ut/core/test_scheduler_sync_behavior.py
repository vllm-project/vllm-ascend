# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Behavior tests covering synced Ascend schedulers across vLLM versions."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.stats import (
    KV_FETCH_COMPLETED_WAITING,
    KV_FETCH_IN_PROGRESS,
    KV_FETCH_STAGES,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus

from tests.ut.core.test_dyntra_lb_scheduler import (
    create_dyntra_lb_scheduler,
    make_dyntra_test_config,
)
from tests.ut.kv_offload.utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
)
from vllm_ascend.ascend_config import clear_ascend_config
from vllm_ascend.core.batch_job_aware_scheduler import BatchJobAwareScheduler
from vllm_ascend.core.dyntra_lb_scheduler import (
    AsyncDyntraLBScheduler,
    DyntraLBScheduler,
    get_dyntra_lb_request_block_num,
)
from vllm_ascend.core.recompute_scheduler import (
    AsyncDyntraLBRecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    RecomputeReqInfo,
    RecomputeScheduler,
)
from vllm_ascend.core.short_request_first_scheduler import ShortRequestFirstRequestQueue
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import vllm_version_is

_UPSTREAM_SCHEDULER = next(
    cls
    for cls in BalanceScheduler.__mro__
    if cls.__module__ == "vllm.v1.core.sched.scheduler" and cls.__name__ == "Scheduler"
)


def _make_scheduler(
    scheduler_cls: type[Any],
    *,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 512,
    block_size: int = 16,
    num_blocks: int = 1000,
    with_connector: bool = True,
    balance_enabled: bool = False,
    max_async_kv_loads_per_step: int | None = None,
    max_num_inflight_kv_loads: int | None = None,
    async_scheduling: bool = False,
    additional_config: dict | None = None,
) -> Any:
    cfg = make_dyntra_test_config(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=block_size,
    )
    if not with_connector:
        cfg.kv_transfer_config = None
    cfg.scheduler_config.async_scheduling = async_scheduling
    cfg.scheduler_config.max_num_inflight_kv_loads = max_num_inflight_kv_loads
    if not vllm_version_is("0.34.0"):
        cfg.scheduler_config.max_async_kv_loads_per_step = max_async_kv_loads_per_step
    if additional_config is not None:
        cfg.additional_config = additional_config
    elif balance_enabled:
        cfg.additional_config = {"scheduler_config": {"enable_balance_scheduling": True}}

    scheduler: Any = create_dyntra_lb_scheduler(cfg, scheduler_cls=scheduler_cls, num_blocks=num_blocks)
    if scheduler_cls is BalanceScheduler:
        scheduler._balance_enabled = balance_enabled
        if balance_enabled:
            scheduler.balance_queue = [torch.tensor(0), torch.tensor(0)]
    return scheduler


def _assert_parked_reservation(scheduler, request) -> None:
    assert request in scheduler._parked_reservations
    assert request not in scheduler._inflight_prefills
    assert scheduler._inflight_prefill_reserved_blocks() == scheduler._request_remaining_blocks(request)


def _assert_no_kv_load_leaks(scheduler) -> None:
    assert_scheduler_empty(scheduler)
    assert len(scheduler.kv_holding_waiting) == 0
    assert len(scheduler.async_kv_waiting) == 0
    assert len(scheduler._inflight_prefills) == 0
    assert len(scheduler._parked_reservations) == 0
    assert scheduler._inflight_prefill_reserved_blocks() == 0
    assert len(scheduler._aborted_kv_loads) == 0
    assert all(scheduler._kv_fetch_counts[stage] == 0 for stage in KV_FETCH_STAGES)


def _empty_runner_output(
    *,
    finished_recving: set[str] | None = None,
    invalid_block_ids: set[int] | None = None,
) -> ModelRunnerOutput:
    kv_output = KVConnectorOutput(
        finished_recving=finished_recving,
        invalid_block_ids=invalid_block_ids or set(),
    )
    return ModelRunnerOutput(
        req_ids=[],
        sampled_token_ids=[],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=kv_output,
    )


# ---------------------------------------------------------------------------
# BalanceScheduler delegation & rank-snapshot freeze/resume
# ---------------------------------------------------------------------------


def test_balance_disabled_delegates_to_upstream_and_increments_step_once(monkeypatch):
    """When disabled, BalanceScheduler delegates directly to upstream and increments current_step once."""
    scheduler = _make_scheduler(BalanceScheduler, with_connector=False, balance_enabled=False)
    upstream_calls = 0
    orig_upstream = _UPSTREAM_SCHEDULER.schedule

    def spy_schedule(self, throttle_prefills: bool = False):
        nonlocal upstream_calls
        upstream_calls += 1
        return orig_upstream(self, throttle_prefills)

    monkeypatch.setattr(_UPSTREAM_SCHEDULER, "schedule", spy_schedule)

    req = create_request(1, num_tokens=16, block_size=16)
    scheduler.add_request(req)
    step_before = scheduler.current_step

    output = scheduler.schedule()

    assert upstream_calls == 1
    assert scheduler.current_step == step_before + 1
    assert req in scheduler.running
    assert output.num_scheduled_tokens[req.request_id] == 16


def test_balance_rank_snapshot_freezes_waiting_and_async_loads_then_resumes():
    """2-rank snapshot progression below -> at cap -> below freezes then resumes."""
    block_size = 16
    scheduler = _make_scheduler(
        BalanceScheduler,
        max_num_seqs=2,
        block_size=block_size,
        with_connector=True,
        balance_enabled=True,
        max_num_inflight_kv_loads=2,
    )

    # Phase 1: below cap (max=1 < 2) -> running request admits normally.
    scheduler.balance_queue = [torch.tensor(0), torch.tensor(1)]
    r_run = create_request(1, num_tokens=block_size, block_size=block_size, do_remote_prefill=False)
    scheduler.add_request(r_run)
    out1 = scheduler.schedule()
    assert scheduler.running == [r_run]
    assert out1.num_scheduled_tokens[r_run.request_id] == block_size

    # Phase 2: at cap (max=2 == max_num_running_reqs) -> running keeps decoding;
    # both plain waiting and capped async KV loads freeze completely.
    scheduler.update_from_output(out1, create_model_runner_output([r_run]))
    r_wait = create_request(2, num_tokens=block_size, block_size=block_size, do_remote_prefill=False)
    r_remote = create_request(3, num_tokens=block_size * 3, block_size=block_size, do_remote_prefill=True)
    scheduler.add_request(r_wait)
    scheduler.add_request(r_remote)
    scheduler.balance_queue = [torch.tensor(1), torch.tensor(2)]

    out2 = scheduler.schedule()
    assert scheduler.running == [r_run]
    assert out2.num_scheduled_tokens == {r_run.request_id: 1}
    assert r_wait in scheduler.waiting
    assert r_remote.status == RequestStatus.WAITING
    assert r_remote in scheduler.async_kv_waiting

    # Phase 3: snapshot drops below cap (max=1 < 2) -> waiting & async load resume.
    scheduler.update_from_output(out2, create_model_runner_output([r_run]))
    scheduler.balance_queue = [torch.tensor(1), torch.tensor(1)]

    out3 = scheduler.schedule()
    assert r_run in scheduler.running
    assert r_wait in scheduler.running
    assert out3.num_scheduled_tokens[r_wait.request_id] == block_size
    assert r_remote.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert r_remote in scheduler.kv_holding_waiting


# ---------------------------------------------------------------------------
# Versioned iteration details / encoder input stats output fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scheduler_cls",
    [BalanceScheduler, DyntraLBScheduler, RecomputeScheduler],
)
def test_versioned_iteration_details_and_encoder_stats_across_schedulers(scheduler_cls):
    """All three copied schedulers emit iteration_details on main and encoder stats on v0.34.0."""
    scheduler = _make_scheduler(
        scheduler_cls,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        block_size=16,
        num_blocks=64,
        with_connector=False,
        balance_enabled=(scheduler_cls is BalanceScheduler),
    )
    scheduler.observability_config.enable_logging_iteration_details = True
    req = create_request(1, num_tokens=16, block_size=16)
    scheduler.add_request(req)

    out = scheduler.schedule()
    assert out.total_num_scheduled_tokens == 16
    if vllm_version_is("0.34.0"):
        assert not hasattr(out, "iteration_details")
        assert not hasattr(out, "ec_manager_metadata")
        assert out.scheduled_encoder_input_stats is None
    else:
        assert out.iteration_details is not None
        assert out.iteration_details.num_prefill_requests == 1


# ---------------------------------------------------------------------------
# DyntraLB KV entrypoints & admission behavior
# ---------------------------------------------------------------------------


@pytest.mark.skipif(vllm_version_is("0.34.0"), reason="max_async_kv_loads_per_step requires vLLM main")
@pytest.mark.parametrize("scheduler_cls", [DyntraLBScheduler, DyntraLBRecomputeScheduler])
def test_dyntra_lb_capped_async_kv_load_acceptance(monkeypatch, scheduler_cls):
    """Verify DyntraLB async KV load throttling with per-step cap=1 and inflight cap=2."""
    block_size = 16
    scheduler = _make_scheduler(
        scheduler_cls,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        block_size=block_size,
        max_async_kv_loads_per_step=1,
        max_num_inflight_kv_loads=2,
    )
    scheduler._lb_kv_prefetch_enabled = True

    external_alloc_req_ids: list[str] = []
    orig_update = scheduler.connector.update_state_after_alloc

    def counting_update(request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            external_alloc_req_ids.append(request.request_id)
        return orig_update(request, blocks, num_external_tokens)

    monkeypatch.setattr(scheduler.connector, "update_state_after_alloc", counting_update)

    r1 = create_request(1, num_tokens=block_size * 3, block_size=block_size, do_remote_prefill=True)
    r2 = create_request(2, num_tokens=block_size * 3, block_size=block_size, do_remote_prefill=True)
    local = create_request(3, num_tokens=block_size, block_size=block_size, do_remote_prefill=False)
    for req in (r1, r2, local):
        scheduler.add_request(req)

    # Consecutive prepare_dyntra_lb_step calls must not allocate or start loads.
    for _ in range(2):
        candidates = scheduler.prepare_dyntra_lb_step()
        assert external_alloc_req_ids == []
        assert r1.status == RequestStatus.WAITING
        assert r2.status == RequestStatus.WAITING
        assert list(scheduler.async_kv_waiting) == [r1, r2]
        assert candidates == [r1, r2, local]

    # Whitelist only the local request (1 block); remote requests have 3 blocks.
    local_blks = get_dyntra_lb_request_block_num(scheduler, local)
    remote_blks = get_dyntra_lb_request_block_num(scheduler, r1)
    assert local_blks != remote_blks
    scheduler.modifications = {"out_blk": [], "in_blk": [local_blks], "freeze": False}

    # Step 1: R1 starts loading (per-step cap=1, 0 forward tokens); R2 waits;
    # local request schedules normally.
    out1 = scheduler.schedule()
    assert external_alloc_req_ids == [r1.request_id]
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert r1 in scheduler.kv_holding_waiting
    assert r1.request_id not in out1.num_scheduled_tokens
    _assert_parked_reservation(scheduler, r1)
    assert r2.status == RequestStatus.WAITING
    assert list(scheduler.async_kv_waiting) == [r2]
    assert scheduler.running == [local]
    assert out1.num_scheduled_tokens[local.request_id] == block_size

    # Step 2: R1 finishes transfer, still not whitelisted -> holds inflight slot.
    # R2 starts loading (filling inflight cap=2), so newly arrived R3 must wait.
    r3 = create_request(4, num_tokens=block_size * 3, block_size=block_size, do_remote_prefill=True)
    scheduler.add_request(r3)
    scheduler.update_from_output(
        out1,
        create_model_runner_output([local], finished_recving={r1.request_id}),
    )
    scheduler.prepare_dyntra_lb_step()
    assert external_alloc_req_ids == [r1.request_id]
    scheduler.modifications = {"out_blk": [], "in_blk": [], "freeze": False}
    out2 = scheduler.schedule()
    assert external_alloc_req_ids == [r1.request_id, r2.request_id]
    assert r1.status == RequestStatus.WAITING
    assert r1 in scheduler.kv_holding_waiting
    assert r1.request_id not in out2.num_scheduled_tokens
    assert scheduler._kv_fetch_counts[KV_FETCH_COMPLETED_WAITING] == 1
    assert scheduler._kv_fetch_counts[KV_FETCH_IN_PROGRESS] == 1
    assert r2.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert r3.status == RequestStatus.WAITING
    assert list(scheduler.async_kv_waiting) == [r3]

    # Step 3: Admitting R1 promotes it to running, releases its parked
    # reservation, and frees an inflight slot so R3 can start loading.
    scheduler.update_from_output(
        out2,
        create_model_runner_output([local], finished_recving={r2.request_id}),
    )
    scheduler.prepare_dyntra_lb_step()
    scheduler.modifications = {"out_blk": [], "in_blk": [remote_blks], "freeze": False}
    out3 = scheduler.schedule()
    assert r1 in scheduler.running
    assert r1 not in scheduler._parked_reservations
    assert r1.request_id in out3.num_scheduled_tokens
    assert r3.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert external_alloc_req_ids == [r1.request_id, r2.request_id, r3.request_id]

    # Finish local and R1 normally via EOS; abort R2 (completed) and R3 (loading).
    scheduler.update_from_output(out3, create_model_runner_output([local, r1], use_eos=True))
    scheduler.finish_requests(r2.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.finish_requests(r3.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.modifications = None
    out4 = scheduler.schedule()
    scheduler.update_from_output(
        out4,
        create_model_runner_output([], finished_recving={r3.request_id}),
    )
    scheduler.schedule()
    _assert_no_kv_load_leaks(scheduler)


def test_dyntra_lb_inflight_cap_only_full_batch_and_empty_whitelist():
    """Verify inflight-cap-only full batch throttling and empty whitelist on both v0.34.0 and main."""
    block_size = 16

    # 1. In-flight cap only: prefetch disabled; full batch allows loads up to cap.
    sched_inflight = _make_scheduler(
        DyntraLBScheduler,
        max_num_seqs=1,
        block_size=block_size,
        max_async_kv_loads_per_step=None,
        max_num_inflight_kv_loads=2,
    )
    sched_inflight._lb_kv_prefetch_enabled = True

    local2 = create_request(10, num_tokens=block_size, block_size=block_size, do_remote_prefill=False)
    remotes = [
        create_request(11 + i, num_tokens=block_size * 3, block_size=block_size, do_remote_prefill=True)
        for i in range(3)
    ]
    sched_inflight.add_request(local2)
    out2 = sched_inflight.schedule()
    for req in remotes:
        sched_inflight.add_request(req)
    sched_inflight.prepare_dyntra_lb_step()
    assert all(req.status == RequestStatus.WAITING for req in remotes)

    sched_inflight.update_from_output(out2, create_model_runner_output([local2]))
    sched_inflight.modifications = {"out_blk": [], "in_blk": [], "freeze": False}
    sched_inflight.schedule()
    assert [req.status for req in remotes] == [
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING,
    ]

    # 2. Empty whitelist with free slot: pure async load still starts with 0 forward tokens.
    sched_empty = _make_scheduler(
        DyntraLBScheduler,
        max_num_seqs=2,
        block_size=block_size,
        max_async_kv_loads_per_step=None,
        max_num_inflight_kv_loads=2,
    )
    sched_empty._lb_kv_prefetch_enabled = True

    r_only = create_request(20, num_tokens=block_size * 3, block_size=block_size, do_remote_prefill=True)
    sched_empty.add_request(r_only)
    sched_empty.prepare_dyntra_lb_step()
    sched_empty.modifications = {"out_blk": [], "in_blk": [], "freeze": False}
    out_empty = sched_empty.schedule()
    assert out_empty.total_num_scheduled_tokens == 0
    assert r_only.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert r_only in sched_empty.kv_holding_waiting


@pytest.mark.skipif(vllm_version_is("0.34.0"), reason="max_async_kv_loads_per_step requires vLLM main")
def test_dyntra_lb_per_step_cap_only_blocks_loads_on_full_batch():
    """Per-step cap only disables prefetch and blocks new async loads when running is full."""
    block_size = 16
    sched_step = _make_scheduler(
        DyntraLBScheduler,
        max_num_seqs=1,
        block_size=block_size,
        max_async_kv_loads_per_step=1,
        max_num_inflight_kv_loads=None,
    )
    sched_step._lb_kv_prefetch_enabled = True

    local = create_request(1, num_tokens=block_size, block_size=block_size, do_remote_prefill=False)
    remote = create_request(2, num_tokens=block_size * 3, block_size=block_size, do_remote_prefill=True)
    sched_step.add_request(local)
    out = sched_step.schedule()
    assert sched_step.running == [local]

    sched_step.add_request(remote)
    sched_step.prepare_dyntra_lb_step()
    assert remote.status == RequestStatus.WAITING
    sched_step.update_from_output(out, create_model_runner_output([local]))
    sched_step.schedule()
    assert remote.status == RequestStatus.WAITING


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_dyntra_lb_waiting_order_merges_with_waiting_tiebreak(policy: str):
    """Merge waiting + async_kv_waiting like _next_waiting_queue without mutation."""
    vllm_config = make_dyntra_test_config()
    vllm_config.scheduler_config.policy = policy
    scheduler = create_dyntra_lb_scheduler(vllm_config, scheduler_cls=DyntraLBScheduler)

    holder = create_request(1)
    w_early = create_request(2)
    a_mid = create_request(3)
    w_tie = create_request(4)
    a_tie = create_request(5)
    a_late = create_request(6)

    holder.priority = 10
    holder.arrival_time = 100.0
    if policy == "fcfs":
        w_early.arrival_time = 1.0
        a_mid.arrival_time = 2.0
        w_tie.arrival_time = 3.0
        a_tie.arrival_time = 3.0
        a_late.arrival_time = 4.0
        scheduler.kv_holding_waiting.add_request(holder)
        scheduler.waiting.add_request(w_early)
        scheduler.waiting.add_request(w_tie)
        scheduler.async_kv_waiting.add_request(a_mid)
        scheduler.async_kv_waiting.add_request(a_tie)
        scheduler.async_kv_waiting.add_request(a_late)
        expected = [holder, w_early, a_mid, w_tie, a_tie, a_late]
    else:
        assert scheduler.policy == SchedulingPolicy.PRIORITY
        w_early.priority, w_early.arrival_time = 0, 5.0
        a_mid.priority, a_mid.arrival_time = 1, 1.0
        a_late.priority, a_late.arrival_time = 3, 1.0

        # Use requests whose __lt__ returns False in both directions for tie check.
        class _EqualPriorityReq:
            def __init__(self, label: str, priority: int, arrival_time: float):
                self.label = label
                self.priority = priority
                self.arrival_time = arrival_time

            def __lt__(self, other) -> bool:
                if self.priority != other.priority:
                    return self.priority < other.priority
                return self.arrival_time < other.arrival_time

        w_tie = _EqualPriorityReq("w_tie", 2, 2.0)
        a_tie = _EqualPriorityReq("a_tie", 2, 2.0)
        scheduler.kv_holding_waiting.add_request(holder)
        scheduler.waiting.add_request(w_tie)
        scheduler.waiting.add_request(w_early)
        scheduler.async_kv_waiting.add_request(a_late)
        scheduler.async_kv_waiting.add_request(a_tie)
        scheduler.async_kv_waiting.add_request(a_mid)
        expected = [holder, w_early, a_mid, w_tie, a_tie, a_late]

    before_holding = list(scheduler.kv_holding_waiting)
    before_waiting = list(scheduler.waiting)
    before_async = list(scheduler.async_kv_waiting)

    ordered = scheduler._waiting_requests_in_schedule_order()

    assert ordered == expected
    assert list(scheduler.kv_holding_waiting) == before_holding
    assert list(scheduler.waiting) == before_waiting
    assert list(scheduler.async_kv_waiting) == before_async


def test_dyntra_lb_uncapped_prefetch_matches_schedule_load(monkeypatch):
    """Without load caps, prefetch and schedule() produce identical load state."""
    block_size = 16

    def _make_pair():
        sched = _make_scheduler(DyntraLBScheduler, max_num_seqs=4, block_size=block_size)
        matched_calls = 0
        alloc_calls = 0
        orig_matched = sched.connector.get_num_new_matched_tokens
        orig_alloc = sched.connector.update_state_after_alloc

        def wrap_matched(*args, **kwargs):
            nonlocal matched_calls
            matched_calls += 1
            return orig_matched(*args, **kwargs)

        def wrap_alloc(*args, **kwargs):
            nonlocal alloc_calls
            alloc_calls += 1
            return orig_alloc(*args, **kwargs)

        monkeypatch.setattr(sched.connector, "get_num_new_matched_tokens", wrap_matched)
        monkeypatch.setattr(sched.connector, "update_state_after_alloc", wrap_alloc)
        req = create_request(1, num_tokens=block_size * 3, block_size=block_size, do_remote_prefill=True)
        sched.add_request(req)
        return sched, req, lambda: (matched_calls, alloc_calls)

    sched_prefetch, req_prefetch, counts_prefetch = _make_pair()
    sched_direct, req_direct, counts_direct = _make_pair()

    sched_prefetch._lb_kv_prefetch_enabled = True
    candidates = sched_prefetch.prepare_dyntra_lb_step()
    assert candidates == []

    sched_direct.schedule()

    assert req_prefetch.status == req_direct.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req_prefetch.num_computed_tokens == req_direct.num_computed_tokens
    assert len(sched_prefetch.kv_cache_manager.get_block_ids(req_prefetch.request_id)[0]) == len(
        sched_direct.kv_cache_manager.get_block_ids(req_direct.request_id)[0]
    )
    assert (
        sched_prefetch.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
        == sched_direct.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    )
    assert sched_prefetch._inflight_prefill_reserved_blocks() == sched_direct._inflight_prefill_reserved_blocks()
    _assert_parked_reservation(sched_prefetch, req_prefetch)
    _assert_parked_reservation(sched_direct, req_direct)
    assert counts_prefetch() == counts_direct() == (1, 1)
    assert sched_prefetch._kv_fetch_stages is not None
    assert sched_direct._kv_fetch_stages is not None
    assert sched_prefetch._kv_fetch_stages[req_prefetch] == KV_FETCH_IN_PROGRESS
    assert sched_direct._kv_fetch_stages[req_direct] == KV_FETCH_IN_PROGRESS

    # Following schedule() must neither re-allocate nor notify the connector again.
    free_before = sched_prefetch.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    out = sched_prefetch.schedule()
    assert out.total_num_scheduled_tokens == 0
    assert counts_prefetch() == (1, 1)
    assert sched_prefetch.kv_cache_manager.block_pool.free_block_queue.num_free_blocks == free_before


def test_dyntra_lb_paused_request_retains_blocks_and_resumes_same_blocks():
    """LB-paused requests retain KV blocks and resume with those exact blocks."""
    block_size = 16
    scheduler = _make_scheduler(DyntraLBScheduler, max_num_seqs=2, block_size=block_size, with_connector=False)

    req = create_request(1, num_tokens=block_size - 2, block_size=block_size)
    scheduler.add_request(req)
    out1 = scheduler.schedule()
    scheduler.update_from_output(out1, create_model_runner_output([req]))
    retained_blocks = scheduler.kv_cache_manager.get_block_ids(req.request_id)
    free_before_pause = scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    blk_num = get_dyntra_lb_request_block_num(scheduler, req)

    scheduler.prepare_dyntra_lb_step()
    scheduler.modifications = {"out_blk": [blk_num], "in_blk": [], "freeze": False}
    out2 = scheduler.schedule()
    assert out2.total_num_scheduled_tokens == 0
    assert req in scheduler.kv_holding_waiting
    assert req.request_id in scheduler._lb_paused_req_ids
    assert scheduler.kv_cache_manager.get_block_ids(req.request_id) == retained_blocks
    assert scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks == free_before_pause

    candidates = scheduler.prepare_dyntra_lb_step()
    assert candidates == [req]
    scheduler.modifications = {"out_blk": [], "in_blk": [blk_num], "freeze": False}
    out3 = scheduler.schedule()
    assert req in scheduler.running
    assert req.request_id not in scheduler._lb_paused_req_ids
    assert len(out3.scheduled_new_reqs) == 1
    assert out3.scheduled_new_reqs[0].req_id == req.request_id
    assert out3.scheduled_new_reqs[0].block_ids == retained_blocks


def test_dyntra_lb_completed_holder_requires_whitelist_to_forward():
    """A request whose async KV load finished still needs whitelist admission."""
    block_size = 16
    scheduler = _make_scheduler(DyntraLBScheduler, max_num_seqs=2, block_size=block_size)

    req = create_request(1, num_tokens=block_size * 2, block_size=block_size, do_remote_prefill=True)
    scheduler.add_request(req)
    out1 = scheduler.schedule()
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.update_from_output(
        out1,
        create_model_runner_output([], finished_recving={req.request_id}),
    )

    candidates = scheduler.prepare_dyntra_lb_step()
    assert candidates == [req]
    assert req in scheduler.kv_holding_waiting
    assert req.status == RequestStatus.WAITING

    scheduler.modifications = {"out_blk": [], "in_blk": [], "freeze": False}
    out_blocked = scheduler.schedule()
    assert out_blocked.total_num_scheduled_tokens == 0
    assert req in scheduler.kv_holding_waiting
    assert req not in scheduler.running

    blk_num = get_dyntra_lb_request_block_num(scheduler, req)
    scheduler.prepare_dyntra_lb_step()
    scheduler.modifications = {"out_blk": [], "in_blk": [blk_num], "freeze": False}
    out_admitted = scheduler.schedule()
    assert req in scheduler.running
    assert req.request_id in out_admitted.num_scheduled_tokens


def test_dyntra_lb_post_plan_arrival_is_not_admitted():
    """Requests arriving after prepare_dyntra_lb_step() are not admitted by that plan."""
    block_size = 16
    scheduler = _make_scheduler(DyntraLBScheduler, max_num_seqs=4, block_size=block_size, with_connector=False)

    planned = create_request(1, num_tokens=block_size, block_size=block_size)
    scheduler.add_request(planned)
    candidates = scheduler.prepare_dyntra_lb_step()
    assert candidates == [planned]

    blk_num = get_dyntra_lb_request_block_num(scheduler, planned)
    scheduler.modifications = {"out_blk": [], "in_blk": [blk_num, blk_num], "freeze": False}

    late_arrival = create_request(2, num_tokens=block_size, block_size=block_size)
    scheduler.add_request(late_arrival)

    out = scheduler.schedule()
    assert planned in scheduler.running
    assert planned.request_id in out.num_scheduled_tokens
    assert late_arrival not in scheduler.running
    assert late_arrival in scheduler.waiting
    assert late_arrival.request_id not in out.num_scheduled_tokens


# ---------------------------------------------------------------------------
# RecomputeScheduler end-to-end schedule() + update_from_output()
# ---------------------------------------------------------------------------


def _setup_recompute_preemption_pressure(monkeypatch, *, oproj_tp: int = 0, mlp_tp: int = 0):
    """Create a RecomputeScheduler where the next decode step forces preemption of req2."""
    block_size = 4
    ftpc = SimpleNamespace(oproj_tensor_parallel_size=oproj_tp, mlp_tensor_parallel_size=mlp_tp)
    monkeypatch.setattr(
        "vllm_ascend.core.recompute_scheduler.get_ascend_config",
        lambda: SimpleNamespace(finegrained_tp_config=ftpc),
    )
    # 3 blocks total (1 null + 2 usable): req1 & req2 each take 1 full block,
    # leaving 0 free blocks when req1 needs a second block on the next decode step.
    scheduler = _make_scheduler(
        RecomputeScheduler,
        max_num_seqs=3,
        max_num_batched_tokens=64,
        block_size=block_size,
        num_blocks=3,
        with_connector=False,
    )
    req1 = create_request(1, num_tokens=block_size, block_size=block_size)
    req2 = create_request(2, num_tokens=block_size, block_size=block_size)
    req1.client_index = 0
    req2.client_index = 1
    scheduler.add_request(req1)
    scheduler.add_request(req2)
    out1 = scheduler.schedule()
    assert scheduler.running == [req1, req2]
    scheduler.update_from_output(out1, create_model_runner_output([req1, req2]))
    return scheduler, req1, req2


def test_recompute_schedule_offload_success_uses_local_preemption(monkeypatch):
    """End-to-end: successful offload keeps local preemption with no recomputed_reqs."""
    scheduler, req1, req2 = _setup_recompute_preemption_pressure(monkeypatch)
    connector = MagicMock()
    connector.update_state_before_preempt.return_value = True
    connector.has_pending_block_frees.return_value = False
    connector.get_loaded_kv_cache_group_ids.return_value = ()
    connector.build_connector_meta.return_value = None
    connector.request_finished.return_value = (False, None)
    scheduler.connector = connector

    out = scheduler.schedule()

    connector.update_state_before_preempt.assert_called_once()
    assert req1 in scheduler.running
    assert req2 in scheduler.waiting
    assert req2.status == RequestStatus.PREEMPTED
    assert req2.request_id not in out.finished_req_ids
    assert out.recomputed_reqs is None


@pytest.mark.parametrize("offload_behavior", [False, RuntimeError("offload boom")])
def test_recompute_schedule_offload_failure_notifies_owning_client_once(monkeypatch, offload_behavior):
    """End-to-end: offload failure/exception finishes for recompute and notifies owning client once."""
    scheduler, req1, req2 = _setup_recompute_preemption_pressure(monkeypatch)
    connector = MagicMock()
    if isinstance(offload_behavior, Exception):
        connector.update_state_before_preempt.side_effect = offload_behavior
    else:
        connector.update_state_before_preempt.return_value = offload_behavior
    connector.has_pending_block_frees.return_value = False
    connector.get_loaded_kv_cache_group_ids.return_value = ()
    connector.build_connector_meta.return_value = None
    connector.request_finished.return_value = (False, None)
    scheduler.connector = connector

    # Add a fresh waiting request to also verify waiting admission is skipped
    # on a step that emitted recomputed_reqs.
    waiter = create_request(3, num_tokens=4, block_size=4)
    scheduler.add_request(waiter)

    out = scheduler.schedule()

    assert req1 in scheduler.running
    assert req2.request_id in out.finished_req_ids
    assert out.recomputed_reqs == [RecomputeReqInfo(req2.request_id, req2.client_index)]
    assert waiter in scheduler.waiting
    assert waiter not in scheduler.running

    outputs = scheduler.update_from_output(out, create_model_runner_output([req1]))
    client1_recomputed = [o for o in outputs[1].outputs if o.request_id == req2.request_id]
    assert len(client1_recomputed) == 1
    assert client1_recomputed[0].stop_reason == "recomputed"
    assert all(o.request_id != req2.request_id for o in outputs[0].outputs)


@pytest.mark.parametrize(
    ("oproj_tp", "mlp_tp"),
    [(2, 0), (0, 2)],
)
def test_recompute_schedule_finegrained_tp_aborts_without_recompute_signal(monkeypatch, oproj_tp, mlp_tp):
    """Under oproj_tp > 1 or mlp_tp > 1, offload failure aborts without sending back to P."""
    scheduler, req1, req2 = _setup_recompute_preemption_pressure(
        monkeypatch,
        oproj_tp=oproj_tp,
        mlp_tp=mlp_tp,
    )
    connector = MagicMock()
    connector.update_state_before_preempt.return_value = False
    connector.has_pending_block_frees.return_value = False
    connector.get_loaded_kv_cache_group_ids.return_value = ()
    connector.build_connector_meta.return_value = None
    connector.request_finished.return_value = (False, None)
    scheduler.connector = connector

    out = scheduler.schedule()

    assert req1 in scheduler.running
    assert req2.request_id in out.finished_req_ids
    assert out.recomputed_reqs is None
    outputs = scheduler.update_from_output(out, create_model_runner_output([req1]))
    for client_outputs in outputs.values():
        assert all(o.stop_reason != "recomputed" for o in client_outputs.outputs)


# ---------------------------------------------------------------------------
# BatchJobAware with platform config (Balance on & off)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("balance_enabled", [False, True])
def test_batch_job_aware_blocks_unadmittable_batch_and_admits_interactive(balance_enabled: bool):
    """Platform-configured BatchJobAware handles blocked batch jobs and admits interactive work."""
    block_size = 16
    additional_config = {
        "scheduler_config": {
            "enable_balance_scheduling": balance_enabled,
            "batch_job_sched_config": {
                "enabled": True,
                "reserve_margin_blocks": 2,
                "reserve_max_blocks": 16,
                "short_decode_token_threshold": 64,
            },
        }
    }
    cfg = make_dyntra_test_config(
        max_num_seqs=4,
        max_num_batched_tokens=512,
        block_size=block_size,
    )
    cfg.kv_transfer_config = None
    cfg.scheduler_config.async_scheduling = False
    cfg.device_config.device_type = "npu"
    cfg.additional_config = additional_config
    clear_ascend_config()
    try:
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("vllm_ascend.platform.get_current_hardware_profile") as mock_profile,
        ):
            mock_profile.return_value.supports.return_value = False
            mock_profile.return_value.default_worker_cls = "vllm_ascend.worker.worker_v1.NPUWorker"
            NPUPlatform.check_and_update_config(cfg)
        assert cfg.scheduler_config.scheduler_cls == (
            "vllm_ascend.core.batch_job_aware_scheduler.BatchJobAwareScheduler"
        )
        effective_block_size = cfg.cache_config.block_size

        # 4 total blocks (3 usable) with reserve_margin_blocks=2: once one
        # running request holds 1 block (2 free left), free_blocks - reserve == 0
        # so the waiting batch job is unadmittable (len > 0, bool == False).
        scheduler = create_dyntra_lb_scheduler(cfg, scheduler_cls=BatchJobAwareScheduler, num_blocks=4)
        scheduler._balance_enabled = balance_enabled
        if balance_enabled:
            scheduler.balance_queue = [torch.tensor(0), torch.tensor(0)]

        running_req = create_request(1, num_tokens=effective_block_size - 2, block_size=effective_block_size)
        running_req.request_id = "r1#job_name[heavy_job]#"
        scheduler.add_request(running_req)
        out1 = scheduler.schedule()
        assert scheduler.running == [running_req]
        scheduler.update_from_output(out1, create_model_runner_output([running_req]))

        # Add a waiting batch request that cannot be admitted while reserve consumes budget.
        blocked_batch = create_request(2, num_tokens=effective_block_size - 2, block_size=effective_block_size)
        blocked_batch.request_id = "r2#job_name[heavy_job]#"
        scheduler.add_request(blocked_batch)
        scheduler._block_reserver.invalidate_cache()
        assert len(scheduler.waiting) == 1
        assert not bool(scheduler.waiting)

        out2 = scheduler.schedule()
        assert blocked_batch in scheduler.waiting
        assert blocked_batch not in scheduler.running
        assert out2.num_scheduled_tokens == {running_req.request_id: 1}

        # Finish the running request to free a block (3 free - 2 margin = 1 block
        # budget); an interactive request must now be admitted cleanly.
        scheduler.update_from_output(out2, create_model_runner_output([running_req], use_eos=True))
        scheduler.finish_requests(blocked_batch.request_id, RequestStatus.FINISHED_ABORTED)
        interactive = create_request(3, num_tokens=effective_block_size - 2, block_size=effective_block_size)
        scheduler.add_request(interactive)

        out3 = scheduler.schedule()
        assert interactive in scheduler.running
        assert out3.num_scheduled_tokens[interactive.request_id] == effective_block_size - 2
    finally:
        clear_ascend_config()


# ---------------------------------------------------------------------------
# ShortRequestFirst x Balance x cache_aware_admission_window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("balance_enabled", [False, True])
@pytest.mark.parametrize(
    "cache_aware_window",
    [
        0,
        pytest.param(
            2,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "ShortRequestFirstRequestQueue.remove_requests() "
                    "drops _long_enqueue_at during cache-aware prefix reordering, "
                    "resetting aged long-request wait time."
                ),
            ),
        ),
    ],
)
def test_short_request_first_aging_under_cache_aware_window(
    monkeypatch,
    balance_enabled: bool,
    cache_aware_window: int,
):
    """Compare SRF long-request aging across Balance on/off and window 0/2."""
    block_size = 16
    additional_config = {
        "scheduler_config": {
            "enable_balance_scheduling": balance_enabled,
            "short_request_first_config": {
                "enabled": True,
                "threshold": 32,
                "long_max_wait_ms": 50.0,
            },
        }
    }
    cfg = make_dyntra_test_config(
        max_num_seqs=1,
        max_num_batched_tokens=512,
        block_size=block_size,
    )
    cfg.kv_transfer_config = None
    cfg.scheduler_config.async_scheduling = False
    cfg.device_config.device_type = "npu"
    cfg.additional_config = additional_config
    cfg.scheduler_config.cache_aware_admission_window = cache_aware_window
    cfg.scheduler_config.cache_aware_admission_threshold = 0.0

    # Run platform config validation and ensure AscendConfig is initialized.
    clear_ascend_config()
    try:
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("vllm_ascend.platform.get_current_hardware_profile") as mock_profile,
        ):
            mock_profile.return_value.supports.return_value = False
            mock_profile.return_value.default_worker_cls = "vllm_ascend.worker.worker_v1.NPUWorker"
            NPUPlatform.check_and_update_config(cfg)

        scheduler = create_dyntra_lb_scheduler(cfg, scheduler_cls=BalanceScheduler, num_blocks=64)
        scheduler._balance_enabled = balance_enabled
        if balance_enabled:
            scheduler.balance_queue = [torch.tensor(0), torch.tensor(0)]
        assert isinstance(scheduler.waiting, ShortRequestFirstRequestQueue)

        short_req = create_request(1, num_tokens=16, block_size=block_size)
        long_req = create_request(2, num_tokens=64, block_size=block_size)
        scheduler.add_request(short_req)
        scheduler.add_request(long_req)

        # Backdate long_req's enqueue timestamp so it has already aged past 50ms.
        aged_timestamp = time.monotonic() - 1.0
        scheduler.waiting._long_enqueue_at[long_req.request_id] = aged_timestamp

        # Make long_req appear to have a deeper cached prefix so cache-aware
        # reordering moves it ahead of short_req when cache_aware_window == 2.
        monkeypatch.setattr(
            scheduler.kv_cache_manager,
            "get_num_cached_tokens",
            lambda req: block_size if req is long_req else 0,
        )

        out = scheduler.schedule()

        # The aged long request must keep its original wait time (not reset) and
        # be popped first into running.
        assert long_req in scheduler.running
        assert long_req.request_id in out.num_scheduled_tokens
        assert short_req in scheduler.waiting
    finally:
        clear_ascend_config()


# ---------------------------------------------------------------------------
# Async KV full lifecycle across all schedulers & cap combinations
# ---------------------------------------------------------------------------


_ALL_LIFECYCLE_SCHEDULERS = [
    pytest.param(BalanceScheduler, False, id="BalanceScheduler"),
    pytest.param(DyntraLBScheduler, False, id="DyntraLBScheduler"),
    pytest.param(RecomputeScheduler, False, id="RecomputeScheduler"),
    pytest.param(DyntraLBRecomputeScheduler, False, id="DyntraLBRecomputeScheduler"),
    pytest.param(AsyncDyntraLBScheduler, True, id="AsyncDyntraLBScheduler"),
    pytest.param(AsyncRecomputeScheduler, True, id="AsyncRecomputeScheduler"),
    pytest.param(AsyncDyntraLBRecomputeScheduler, True, id="AsyncDyntraLBRecomputeScheduler"),
]


@pytest.mark.parametrize(("scheduler_cls", "async_scheduling"), _ALL_LIFECYCLE_SCHEDULERS)
def test_async_kv_lifecycle_normal_abort_and_failure(
    monkeypatch,
    scheduler_cls: type,
    async_scheduling: bool,
):
    """Cover normal load->promote->finish, abort, and load failure across all variants."""
    block_size = 16
    scheduler = _make_scheduler(
        scheduler_cls,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        block_size=block_size,
        num_blocks=64,
        balance_enabled=(scheduler_cls is BalanceScheduler),
        async_scheduling=async_scheduling,
    )
    matched_calls = 0
    ext_alloc_calls = 0
    orig_matched = scheduler.connector.get_num_new_matched_tokens
    orig_alloc = scheduler.connector.update_state_after_alloc

    def wrap_matched(*args, **kwargs):
        nonlocal matched_calls
        matched_calls += 1
        return orig_matched(*args, **kwargs)

    def wrap_alloc(request, blocks, num_external_tokens):
        nonlocal ext_alloc_calls
        if num_external_tokens > 0:
            ext_alloc_calls += 1
        return orig_alloc(request, blocks, num_external_tokens)

    monkeypatch.setattr(scheduler.connector, "get_num_new_matched_tokens", wrap_matched)
    monkeypatch.setattr(scheduler.connector, "update_state_after_alloc", wrap_alloc)

    # --- 1. Normal async load -> promote to running -> finish ---
    req_ok = create_request(1, num_tokens=block_size * 2 + 4, block_size=block_size, do_remote_prefill=True)
    scheduler.add_request(req_ok)
    out1 = scheduler.schedule()
    assert out1.total_num_scheduled_tokens == 0
    assert req_ok.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req_ok in scheduler.kv_holding_waiting
    assert matched_calls == 1
    assert ext_alloc_calls == 1
    _assert_parked_reservation(scheduler, req_ok)

    scheduler.update_from_output(out1, _empty_runner_output(finished_recving={req_ok.request_id}))
    out2 = scheduler.schedule()
    assert req_ok in scheduler.running
    assert out2.num_scheduled_tokens[req_ok.request_id] > 0
    assert scheduler._inflight_prefill_reserved_blocks() == 0

    scheduler.update_from_output(out2, create_model_runner_output([req_ok], use_eos=True))
    scheduler.schedule()
    _assert_no_kv_load_leaks(scheduler)

    # --- 2. Abort while loading ---
    req_abort = create_request(2, num_tokens=block_size * 2 + 4, block_size=block_size, do_remote_prefill=True)
    scheduler.add_request(req_abort)
    out3 = scheduler.schedule()
    assert out3.total_num_scheduled_tokens == 0
    assert req_abort.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert ext_alloc_calls == 2
    _assert_parked_reservation(scheduler, req_abort)

    scheduler.finish_requests(req_abort.request_id, RequestStatus.FINISHED_ABORTED)
    out4 = scheduler.schedule()
    scheduler.update_from_output(out4, _empty_runner_output(finished_recving={req_abort.request_id}))
    scheduler.schedule()
    _assert_no_kv_load_leaks(scheduler)

    # --- 3. Load failure via invalid_block_ids (default failure_policy="fail") ---
    req_fail = create_request(3, num_tokens=block_size * 2 + 4, block_size=block_size, do_remote_prefill=True)
    scheduler.add_request(req_fail)
    out5 = scheduler.schedule()
    assert out5.total_num_scheduled_tokens == 0
    assert req_fail.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert ext_alloc_calls == 3
    loaded_block_ids = set(scheduler.kv_cache_manager.get_block_ids(req_fail.request_id)[0])
    assert loaded_block_ids

    # Report all loaded blocks invalid: under default failure_policy="fail",
    # the scheduler finishes the request with ERROR and releases all blocks.
    fail_outputs = scheduler.update_from_output(
        out5,
        _empty_runner_output(
            finished_recving={req_fail.request_id},
            invalid_block_ids=loaded_block_ids,
        ),
    )
    assert req_fail.status == RequestStatus.FINISHED_ERROR
    err_outputs = [o for o in fail_outputs[0].outputs if o.request_id == req_fail.request_id]
    assert len(err_outputs) == 1
    assert err_outputs[0].finish_reason == FinishReason.ERROR
    scheduler.schedule()
    _assert_no_kv_load_leaks(scheduler)


@pytest.mark.parametrize(
    "scheduler_cls",
    [BalanceScheduler, DyntraLBScheduler, RecomputeScheduler],
)
@pytest.mark.parametrize(
    ("max_per_step", "max_inflight"),
    [
        pytest.param(
            1,
            None,
            marks=pytest.mark.skipif(
                vllm_version_is("0.34.0"),
                reason="max_async_kv_loads_per_step requires vLLM main",
            ),
        ),
        (None, 2),
        pytest.param(
            1,
            2,
            marks=pytest.mark.skipif(
                vllm_version_is("0.34.0"),
                reason="max_async_kv_loads_per_step requires vLLM main",
            ),
        ),
    ],
)
def test_async_kv_load_caps_combinations_across_copied_schedulers(
    monkeypatch,
    scheduler_cls: type,
    max_per_step: int | None,
    max_inflight: int | None,
):
    """Test per-step only, in-flight only, and combined caps across all 3 copied schedulers."""
    block_size = 16
    scheduler = _make_scheduler(
        scheduler_cls,
        max_num_seqs=3,
        max_num_batched_tokens=512,
        block_size=block_size,
        num_blocks=64,
        balance_enabled=(scheduler_cls is BalanceScheduler),
        max_async_kv_loads_per_step=max_per_step,
        max_num_inflight_kv_loads=max_inflight,
    )
    ext_alloc_req_ids: list[str] = []
    orig_alloc = scheduler.connector.update_state_after_alloc

    def wrap_alloc(request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            ext_alloc_req_ids.append(request.request_id)
        return orig_alloc(request, blocks, num_external_tokens)

    monkeypatch.setattr(scheduler.connector, "update_state_after_alloc", wrap_alloc)

    remotes = [
        create_request(i + 1, num_tokens=block_size * 2 + 4, block_size=block_size, do_remote_prefill=True)
        for i in range(3)
    ]
    local = create_request(10, num_tokens=block_size, block_size=block_size, do_remote_prefill=False)
    for req in (*remotes, local):
        scheduler.add_request(req)

    assert list(scheduler.async_kv_waiting) == remotes
    assert list(scheduler.waiting) == [local]

    # Step 1: starts min(per_step or inf, inflight or inf) remote loads and
    # still admits the local request in the same step.
    expected_step1 = 1 if max_per_step == 1 else 2
    out1 = scheduler.schedule()
    assert len(ext_alloc_req_ids) == expected_step1
    assert out1.num_scheduled_tokens == {local.request_id: block_size}
    assert scheduler.running == [local]
    assert list(scheduler.kv_holding_waiting) == remotes[:expected_step1]
    assert list(scheduler.async_kv_waiting) == remotes[expected_step1:]
    expected_reserved = sum(scheduler._request_remaining_blocks(r) for r in remotes[:expected_step1])
    assert scheduler._inflight_prefill_reserved_blocks() == expected_reserved

    # Step 2: while step-1 loads are still in flight:
    # - (1, None): starts 2nd load (total 2)
    # - (None, 2): already at inflight cap 2 -> starts 0 more
    # - (1, 2): starts 2nd load (reaching inflight cap 2)
    scheduler.update_from_output(out1, create_model_runner_output([local]))
    out2 = scheduler.schedule()
    expected_step2_total = 2
    assert len(ext_alloc_req_ids) == expected_step2_total
    assert out2.num_scheduled_tokens == {local.request_id: 1}
    assert list(scheduler.kv_holding_waiting) == remotes[:2]
    assert list(scheduler.async_kv_waiting) == [remotes[2]]

    # Step 3: finish local and mark first two loads completed -> both promote
    # to running, freeing inflight capacity so remotes[2] starts loading.
    scheduler.update_from_output(
        out2,
        create_model_runner_output(
            [local],
            finished_recving={remotes[0].request_id, remotes[1].request_id},
            use_eos=True,
        ),
    )
    out3 = scheduler.schedule()
    assert set(scheduler.running) == {remotes[0], remotes[1]}
    assert remotes[0].request_id in out3.num_scheduled_tokens
    assert remotes[1].request_id in out3.num_scheduled_tokens
    assert remotes[2].status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert len(ext_alloc_req_ids) == 3

    # Drain all remaining requests to verify zero leaks.
    scheduler.update_from_output(
        out3,
        create_model_runner_output(
            [remotes[0], remotes[1]],
            finished_recving={remotes[2].request_id},
            use_eos=True,
        ),
    )
    out4 = scheduler.schedule()
    assert scheduler.running == [remotes[2]]
    scheduler.update_from_output(out4, create_model_runner_output([remotes[2]], use_eos=True))
    scheduler.schedule()
    _assert_no_kv_load_leaks(scheduler)
