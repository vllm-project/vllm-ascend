# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.request import RequestStatus

from tests.ut.core.test_dyntra_lb_scheduler import create_dyntra_lb_scheduler, make_dyntra_test_config
from tests.ut.kv_offload.utils import create_model_runner_output, create_request
from vllm_ascend.core.dyntra_lb_scheduler import DyntraLBPolicyMixin
from vllm_ascend.core.recompute_scheduler import (
    AsyncDyntraLBRecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    RecomputeReqInfo,
    RecomputeScheduler,
    RecomputeSchedulerConfig,
)


@pytest.fixture(autouse=True)
def _oproj_tp_config(monkeypatch):
    # _preempt_or_recompute reads the live AscendConfig singleton; default it to off.
    finegrained_tp_config = SimpleNamespace(oproj_tensor_parallel_size=0, mlp_tensor_parallel_size=0)
    monkeypatch.setattr(
        "vllm_ascend.core.recompute_scheduler.get_ascend_config",
        lambda: SimpleNamespace(finegrained_tp_config=finegrained_tp_config),
    )
    return finegrained_tp_config


def _make_preempt_scheduler(*, connector=None):
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.connector = connector
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.get_block_ids.return_value = ([3, 4],)
    scheduler._recomputed_reqs = []
    return scheduler


def test_recompute_scheduler_keeps_local_schedule_for_ascend_spec_padding():
    assert RecomputeScheduler.schedule is not Scheduler.schedule
    assert RecomputeScheduler.update_from_output is not Scheduler.update_from_output


def test_recompute_scheduler_config_leaves_class_selection_to_platform():
    vllm_config = make_dyntra_test_config()

    vllm_config.scheduler_config.async_scheduling = False
    sync_config = RecomputeSchedulerConfig.initialize_from_config(vllm_config)
    assert sync_config.scheduler_cls is None

    vllm_config.scheduler_config.async_scheduling = True
    async_config = RecomputeSchedulerConfig.initialize_from_config(vllm_config)
    assert async_config.scheduler_cls is None


def test_preempt_offloads_before_upstream_releases_blocks():
    connector = MagicMock()
    connector.update_state_before_preempt.return_value = True
    scheduler = _make_preempt_scheduler(connector=connector)
    request = SimpleNamespace(
        request_id="req-1",
        client_index=0,
        num_computed_tokens=17,
    )

    def assert_offload_completed_before_release(*args, **kwargs):
        connector.update_state_before_preempt.assert_called_once_with(
            request,
            ([3, 4],),
            17,
        )

    with patch.object(
        Scheduler,
        "_preempt_request",
        side_effect=assert_offload_completed_before_release,
    ) as upstream_preempt:
        locally_preempted = scheduler._preempt_or_recompute(
            request,
            1.5,
            drop_stale_output=True,
        )

    assert locally_preempted
    scheduler.kv_cache_manager.get_block_ids.assert_called_once_with("req-1")
    upstream_preempt.assert_called_once_with(
        request,
        1.5,
        drop_stale_output=True,
    )


@pytest.mark.parametrize("connector", [None, MagicMock(spec=[])])
def test_preempt_without_offload_hook_sends_request_back_to_p(connector):
    scheduler = _make_preempt_scheduler(connector=connector)
    request = SimpleNamespace(
        request_id="req-1",
        client_index=2,
        num_computed_tokens=17,
    )
    scheduler.finish_requests = MagicMock(return_value=[request])

    with (
        patch.object(Scheduler, "_preempt_request") as upstream_preempt,
        patch("vllm_ascend.core.recompute_scheduler.logger.warning") as warning,
    ):
        locally_preempted = scheduler._preempt_or_recompute(
            request,
            1.5,
        )

    assert not locally_preempted
    scheduler.kv_cache_manager.get_block_ids.assert_not_called()
    warning.assert_called_once()
    upstream_preempt.assert_not_called()
    scheduler.finish_requests.assert_called_once_with(
        "req-1",
        RequestStatus.FINISHED_ABORTED,
    )
    assert scheduler._recomputed_reqs == [RecomputeReqInfo("req-1", 2)]


def test_preempt_offload_failure_sends_request_back_to_p():
    connector = MagicMock()
    connector.update_state_before_preempt.return_value = False
    scheduler = _make_preempt_scheduler(connector=connector)
    request = SimpleNamespace(
        request_id="req-1",
        client_index=0,
        num_computed_tokens=17,
    )
    scheduler.finish_requests = MagicMock(return_value=[request])

    with (
        patch.object(Scheduler, "_preempt_request") as upstream_preempt,
        patch("vllm_ascend.core.recompute_scheduler.logger.warning") as warning,
    ):
        locally_preempted = scheduler._preempt_or_recompute(
            request,
            1.5,
        )

    assert not locally_preempted
    warning.assert_called_once()
    upstream_preempt.assert_not_called()
    scheduler.finish_requests.assert_called_once()


def test_preempt_offload_failure_aborts_under_oproj_tp(_oproj_tp_config):
    _oproj_tp_config.oproj_tensor_parallel_size = 2
    connector = MagicMock()
    connector.update_state_before_preempt.return_value = False
    scheduler = _make_preempt_scheduler(connector=connector)
    request = SimpleNamespace(
        request_id="req-1",
        client_index=0,
        num_computed_tokens=17,
    )
    scheduler.finish_requests = MagicMock(return_value=[request])

    with (
        patch.object(Scheduler, "_preempt_request") as upstream_preempt,
        patch("vllm_ascend.core.recompute_scheduler.logger.error") as error,
    ):
        locally_preempted = scheduler._preempt_or_recompute(
            request,
            1.5,
        )

    assert not locally_preempted
    error.assert_called_once()
    upstream_preempt.assert_not_called()
    scheduler.finish_requests.assert_called_once_with(
        "req-1",
        RequestStatus.FINISHED_ABORTED,
    )
    # Not recorded for recomputation: the request must not be sent back to P.
    assert scheduler._recomputed_reqs == []


def test_reset_preemption_skips_offload():
    connector = MagicMock()
    scheduler = _make_preempt_scheduler(connector=connector)
    request = SimpleNamespace(request_id="req-1", num_computed_tokens=17)

    with patch.object(Scheduler, "_preempt_request") as upstream_preempt:
        scheduler._preempt_request(request, 1.5, drop_stale_output=True)

    connector.update_state_before_preempt.assert_not_called()
    scheduler.kv_cache_manager.get_block_ids.assert_not_called()
    upstream_preempt.assert_called_once_with(
        request,
        1.5,
        drop_stale_output=True,
    )


def test_preempt_without_computed_kv_still_uses_recompute_fallback():
    connector = MagicMock()
    connector.update_state_before_preempt.return_value = False
    scheduler = _make_preempt_scheduler(connector=connector)
    request = SimpleNamespace(
        request_id="req-1",
        client_index=0,
        num_computed_tokens=0,
    )

    scheduler.finish_requests = MagicMock(return_value=[request])

    with patch.object(Scheduler, "_preempt_request") as upstream_preempt:
        locally_preempted = scheduler._preempt_or_recompute(
            request,
            1.5,
        )

    assert not locally_preempted
    connector.update_state_before_preempt.assert_called_once_with(
        request,
        ([3, 4],),
        0,
    )
    upstream_preempt.assert_not_called()
    scheduler.finish_requests.assert_called_once()


def test_preempt_offload_exception_uses_recompute_fallback():
    connector = MagicMock()
    connector.update_state_before_preempt.side_effect = RuntimeError("offload failed")
    scheduler = _make_preempt_scheduler(connector=connector)
    request = SimpleNamespace(
        request_id="req-1",
        client_index=0,
        num_computed_tokens=17,
    )
    scheduler.finish_requests = MagicMock(return_value=[request])

    with (
        patch.object(Scheduler, "_preempt_request") as upstream_preempt,
        patch("vllm_ascend.core.recompute_scheduler.logger.warning") as warning,
    ):
        locally_preempted = scheduler._preempt_or_recompute(request, 1.5)

    assert not locally_preempted
    warning.assert_called_once()
    upstream_preempt.assert_not_called()
    scheduler.finish_requests.assert_called_once()


def test_recomputed_output_precedes_regular_output():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler_output = SimpleNamespace(recomputed_reqs=[RecomputeReqInfo("recomputed-request", 0)])
    regular_output = EngineCoreOutput(
        request_id="regular-request",
        new_token_ids=[1],
    )
    upstream_outputs = {0: EngineCoreOutputs(outputs=[regular_output])}

    with patch.object(
        Scheduler,
        "update_from_output",
        return_value=upstream_outputs,
    ):
        outputs = scheduler.update_from_output(
            scheduler_output,
            MagicMock(),
        )

    assert [output.request_id for output in outputs[0].outputs] == [
        "recomputed-request",
        "regular-request",
    ]
    assert outputs[0].outputs[0].stop_reason == "recomputed"


def test_recompute_scheduler_variants_keep_offload_preemption():
    assert issubclass(AsyncRecomputeScheduler, RecomputeScheduler)
    assert issubclass(DyntraLBRecomputeScheduler, RecomputeScheduler)
    assert issubclass(DyntraLBRecomputeScheduler, DyntraLBPolicyMixin)
    assert issubclass(AsyncDyntraLBRecomputeScheduler, AsyncRecomputeScheduler)
    assert issubclass(AsyncDyntraLBRecomputeScheduler, DyntraLBPolicyMixin)


def _make_schedule_test_scheduler():
    config = make_dyntra_test_config()
    config.kv_transfer_config = None
    return create_dyntra_lb_scheduler(config, scheduler_cls=RecomputeScheduler)


@pytest.mark.parametrize("policy", [SchedulingPolicy.FCFS, SchedulingPolicy.PRIORITY])
@pytest.mark.parametrize("pending_connector_free", [False, True])
def test_schedule_does_not_preempt_when_blocks_cannot_be_reused(policy, pending_connector_free):
    scheduler = _make_schedule_test_scheduler()
    requests = [create_request(request_id=i) for i in (1, 2)]
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    scheduler.update_from_output(output, create_model_runner_output(requests))
    scheduler.policy = policy
    scheduler.defer_block_free = True
    scheduler.processed_step_seq = 0
    for request in requests:
        # Test the connector guard and the in-flight-step guard independently.
        request.last_sched_seq = 0 if pending_connector_free else 1
    connector = MagicMock()
    connector.has_pending_block_frees.return_value = pending_connector_free
    scheduler.connector = connector
    original_running = list(scheduler.running)
    original_blocks = [scheduler.kv_cache_manager.get_block_ids(r.request_id) for r in requests]

    with (
        patch.object(scheduler.kv_cache_manager, "allocate_slots", return_value=None),
        patch.object(scheduler, "_preempt_or_recompute") as preempt,
    ):
        output = scheduler.schedule()

    preempt.assert_not_called()
    connector.update_state_before_preempt.assert_not_called()
    assert scheduler.running == original_running
    assert [scheduler.kv_cache_manager.get_block_ids(r.request_id) for r in requests] == original_blocks
    assert all(r.status == RequestStatus.RUNNING for r in requests)
    assert not output.num_scheduled_tokens
    assert not output.recomputed_reqs


@pytest.mark.parametrize("resume_with_retained_kv", [False, True])
def test_schedule_waits_for_encoder_cache_on_running_and_resumed_requests(resume_with_retained_kv):
    scheduler = _make_schedule_test_scheduler()
    request = create_request(request_id=1)
    scheduler.add_request(request)
    output = scheduler.schedule()
    scheduler.update_from_output(output, create_model_runner_output([request]))
    if resume_with_retained_kv:
        scheduler.running.remove(request)
        request.status = RequestStatus.PREEMPTED
        scheduler.waiting.add_request(request)
    else:
        # Async running requests must exclude output placeholders from the offset.
        request.num_output_placeholders = 1
    request.mm_features = [MagicMock()]
    connector = MagicMock()
    connector.ensure_cache_available.return_value = False
    scheduler.ec_connector = connector
    original_computed = request.num_computed_tokens

    with patch.object(scheduler.kv_cache_manager, "allocate_slots") as allocate:
        output = scheduler.schedule()

    allocate.assert_not_called()
    expected_offset = original_computed - request.num_output_placeholders
    connector.ensure_cache_available.assert_called_once_with(request, expected_offset)
    assert request.num_computed_tokens == original_computed
    assert request.request_id not in output.num_scheduled_tokens
    if resume_with_retained_kv:
        assert request in scheduler.skipped_waiting
        assert request.status == RequestStatus.PREEMPTED
    else:
        assert request in scheduler.running
        assert request.status == RequestStatus.RUNNING


def test_first_request_preserves_ascend_speculative_padding_without_cached_tokens():
    scheduler = _make_schedule_test_scheduler()
    scheduler.num_spec_tokens = 2
    scheduler.num_lookahead_tokens = 2
    scheduler.dynamic_sd_lookup = None
    request = create_request(request_id=1, num_tokens=1)
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == 3
    assert output.scheduled_spec_decode_tokens[request.request_id] == [-1, -1]
