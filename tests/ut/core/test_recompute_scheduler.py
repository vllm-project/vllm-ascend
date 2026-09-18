# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.request import RequestStatus

from tests.ut.core.test_dyntra_lb_scheduler import make_dyntra_test_config
from vllm_ascend.core.dyntra_lb_scheduler import DyntraLBPolicyMixin
from vllm_ascend.core.recompute_scheduler import (
    AsyncDyntraLBRecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    RecomputeReqInfo,
    RecomputeScheduler,
    RecomputeSchedulerConfig,
)


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


def test_recompute_scheduler_config_picks_sync_and_async_class():
    vllm_config = make_dyntra_test_config()

    vllm_config.scheduler_config.async_scheduling = False
    sync_config = RecomputeSchedulerConfig.initialize_from_config(vllm_config)
    assert sync_config.scheduler_cls == ("vllm_ascend.core.recompute_scheduler.RecomputeScheduler")

    vllm_config.scheduler_config.async_scheduling = True
    async_config = RecomputeSchedulerConfig.initialize_from_config(vllm_config)
    assert async_config.scheduler_cls == ("vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler")


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
