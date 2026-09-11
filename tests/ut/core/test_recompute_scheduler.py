# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from tests.ut.core.test_dyntra_lb_scheduler import make_dyntra_test_config
from vllm_ascend.core.dyntra_lb_scheduler import DyntraLBPolicyMixin
from vllm_ascend.core.recompute_scheduler import (
    AsyncDyntraLBRecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    RecomputeScheduler,
    RecomputeSchedulerConfig,
)


def _make_preempt_scheduler(*, connector=None):
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.connector = connector
    scheduler.kv_cache_manager = MagicMock()
    scheduler.kv_cache_manager.get_block_ids.return_value = ([3, 4],)
    return scheduler


def test_recompute_scheduler_keeps_local_schedule_for_ascend_spec_padding():
    assert RecomputeScheduler.schedule is not Scheduler.schedule
    assert RecomputeScheduler.update_from_output is Scheduler.update_from_output


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
    request = SimpleNamespace(request_id="req-1", num_computed_tokens=17)

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
        scheduler._preempt_request(request, 1.5)

    scheduler.kv_cache_manager.get_block_ids.assert_called_once_with("req-1")
    upstream_preempt.assert_called_once_with(
        request,
        1.5,
        drop_stale_output=False,
    )


@pytest.mark.parametrize("connector", [None, MagicMock(spec=[])])
def test_preempt_without_offload_hook_falls_back_to_local_recompute(connector):
    scheduler = _make_preempt_scheduler(connector=connector)
    request = SimpleNamespace(request_id="req-1", num_computed_tokens=17)

    with (
        patch.object(Scheduler, "_preempt_request") as upstream_preempt,
        patch("vllm_ascend.core.recompute_scheduler.logger.warning") as warning,
    ):
        scheduler._preempt_request(request, 1.5)

    scheduler.kv_cache_manager.get_block_ids.assert_not_called()
    warning.assert_called_once()
    upstream_preempt.assert_called_once_with(
        request,
        1.5,
        drop_stale_output=False,
    )


def test_preempt_offload_failure_falls_back_to_local_recompute():
    connector = MagicMock()
    connector.update_state_before_preempt.return_value = False
    scheduler = _make_preempt_scheduler(connector=connector)
    request = SimpleNamespace(request_id="req-1", num_computed_tokens=17)

    with (
        patch.object(Scheduler, "_preempt_request") as upstream_preempt,
        patch("vllm_ascend.core.recompute_scheduler.logger.warning") as warning,
    ):
        scheduler._preempt_request(request, 1.5)

    warning.assert_called_once()
    upstream_preempt.assert_called_once_with(
        request,
        1.5,
        drop_stale_output=False,
    )


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


def test_preempt_without_computed_kv_skips_offload():
    connector = MagicMock()
    scheduler = _make_preempt_scheduler(connector=connector)
    request = SimpleNamespace(request_id="req-1", num_computed_tokens=0)

    with patch.object(Scheduler, "_preempt_request") as upstream_preempt:
        scheduler._preempt_request(request, 1.5)

    connector.update_state_before_preempt.assert_not_called()
    scheduler.kv_cache_manager.get_block_ids.assert_not_called()
    upstream_preempt.assert_called_once_with(
        request,
        1.5,
        drop_stale_output=False,
    )


def test_recompute_scheduler_variants_keep_offload_preemption():
    assert issubclass(AsyncRecomputeScheduler, RecomputeScheduler)
    assert issubclass(DyntraLBRecomputeScheduler, RecomputeScheduler)
    assert issubclass(DyntraLBRecomputeScheduler, DyntraLBPolicyMixin)
    assert issubclass(AsyncDyntraLBRecomputeScheduler, AsyncRecomputeScheduler)
    assert issubclass(AsyncDyntraLBRecomputeScheduler, DyntraLBPolicyMixin)
