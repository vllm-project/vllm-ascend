# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility and inheritance tests for Ascend synced schedulers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from tests.ut.core.test_dyntra_lb_scheduler import (
    create_dyntra_lb_scheduler,
    make_dyntra_test_config,
)
from tests.ut.kv_offload.utils import create_request
from vllm_ascend.core.dyntra_lb_scheduler import (
    AsyncDyntraLBScheduler,
    DyntraLBScheduler,
)
from vllm_ascend.core.recompute_scheduler import (
    AsyncDyntraLBRecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    RecomputeScheduler,
)
from vllm_ascend.core.scheduler_compat import (
    get_max_async_kv_loads_per_step,
    scheduler_output_version_kwargs,
)
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler
from vllm_ascend.utils import vllm_version_is

_ALL_SCHEDULER_CLASSES = (
    RecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    AsyncDyntraLBRecomputeScheduler,
    DyntraLBScheduler,
    AsyncDyntraLBScheduler,
    BalanceScheduler,
    AsyncScheduler,
)

# Baseline MROs preserved across both vLLM main and v0.34.0.
_EXPECTED_BASELINE_MROS: dict[type, tuple[str, ...]] = {
    RecomputeScheduler: (
        "vllm_ascend.core.recompute_scheduler.RecomputeScheduler",
        "vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler",
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm.v1.core.sched.interface.SchedulerInterface",
        "abc.ABC",
        "builtins.object",
    ),
    AsyncRecomputeScheduler: (
        "vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler",
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        "vllm_ascend.core.recompute_scheduler.RecomputeScheduler",
        "vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler",
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm.v1.core.sched.interface.SchedulerInterface",
        "abc.ABC",
        "builtins.object",
    ),
    DyntraLBRecomputeScheduler: (
        "vllm_ascend.core.recompute_scheduler.DyntraLBRecomputeScheduler",
        "vllm_ascend.core.dyntra_lb_scheduler.DyntraLBPolicyMixin",
        "vllm_ascend.core.recompute_scheduler.RecomputeScheduler",
        "vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler",
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm.v1.core.sched.interface.SchedulerInterface",
        "abc.ABC",
        "builtins.object",
    ),
    AsyncDyntraLBRecomputeScheduler: (
        "vllm_ascend.core.recompute_scheduler.AsyncDyntraLBRecomputeScheduler",
        "vllm_ascend.core.dyntra_lb_scheduler.DyntraLBPolicyMixin",
        "vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler",
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        "vllm_ascend.core.recompute_scheduler.RecomputeScheduler",
        "vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler",
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm.v1.core.sched.interface.SchedulerInterface",
        "abc.ABC",
        "builtins.object",
    ),
    DyntraLBScheduler: (
        "vllm_ascend.core.dyntra_lb_scheduler.DyntraLBScheduler",
        "vllm_ascend.core.dyntra_lb_scheduler.DyntraLBPolicyMixin",
        "vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler",
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm.v1.core.sched.interface.SchedulerInterface",
        "abc.ABC",
        "builtins.object",
    ),
    AsyncDyntraLBScheduler: (
        "vllm_ascend.core.dyntra_lb_scheduler.AsyncDyntraLBScheduler",
        "vllm_ascend.core.dyntra_lb_scheduler.DyntraLBScheduler",
        "vllm_ascend.core.dyntra_lb_scheduler.DyntraLBPolicyMixin",
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        "vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler",
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm.v1.core.sched.interface.SchedulerInterface",
        "abc.ABC",
        "builtins.object",
    ),
    BalanceScheduler: (
        "vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler",
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm.v1.core.sched.interface.SchedulerInterface",
        "abc.ABC",
        "builtins.object",
    ),
    AsyncScheduler: (
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        "vllm_ascend.patch.platform.patch_balance_schedule.BalanceScheduler",
        "vllm.v1.core.sched.scheduler.Scheduler",
        "vllm.v1.core.sched.interface.SchedulerInterface",
        "abc.ABC",
        "builtins.object",
    ),
}


@pytest.mark.parametrize("scheduler_cls", _ALL_SCHEDULER_CLASSES)
def test_scheduler_mro_matches_baseline(scheduler_cls: type) -> None:
    """All concrete schedulers preserve their exact baseline inheritance chains."""
    actual = tuple(f"{cls.__module__}.{cls.__name__}" for cls in scheduler_cls.__mro__)
    assert actual == _EXPECTED_BASELINE_MROS[scheduler_cls]


def test_get_max_async_kv_loads_per_step_version_compat() -> None:
    """Return None on v0.34.0 and read the configured field on main."""
    if vllm_version_is("0.34.0"):
        assert get_max_async_kv_loads_per_step(SimpleNamespace()) is None
    else:
        assert get_max_async_kv_loads_per_step(SimpleNamespace(max_async_kv_loads_per_step=3)) == 3


def test_scheduler_output_version_kwargs() -> None:
    """Return scheduled_encoder_input_stats on v0.34.0 and iteration_details on main."""
    sentinel_stats = object()
    sentinel_details = object()
    scheduler = SimpleNamespace(
        log_stats=True,
        observability_config=SimpleNamespace(enable_logging_iteration_details=True),
        _make_scheduled_encoder_input_stats=lambda inputs: sentinel_stats,
    )
    kwargs = scheduler_output_version_kwargs(scheduler, sentinel_details, {"req-1": [0]})
    if vllm_version_is("0.34.0"):
        assert kwargs == {"scheduled_encoder_input_stats": sentinel_stats}
    else:
        assert kwargs == {"iteration_details": sentinel_details}


def test_per_instance_async_kv_waiting_isolation() -> None:
    """Distinct scheduler instances must never share async_kv_waiting or _parked_reservations."""
    cfg1 = make_dyntra_test_config()
    cfg2 = make_dyntra_test_config()
    sched1 = create_dyntra_lb_scheduler(cfg1, scheduler_cls=DyntraLBScheduler)
    sched2 = create_dyntra_lb_scheduler(cfg2, scheduler_cls=DyntraLBScheduler)

    assert sched1.async_kv_waiting is not sched2.async_kv_waiting
    assert sched1._parked_reservations is not sched2._parked_reservations

    req = create_request(1)
    sched1.async_kv_waiting.add_request(req)
    assert list(sched1.async_kv_waiting) == [req]
    assert list(sched2.async_kv_waiting) == []
