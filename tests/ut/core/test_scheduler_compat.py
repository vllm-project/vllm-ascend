# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility and inheritance tests for Ascend synced schedulers."""

from __future__ import annotations

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
    _V033SchedulerMembers,
    with_v033_compat,
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

# Baseline MROs recorded prior to scheduler synchronization (without any
# v0.33.0 compatibility shim classes).
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

_UPSTREAM_SCHEDULER = next(
    cls
    for cls in BalanceScheduler.__mro__
    if cls.__module__ == "vllm.v1.core.sched.scheduler" and cls.__name__ == "Scheduler"
)


@pytest.mark.parametrize("scheduler_cls", _ALL_SCHEDULER_CLASSES)
def test_scheduler_mro_matches_baseline(scheduler_cls: type) -> None:
    """All concrete schedulers preserve their baseline inheritance chains."""
    compat_count = sum(cls is _V033SchedulerMembers for cls in scheduler_cls.__mro__)
    if vllm_version_is("0.33.0"):
        assert compat_count == 1
    else:
        assert compat_count == 0

    filtered = tuple(
        f"{cls.__module__}.{cls.__name__}"
        for cls in scheduler_cls.__mro__
        if cls.__module__ != "vllm_ascend.core.scheduler_compat"
    )
    assert filtered == _EXPECTED_BASELINE_MROS[scheduler_cls]


def test_with_v033_compat_factory_semantics() -> None:
    """Factory is a no-op on main and idempotent for already-wrapped bases on v0.33.0."""

    class _DummyBase:
        pass

    if vllm_version_is("0.33.0"):
        wrapped = with_v033_compat(_DummyBase)
        assert wrapped is not _DummyBase
        assert issubclass(wrapped, _V033SchedulerMembers)
        assert with_v033_compat(wrapped) is wrapped
        assert with_v033_compat(BalanceScheduler) is BalanceScheduler
    else:
        assert with_v033_compat(_DummyBase) is _DummyBase
        assert with_v033_compat(BalanceScheduler) is BalanceScheduler


@pytest.mark.skipif(
    not vllm_version_is("0.33.0"),
    reason="v0.33.0 member non-shadowing check only applies on v0.33.0",
)
def test_v033_compat_members_do_not_shadow_upstream_schedulers() -> None:
    """No non-dunder member on _V033SchedulerMembers exists on v0.33.0 Scheduler/AsyncScheduler."""
    compat_members = [
        name for name in _V033SchedulerMembers.__dict__ if not (name.startswith("__") and name.endswith("__"))
    ]
    assert compat_members

    # Inspect upstream Scheduler and AsyncScheduler own attributes (excluding
    # the injected compat shim class in AsyncScheduler.__mro__).
    upstream_classes = [
        cls
        for cls in (*_UPSTREAM_SCHEDULER.__mro__, AsyncScheduler)
        if cls.__module__ != "vllm_ascend.core.scheduler_compat" and cls is not BalanceScheduler
    ]
    for member in compat_members:
        for cls in upstream_classes:
            assert member not in cls.__dict__, (
                f"_V033SchedulerMembers.{member} would shadow {cls.__module__}.{cls.__name__}.{member}"
            )


def test_v033_compat_per_instance_async_kv_waiting_isolation() -> None:
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
