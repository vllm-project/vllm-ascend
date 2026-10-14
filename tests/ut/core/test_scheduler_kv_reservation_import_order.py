# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Early class imports must not bypass speculative KV admission accounting."""

import builtins
import importlib
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

_SCHEDULERS = [
    ("dyntra_lb_scheduler", "DyntraLBScheduler"),
    ("dyntra_lb_scheduler", "AsyncDyntraLBScheduler"),
    ("recompute_scheduler", "RecomputeScheduler"),
    ("recompute_scheduler", "AsyncRecomputeScheduler"),
    ("recompute_scheduler", "DyntraLBRecomputeScheduler"),
    ("recompute_scheduler", "AsyncDyntraLBRecomputeScheduler"),
    ("scheduler_profiling_chunk", "ProfilingChunkScheduler"),
    ("scheduler_profiling_chunk", "ProfilingChunkAsyncScheduler"),
    ("batch_job_aware_scheduler", "BatchJobAwareScheduler"),
    ("batch_job_aware_scheduler", "BatchJobAwareAsyncScheduler"),
]


def _bootstrap_before_platform_registration():
    """Reuse CPU hardware mocks while deferring the real patch entry point.

    Importing utils requires conftest's torch_npu setup. Intercept only its
    completed import, so the unmodified conftest can finish its normal setup
    without eagerly installing platform patches before the class under test.
    """
    original_import = builtins.__import__
    deferred_module = None
    adapt_patch = None

    def defer_adapt_patch(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal deferred_module, adapt_patch
        module = original_import(name, globals, locals, fromlist, level)
        if name == "vllm_ascend.utils" and "adapt_patch" in (fromlist or ()) and deferred_module is None:
            deferred_module = module
            adapt_patch = module.adapt_patch
            module.adapt_patch = Mock()
        return module

    try:
        with patch.object(builtins, "__import__", new=defer_adapt_patch):
            runpy.run_path(str(Path(__file__).resolve().parents[1] / "conftest.py"))
    finally:
        if deferred_module is not None:
            deferred_module.adapt_patch = adapt_patch
    assert adapt_patch is not None
    assert "vllm_ascend.patch.platform" not in sys.modules
    return adapt_patch


def _check_early_import(module_name, class_name):
    adapt_patch = _bootstrap_before_platform_registration()
    module = importlib.import_module(f"vllm_ascend.core.{module_name}")
    scheduler_cls = getattr(module, class_name)
    # This is the class reference held by a caller before EngineArgs triggers
    # platform registration. A later module alias replacement cannot change it.
    original_mro = scheduler_cls.__mro__
    adapt_patch()
    adapt_patch(True)
    assert scheduler_cls.__mro__ == original_mro

    # Lazy imports deliberately follow registration in this isolated process.
    from vllm.v1.request import RequestStatus

    from tests.ut.core import test_scheduler_kv_reservation as fixtures
    from tests.ut.kv_offload.utils import create_model_runner_output, create_request
    from vllm_ascend.core.scheduler_kv_reservation import request_remaining_blocks

    for blocks, admitted in ((23, False), (24, True)):
        scheduler = fixtures._make_scheduler(scheduler_cls, num_blocks=blocks)
        fixtures._assert_first_async_admission(scheduler, admitted)
    # The tighter pool reproduces the original combined under-reservation;
    # one block below the exact fit also catches a stale in-flight helper even
    # if the current request's margin was already fixed independently.
    for blocks in (38, 41):
        fixtures._assert_async_load_pressure(fixtures._make_scheduler(scheduler_cls, num_blocks=blocks))

    # Both prompts and both margins exactly fit. In particular, release's
    # legacy default helper must not add the current request margin twice.
    scheduler = fixtures._make_scheduler(scheduler_cls, num_blocks=42)
    requests = fixtures._add_remote_requests(scheduler)
    output = scheduler.schedule()
    assert all(request.status == RequestStatus.WAITING_FOR_REMOTE_KVS for request in requests)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 6
    assert sum(request_remaining_blocks(scheduler, request) for request in requests) == 6
    request_ids = {request.request_id for request in requests}
    scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(output, create_model_runner_output([], finished_recving=request_ids))
    scheduler.schedule()
    fixtures._assert_pool_released(scheduler)

    if not hasattr(scheduler_cls, "_run_lb_kv_prefetch"):
        return

    # Dyntra's prefetch is a separate admission entry, ahead of schedule().
    for blocks, admitted in ((23, False), (24, True)):
        scheduler = fixtures._make_scheduler(scheduler_cls, num_blocks=blocks)
        scheduler._lb_kv_prefetch_enabled = True
        request = create_request(request_id=1, num_tokens=256, block_size=16, do_remote_prefill=True)
        scheduler.add_request(request)
        not_ready = scheduler._run_lb_kv_prefetch()
        if admitted:
            assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 3
        else:
            assert request.status == RequestStatus.WAITING
            assert request.request_id in not_ready
            fixtures._assert_no_blocks(scheduler, request)
            assert not scheduler.connector.allocated

    scheduler = fixtures._make_scheduler(scheduler_cls, num_blocks=41)
    scheduler._lb_kv_prefetch_enabled = True
    first, second = fixtures._add_remote_requests(scheduler)
    not_ready = scheduler._run_lb_kv_prefetch()
    assert first.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert second.status == RequestStatus.WAITING
    assert second.request_id in not_ready
    assert scheduler.connector.allocated == {first.request_id}
    fixtures._assert_no_blocks(scheduler, second)
    assert request_remaining_blocks(scheduler, first) == 3
    output = scheduler.schedule()
    scheduler.update_from_output(output, create_model_runner_output([], finished_recving={first.request_id}))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[first.request_id] > 0
    assert not output.preempted_req_ids
    assert second.status == RequestStatus.WAITING
    fixtures._assert_no_blocks(scheduler, second)
    scheduler.update_from_output(output, create_model_runner_output([first]))
    scheduler.finish_requests([first.request_id, second.request_id], RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()
    fixtures._assert_pool_released(scheduler)


@pytest.mark.parametrize("module_name,class_name", _SCHEDULERS, ids=[name for _, name in _SCHEDULERS])
def test_scheduler_import_before_platform_registration(module_name, class_name):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), module_name, class_name],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    # Prefer this repository's tests package to an upstream source checkout's.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    _check_early_import(*sys.argv[1:])
