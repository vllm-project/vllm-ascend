# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reservation helpers must compose with existing imports, async hooks and SWA."""

import subprocess
import sys
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import pytest
from vllm.config import SchedulerConfig
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.request import RequestStatus

from tests.ut.core.test_scheduler_kv_reservation import (
    _add_remote_requests,
    _assert_no_blocks,
    _assert_pool_released,
    _make_scheduler,
)
from tests.ut.kv_offload.utils import create_model_runner_output
from tests.ut.patch.platform import test_patch_swa_bounded_replay_legacy as legacy_tests
from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs
from vllm_ascend.patch.platform.patch_swa_bounded_replay_legacy import apply_legacy_replay_patch
from vllm_ascend.utils import vllm_version_is


@pytest.fixture
def replay_scheduler_factory():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    apply_legacy_replay_patch(SimpleNamespace(cache_config=SimpleNamespace(swa_bounded_replay=True)))

    def make(scheduler_cls, num_blocks):
        config = SchedulerConfig.default_factory(async_scheduling=issubclass(scheduler_cls, AsyncScheduler))
        assert config.get_scheduler_cls() is scheduler_cls
        return _make_scheduler(
            scheduler_cls,
            block_size=16,
            num_blocks=num_blocks,
            num_speculative_tokens=3,
            speculative_method="mtp",
            bounded_replay=True,
            replay_window=32,
        )

    return make


@pytest.mark.skipif(vllm_version_is("0.30.0"), reason="The release SWA backport supports local caching only")
@pytest.mark.parametrize("scheduler_cls", [Scheduler, AsyncScheduler])
@pytest.mark.parametrize("num_blocks,num_admitted", [(8, 0), (9, 1), (15, 1), (16, 2)])
def test_swa_async_load_reserves_once(replay_scheduler_factory, scheduler_cls, num_blocks, num_admitted):
    scheduler = replay_scheduler_factory(scheduler_cls, num_blocks)
    assert scheduler.prefix_replay_tokens == 32
    assert scheduler.num_spec_tokens == scheduler.num_lookahead_tokens == 3
    requests = _add_remote_requests(scheduler)
    output = scheduler.schedule()
    admitted_ids = {request.request_id for request in requests[:num_admitted]}
    assert scheduler.connector.allocated == admitted_ids
    assert not output.num_scheduled_tokens
    for request in requests[:num_admitted]:
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    for request in requests[num_admitted:]:
        assert request.status == RequestStatus.WAITING
        _assert_no_blocks(scheduler, request)
    # Prompts occupy 4+2 and 3+2 blocks, each with two further speculative
    # blocks reserved. The null block makes the exact fits 9 and 16.
    free_blocks = num_blocks - 1 - sum((6, 5)[:num_admitted])
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_blocks
    scheduler.finish_requests([request.request_id for request in requests], RequestStatus.FINISHED_ABORTED)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_blocks
    scheduler.update_from_output(output, create_model_runner_output([], finished_recving=admitted_ids))
    scheduler.schedule()
    _assert_pool_released(scheduler)


@pytest.mark.skipif(vllm_version_is("0.30.0"), reason="The release SWA backport supports local caching only")
@pytest.mark.parametrize("scheduler_cls", [Scheduler, AsyncScheduler])
def test_swa_async_load_replay_and_resume(replay_scheduler_factory, scheduler_cls):
    scheduler = replay_scheduler_factory(scheduler_cls, 9)
    request, waiting = _add_remote_requests(scheduler)
    output = scheduler.schedule()
    manager = scheduler.kv_cache_manager
    full_prefix = manager.get_blocks(request.request_id).blocks[0][:]
    scheduler.update_from_output(output, create_model_runner_output([], finished_recving={request.request_id}))
    output = scheduler.schedule()
    assert not output.preempted_req_ids
    assert output.num_scheduled_tokens == {request.request_id: 32}
    assert output.scheduled_new_reqs[0].replay_start == request.replay_start == 32
    assert manager.get_blocks(request.request_id).blocks[0][: len(full_prefix)] == full_prefix
    assert all(block.ref_cnt == 1 for block in full_prefix)
    assert manager.block_pool.get_num_free_blocks() == 0
    assert request.num_output_placeholders == int(issubclass(scheduler_cls, AsyncScheduler))
    scheduler.update_from_output(output, create_model_runner_output([request]))
    assert request.num_in_flight_tokens == request.num_output_placeholders == 0
    scheduler.finish_requests(waiting.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.connector.matched_tokens[request.request_id] = 0
    scheduler.running.remove(request)
    scheduler._preempt_request(request, monotonic())
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {request.request_id: 33}
    assert resumed.scheduled_cached_reqs.replay_start == {request.request_id: 32}
    scheduler.update_from_output(resumed, create_model_runner_output([request]))
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()
    _assert_pool_released(scheduler)


@pytest.mark.skipif(not vllm_version_is("0.30.0"), reason="Main supports SWA replay with a KV connector")
@pytest.mark.parametrize("scheduler_cls", [Scheduler, AsyncScheduler])
def test_legacy_swa_keeps_connector_rejection_and_local_replay(replay_scheduler_factory, scheduler_cls):
    with pytest.raises(ValueError, match="local prefix caching only"):
        replay_scheduler_factory(scheduler_cls, 9)
    fixture = legacy_tests.replay_scheduler.__wrapped__(
        SimpleNamespace(param=(False, issubclass(scheduler_cls, AsyncScheduler)))
    )
    try:
        scheduler = next(fixture)
        assert callable(scheduler._spec_decode_step_blocks)
        legacy_tests.test_actual_scheduler_replays_across_chunks_and_preserves_shared_blocks(scheduler, 96)
    finally:
        fixture.close()


@pytest.mark.parametrize("swa_first", [False, True])
def test_early_scheduler_imports_keep_original_mro_and_reservation(swa_first):
    # In a fresh interpreter all dedicated classes capture the
    # upstream base before the normal platform patch registration.
    # This checks patch installation with a zero replay window; nonzero replay
    # and resumed payloads are covered by the tests above in startup order.
    script = """
import builtins
import importlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
original_import = builtins.__import__
deferred = []
def defer(name, globals=None, locals=None, fromlist=(), level=0):
    module = original_import(name, globals, locals, fromlist, level)
    if name == 'vllm_ascend.utils' and 'adapt_patch' in (fromlist or ()) and not deferred:
        deferred.append((module, module.adapt_patch))
        module.adapt_patch = Mock()
    return module
with patch.object(builtins, '__import__', defer):
    runpy.run_path(sys.argv[1])
utils, adapt_patch = deferred[0]
utils.adapt_patch = adapt_patch
assert 'vllm_ascend.patch.platform' not in sys.modules
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
classes = [Scheduler, AsyncScheduler]
for module_name in ('dyntra_lb_scheduler', 'recompute_scheduler',
                    'scheduler_profiling_chunk', 'batch_job_aware_scheduler',
                    'short_request_first_scheduler'):
    module = importlib.import_module('vllm_ascend.core.' + module_name)
    classes.extend(cls for cls in vars(module).values()
                   if isinstance(cls, type) and cls.__module__ == module.__name__
                   and issubclass(cls, Scheduler))
mros = [cls.__mro__ for cls in classes]
if sys.argv[2] == 'True' and not hasattr(Scheduler, '_mark_prefix_replay'):
    legacy = runpy.run_path(str(Path(sys.argv[1]).parents[2] /
                               'vllm_ascend/patch/platform/patch_swa_bounded_replay_legacy.py'))
    legacy['_patch_scheduler']()
    legacy['_patch_cache']()
schedule = Scheduler.schedule
adapt_patch(is_global_patch=True)
assert Scheduler.schedule is schedule
assert [cls.__mro__ for cls in classes] == mros
from vllm_ascend.patch.platform.patch_swa_bounded_replay_legacy import apply_legacy_replay_patch
apply_legacy_replay_patch(SimpleNamespace(cache_config=SimpleNamespace(swa_bounded_replay=True)))
from tests.ut.core.test_scheduler_kv_reservation import _make_scheduler, _add_remote_requests
from vllm.v1.request import RequestStatus
for cls in classes:
    assert cls._request_remaining_blocks is Scheduler._request_remaining_blocks
    for blocks, admitted in ((23, False), (24, True)):
        scheduler = _make_scheduler(cls, num_blocks=blocks, bounded_replay=True)
        request, other = _add_remote_requests(scheduler)
        scheduler.finish_requests(other.request_id, RequestStatus.FINISHED_ABORTED)
        scheduler.schedule()
        assert (request.status == RequestStatus.WAITING_FOR_REMOTE_KVS) == admitted, cls
"""
    conftest = Path(__file__).resolve().parents[1] / "conftest.py"
    result = subprocess.run(
        [sys.executable, "-c", script, str(conftest), str(swa_first)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
