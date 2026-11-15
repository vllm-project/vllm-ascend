# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from tests.ut.core import test_profiling_chunk as profiling_tests
from tests.ut.core.test_dyntra_lb_scheduler import create_dyntra_lb_scheduler, make_dyntra_test_config
from tests.ut.kv_offload.utils import create_model_runner_output, create_request
from vllm_ascend.core.dyntra_lb_scheduler import AsyncDyntraLBScheduler, DyntraLBScheduler
from vllm_ascend.core.recompute_scheduler import (
    AsyncDyntraLBRecomputeScheduler,
    AsyncRecomputeScheduler,
    DyntraLBRecomputeScheduler,
    RecomputeScheduler,
)


@pytest.fixture(
    params=[
        DyntraLBScheduler,
        AsyncDyntraLBScheduler,
        RecomputeScheduler,
        AsyncRecomputeScheduler,
        DyntraLBRecomputeScheduler,
        AsyncDyntraLBRecomputeScheduler,
        profiling_tests.TestProfilingChunkScheduler,
        profiling_tests.TestProfilingChunkAsyncScheduler,
    ],
    ids=lambda cls: cls.__name__,
)
def scheduler(request):
    if request.param in (profiling_tests.TestProfilingChunkScheduler, profiling_tests.TestProfilingChunkAsyncScheduler):
        instance = request.param().create_scheduler()
    else:
        config = make_dyntra_test_config()
        config.kv_transfer_config = None
        instance = create_dyntra_lb_scheduler(config, scheduler_cls=request.param)
    # Exercise the upstream scheduling contract without loading a draft model.
    instance.use_v2_model_runner = True
    return instance


@pytest.fixture
def skip_scheduler(scheduler):
    if not hasattr(Scheduler, "_drafts_unused"):
        pytest.skip("vLLM v0.32.0 predates the unused-draft scheduling contract")
    scheduler.can_skip_drafts = True
    return scheduler


@pytest.mark.parametrize("max_tokens,expected_skip", [(1, True), (2, False)])
def test_new_request_final_prefill(skip_scheduler, max_tokens, expected_skip):
    request = create_request(request_id=1, max_tokens=max_tokens)
    skip_scheduler.add_request(request)

    output = skip_scheduler.schedule()

    assert output.num_scheduled_tokens == {request.request_id: request.num_prompt_tokens}
    assert output.skip_drafts is expected_skip


def test_nonfinal_prefill_in_waiting_and_running(skip_scheduler):
    skip_scheduler.max_num_scheduled_tokens = 8
    request = create_request(request_id=1, num_tokens=24)
    skip_scheduler.add_request(request)

    for _ in range(2):
        output = skip_scheduler.schedule()
        assert output.num_scheduled_tokens == {request.request_id: 8}
        assert output.skip_drafts
        result = create_model_runner_output([request])
        result.sampled_token_ids = [[]]
        skip_scheduler.update_from_output(output, result)

    # The final chunk needs drafts for the following decode step.
    output = skip_scheduler.schedule()
    assert output.num_scheduled_tokens == {request.request_id: 8}
    assert not output.skip_drafts


@pytest.mark.parametrize(
    "placeholders,max_tokens,expected_skip", [(0, 2, True), (0, 3, False), (1, 3, True), (1, 4, False)]
)
def test_running_request_output_limit(skip_scheduler, placeholders, max_tokens, expected_skip):
    request = create_request(request_id=1, max_tokens=max_tokens)
    skip_scheduler.add_request(request)
    prefill = skip_scheduler.schedule()
    skip_scheduler.update_from_output(prefill, create_model_runner_output([request]))
    request.num_output_placeholders = placeholders

    output = skip_scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] > 0
    assert output.skip_drafts is expected_skip


@pytest.mark.parametrize("terminal_first", [True, False])
def test_mixed_batch_still_needs_drafts(skip_scheduler, terminal_first):
    terminal = create_request(request_id=1, max_tokens=1)
    continuing = create_request(request_id=2, max_tokens=2)
    requests = [terminal, continuing] if terminal_first else [continuing, terminal]
    for request in requests:
        skip_scheduler.add_request(request)

    output = skip_scheduler.schedule()

    assert len(output.num_scheduled_tokens) == 2
    assert not output.skip_drafts


def test_empty_batch(skip_scheduler):
    output = skip_scheduler.schedule()
    assert not output.num_scheduled_tokens
    assert output.skip_drafts


@pytest.mark.parametrize("gate", [False, None])
def test_disabled_or_missing_gate_keeps_drafts(scheduler, gate):
    if gate is None:
        scheduler.__dict__.pop("can_skip_drafts", None)
    else:
        scheduler.can_skip_drafts = gate
    request = create_request(request_id=1, max_tokens=1)
    scheduler.add_request(request)
    # Old vLLM versions have neither the gate nor the helper. Neither path
    # may evaluate the helper or invent a new output field on those versions.
    with patch.object(
        scheduler, "_drafts_unused", create=True, side_effect=AssertionError("draft skipping is disabled")
    ):
        output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] > 0
    if hasattr(Scheduler, "_drafts_unused"):
        assert not output.skip_drafts
    else:
        assert not hasattr(output, "skip_drafts")
