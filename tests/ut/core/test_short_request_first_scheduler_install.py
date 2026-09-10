#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for installing ShortRequestFirst on a standard scheduler."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

from vllm_ascend.core.short_request_first_scheduler import (
    ShortRequestFirstAsyncScheduler,
    ShortRequestFirstRequestQueue,
    install_short_request_first_waiting_queue,
)


def make_scheduler(policy: SchedulingPolicy = SchedulingPolicy.FCFS):
    return SimpleNamespace(
        policy=policy,
        waiting=create_request_queue(policy),
    )


def make_request(
    request_id: str,
    prompt_len: int,
    computed: int = 0,
    *,
    status: RequestStatus = RequestStatus.WAITING,
    num_output_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=prompt_len,
        num_computed_tokens=computed,
        status=status,
        num_output_tokens=num_output_tokens,
    )


def test_installs_queue_on_empty_fcfs_scheduler():
    scheduler = make_scheduler()

    queue = install_short_request_first_waiting_queue(
        scheduler,
        threshold=256,
        long_max_wait_ms=0.0,
    )

    assert isinstance(queue, ShortRequestFirstRequestQueue)
    assert scheduler.waiting is queue


def test_install_is_idempotent():
    scheduler = make_scheduler()
    queue = install_short_request_first_waiting_queue(
        scheduler,
        threshold=256,
        long_max_wait_ms=0.0,
    )

    assert (
        install_short_request_first_waiting_queue(
            scheduler,
            threshold=128,
            long_max_wait_ms=1.0,
        )
        is queue
    )


def test_rejects_non_fcfs_scheduler():
    scheduler = make_scheduler(SchedulingPolicy.PRIORITY)

    with pytest.raises(ValueError, match="requires FCFS"):
        install_short_request_first_waiting_queue(
            scheduler,
            threshold=256,
            long_max_wait_ms=0.0,
        )


def test_rejects_installation_after_request_admission():
    scheduler = make_scheduler()
    scheduler.waiting.add_request(make_request("already-waiting", 10))

    with pytest.raises(RuntimeError, match="before request admission"):
        install_short_request_first_waiting_queue(
            scheduler,
            threshold=256,
            long_max_wait_ms=0.0,
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_async_scheduler_only_installs_queue_when_enabled(enabled):
    vllm_config = SimpleNamespace()
    ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            short_request_first_config=SimpleNamespace(
                enabled=enabled,
                threshold=256,
                long_max_wait_ms=0.0,
            )
        )
    )

    def initialize_async_scheduler(scheduler, *args, **kwargs):
        scheduler.vllm_config = kwargs["vllm_config"]
        scheduler.policy = SchedulingPolicy.FCFS
        scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)

    with (
        patch.object(AsyncScheduler, "__init__", initialize_async_scheduler),
        patch("vllm_ascend.ascend_config.init_ascend_config", return_value=ascend_config),
    ):
        scheduler = ShortRequestFirstAsyncScheduler(vllm_config=vllm_config)

    assert isinstance(scheduler.waiting, ShortRequestFirstRequestQueue) is enabled


@pytest.mark.parametrize(
    "scheduler_request",
    [
        make_request("preempted", 1024, status=RequestStatus.PREEMPTED),
        make_request("computed", 1024, computed=1),
        make_request("output", 1024, num_output_tokens=1),
    ],
)
def test_recovery_requests_use_immediate_lane(scheduler_request):
    scheduler = make_scheduler()
    queue = install_short_request_first_waiting_queue(
        scheduler,
        threshold=256,
        long_max_wait_ms=0.0,
    )
    queue.add_request(scheduler_request)

    assert queue.num_immediate_requests == 1
    assert queue.pop_request() is scheduler_request
