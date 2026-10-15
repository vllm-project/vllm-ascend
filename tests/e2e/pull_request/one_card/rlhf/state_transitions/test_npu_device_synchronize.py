#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm PR #52914
#   (vllm/v1/worker/worker_base.py, vllm/v1/engine/core.py)
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
#

"""Ascend guards for the device barrier on the RL pause / wake path.

vLLM #52914 makes a completed pause promise an *idle device* and mirrors the
contract on the wake transition: ``wake_up`` now ends with a device barrier, so
an RL caller may start same-device work (e.g. weight sync) as soon as it
returns. The engine reaches that barrier through
``collective_rpc("synchronize_device")``, which on Ascend waits with
``torch.npu.synchronize()``.

These tests run on real hardware and assert the observable half of that
contract -- the NPU stream owns no queued work once the barrier has run -- using
the NPU substitute for ``torch.cuda._sleep`` described in ``npu_stream_busy``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.e2e.pull_request.one_card.rlhf.state_transitions.npu_stream_busy import (
    enqueue_npu_busy_work,
    npu_stream_idle,
)


@pytest.fixture(autouse=True)
def npu_device():
    """Select an NPU for the test process, skipping the module when there is none."""
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.npu.set_device(0)
    torch.npu.synchronize()
    yield
    torch.npu.synchronize()


def test_synchronize_device_drains_inflight_npu_work():
    """``synchronize_device()`` must not return before the NPU stream is idle."""
    from vllm_ascend.worker.worker import NPUWorker

    worker = NPUWorker.__new__(NPUWorker)

    enqueue_npu_busy_work()
    assert not npu_stream_idle(), "helper failed to leave device work in flight"

    worker.synchronize_device()

    assert npu_stream_idle()


def test_wake_up_does_not_return_before_npu_work_is_done():
    """``wake_up()`` leaves the device idle for a caller syncing weights next."""
    from vllm_ascend.worker.worker import NPUWorker

    ascend_config = SimpleNamespace(
        weight_nz_mode=0,
        rl_config=SimpleNamespace(enabled=False, sleep_mode_extra_cleanup=False),
    )

    with (
        patch("vllm_ascend.worker.worker.get_ascend_config", return_value=ascend_config),
        patch("vllm_ascend.worker.worker.CaMemAllocator"),
    ):
        worker = NPUWorker.__new__(NPUWorker)
        worker.model_runner = SimpleNamespace(model=MagicMock())
        worker._sleep_saved_buffers = {}

        enqueue_npu_busy_work()
        assert not npu_stream_idle(), "helper failed to leave device work in flight"

        worker.wake_up(tags=["weights"])

        assert npu_stream_idle(), "wake_up() returned while NPU work was still in flight"
