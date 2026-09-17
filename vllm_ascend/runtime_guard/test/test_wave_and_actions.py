#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import threading
import time
from unittest.mock import MagicMock, patch

from vllm_ascend.runtime_guard.action.queue import ActionQueue
from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor
from vllm_ascend.runtime_guard.quota import DumpQuota
from vllm_ascend.runtime_guard.wave_tracker import WaveTracker


class _Cfg:
    def dump_max_times(self):
        return 2

    def dump_cooldown_seconds(self):
        return 0


def test_wave_tracker_sample_stamp():
    wt = WaveTracker()
    wt.advance(allow_arm=True)
    wt.record_sample_waves(["a", "b"])
    assert wt.take_sample_wave("a") == 1
    assert wt.take_sample_wave("a") is None


def test_dump_quota_consume():
    q = DumpQuota(_Cfg())
    assert q.can_consume()
    assert q.try_consume()
    assert q.total_count == 1


def test_processor_singleton_bind_get_reset():
    RuntimeGuardProcessor.reset_for_tests()
    assert RuntimeGuardProcessor.try_get() is None

    inits: list[object] = []
    rebinds: list[object] = []

    def _fake_init(self, runner):
        self.runner = runner
        self.action_executor = MagicMock()
        inits.append(runner)

    def _fake_rebind(self, runner):
        self.runner = runner
        rebinds.append(runner)

    def _fake_shutdown(self):
        return None

    r1, r2 = object(), object()
    with (
        patch.object(RuntimeGuardProcessor, "_init_from_runner", _fake_init),
        patch.object(RuntimeGuardProcessor, "_rebind_runner", _fake_rebind),
        patch.object(RuntimeGuardProcessor, "shutdown", _fake_shutdown),
    ):
        p1 = RuntimeGuardProcessor.bind(r1)
        p2 = RuntimeGuardProcessor.bind(r2)
        p3 = RuntimeGuardProcessor(r2)

    assert p1 is p2 is p3
    assert RuntimeGuardProcessor.get() is p1
    assert inits == [r1]
    assert rebinds == [r2, r2]
    assert p1.runner is r2

    RuntimeGuardProcessor.reset_for_tests()
    assert RuntimeGuardProcessor.try_get() is None


def test_action_queue_runs_job_async():
    done = threading.Event()
    seen: list[int] = []

    def job():
        seen.append(1)
        done.set()

    q = ActionQueue(maxsize=8, name="ut-runtime-guard-actions")
    q.start()
    try:
        q.submit(job)
        assert done.wait(timeout=2.0)
        assert seen == [1]
    finally:
        q.stop()


def test_action_queue_full_falls_back_inline():
    # B3 contract: light sinks still run inline on a full queue; heavy jobs
    # (dump_kv) are dropped instead — covered by test_v9a in regressions.
    q = ActionQueue(maxsize=1, name="ut-runtime-guard-full")
    q.start()
    try:
        gate = threading.Event()

        def blocker():
            gate.wait(timeout=2.0)

        q.submit(blocker)
        time.sleep(0.05)
        q.submit(lambda: None)
        ran: list[int] = []
        q.submit(lambda: ran.append(1))
        assert ran == [1]
        gate.set()
    finally:
        q.stop()
