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

# mypy: ignore-errors
"""UT: manual_dump fires put seq/target into report + dump job detail."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vllm_ascend.observability.runtime_guard.state import (
    MANUAL_TRIGGER_REQ_ID,
    MANUAL_TRIGGER_TYPE,
)

from ._helpers import bare_processor


class _WatermarkCfg:
    """Minimal RuntimeConfig stand-in for ``_maybe_fire_manual_local``."""

    def __init__(self, *, target: int = 3, done: int = 0) -> None:
        self._target = target
        self._done = done
        self.consume_calls = 0

    def manual_trigger_count(self) -> int:
        return max(0, self._target - self._done)

    def manual_dumps_done(self) -> int:
        return self._done

    def manual_dump_target(self) -> int:
        return self._target

    def manual_trigger_continuous(self) -> bool:
        return False

    def dump_enabled(self) -> bool:
        return True

    def consume_manual_trigger(self) -> bool:
        self.consume_calls += 1
        if self._done >= self._target:
            return False
        self._done += 1
        return True


def test_maybe_fire_manual_local_detail_has_count_and_target():
    p = bare_processor()
    cfg = _WatermarkCfg(target=3, done=1)
    p.runtime_config = cfg
    p.wave_tracker = MagicMock()
    p.wave_tracker.current_wave.return_value = 7
    p._batch_request_io_rows = MagicMock(return_value=[("cmpl-a", 0)])
    p._handle_manual_trigger = MagicMock()
    captured_jobs: list = []
    p._run_kv_dumps = MagicMock(side_effect=lambda jobs: captured_jobs.extend(jobs))

    with (
        patch(
            "vllm_ascend.observability.runtime_guard.processor_report.should_dump_kv_on_rank",
            return_value=True,
        ),
        patch(
            "vllm_ascend.observability.runtime_guard.processor_report.is_action_leader_rank",
            return_value=True,
        ),
    ):
        p._maybe_fire_manual_local(allow_manual_dump=True)

    assert cfg.consume_calls == 1
    assert cfg.manual_dumps_done() == 2

    assert p._handle_manual_trigger.called
    event = p._handle_manual_trigger.call_args.args[0]
    assert event.trigger_type == MANUAL_TRIGGER_TYPE
    assert event.req_id == MANUAL_TRIGGER_REQ_ID
    assert event.detail["manual_dump_count"] == 2  # done+1 before consume
    assert event.detail["manual_dump_target"] == 3
    assert event.detail["source"] == "dump.manual_dump"

    assert len(captured_jobs) == 1
    assert captured_jobs[0]["detail"]["manual_dump_count"] == 2
    assert captured_jobs[0]["detail"]["manual_dump_target"] == 3
    assert captured_jobs[0]["req_id"] == "cmpl-a"
