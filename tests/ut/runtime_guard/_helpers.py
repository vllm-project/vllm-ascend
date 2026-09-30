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
"""Shared UT helpers for runtime_guard (avoid cross-importing test modules)."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

from vllm_ascend.observability.runtime_guard.processor import RuntimeGuardProcessor


def bare_processor() -> RuntimeGuardProcessor:
    """Minimal ``RuntimeGuardProcessor`` shell for soft-fail / hook wiring UTs."""
    p = object.__new__(RuntimeGuardProcessor)
    p.detectors = MagicMock()
    p.detectors.after_sample_hot_path = MagicMock(side_effect=RuntimeError("boom"))
    p.detectors.check_before_sample = MagicMock(side_effect=RuntimeError("boom"))
    p.wave_tracker = None
    p.runner = None
    p._handle_alert = MagicMock()
    p._reap_finished_requests = MagicMock()
    p._last_input_batch = None
    p.action_executor = None
    # End-of-wave gate uses collectives; bare shells no-op it.
    p.runtime_config = MagicMock()
    p.end_of_wave_sync = MagicMock()  # type: ignore[method-assign]
    return p


@contextmanager
def capture_logger_text(logger_name: str, level: int = logging.INFO) -> Iterator[io.StringIO]:
    """Capture text from a leaf logger (pytest ``caplog``-safe under ascend logging).

    ``configure_ascend_logging`` / ``apply_ascend_log_level`` set
    ``vllm_ascend.propagate = False`` with a dedicated StreamHandler, so records
    never reach the root ``caplog`` handler. Attach to the leaf instead
    (same pattern as ``test_v12_logits_finite_unattributable_row_warns_…``).
    """
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(level)
    lg = logging.getLogger(logger_name)
    old_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(level)
    try:
        yield buf
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old_level)


def attach_bus_worker(proc: Any, *, name: str = "ut-bus") -> Any:
    """Start a ``DueBitsBusWorker`` on a bare processor stand-in and return it.

    Callers must ``worker.stop()`` (or use a try/finally). Merged-bus UTs must
    ``_wave_head_merged_bus`` then ``_drain_merged_bus`` — there is no sync fallback.
    """
    from vllm_ascend.observability.runtime_guard.bus_worker import DueBitsBusWorker
    from vllm_ascend.observability.runtime_guard.processor_bus import RuntimeGuardBusMixin

    worker = DueBitsBusWorker(name=name)
    worker.start()
    proc._bus_worker = worker
    proc._merged_bus_inflight = False
    proc._pending_merged_bus_dump_jobs = []
    proc._pending_merged_bus_can_dump = False
    proc._pending_merged_bus_is_first = False
    # Bind mixin helpers used by SimpleNamespace stand-ins.
    for method_name in (
        "_prepare_merged_bus_locals",
        "_wave_head_merged_bus",
        "_drain_merged_bus",
        "_apply_merged_bus_result",
    ):
        if not callable(getattr(proc, method_name, None)):
            setattr(proc, method_name, getattr(RuntimeGuardBusMixin, method_name).__get__(proc))
    return worker
