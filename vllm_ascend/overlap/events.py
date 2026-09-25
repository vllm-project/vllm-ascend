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
# This file is a part of the vllm-ascend project.
#
"""Named event slots for cross-stream dependency tracking.

Today the inter-stream dependencies between MoE stages (shared experts,
dispatch, GMM2, combine) are threaded through as bare ``torch.npu.Event``
objects created ad hoc via ``torch.npu.current_stream().record_event()``.
This ledger provides the bookkeeping layer those call sites can migrate to:
dependencies get stable names, events are created lazily on first record,
and re-recording a slot reuses the same event object instead of allocating
a new one every step.

Graph-capture safety: recording or waiting on an event while a graph is
being captured would silently bake that dependency into the graph. Until
per-capture event pooling lands, both operations are disabled while
``torch.npu.is_current_stream_capturing()`` reports capture in progress;
:meth:`EventLedger.record` then returns ``None`` and :meth:`EventLedger.wait`
returns ``False`` so callers can detect the degraded mode.

This module is API-only in this change; no existing event-bit code path is
modified yet.
"""

from __future__ import annotations

import threading

import torch


def _is_graph_capturing() -> bool:
    """Best-effort probe for NPU graph capture on the current stream."""
    probe = getattr(torch.npu, "is_current_stream_capturing", None)
    if probe is None:
        # torch_npu builds without the probe cannot be detected; assume no
        # capture so eager-mode behavior is never accidentally disabled.
        return False
    return bool(probe())


class EventLedger:
    """Owns named singleton NPU events and records/waits them lazily.

    Access the process-wide instance through :func:`get_event_ledger`.
    """

    def __init__(self) -> None:
        self._events: dict[str, torch.npu.Event] = {}
        # Guards first creation of each event; steady-state reads stay
        # lock-free like the stream registry.
        self._lock = threading.Lock()

    def record(self, name: str, stream: torch.npu.Stream | None = None) -> torch.npu.Event | None:
        """Record ``name`` on ``stream`` (default: current stream).

        Returns the recorded event, or ``None`` when graph capture is in
        progress and recording was disabled.
        """
        if _is_graph_capturing():
            return None
        event = self._events.get(name)
        if event is None:
            with self._lock:
                event = self._events.get(name)
                if event is None:
                    event = torch.npu.Event(enable_timing=False)
                    self._events[name] = event
        if stream is None:
            stream = torch.npu.current_stream()
        event.record(stream)
        return event

    def wait(self, name: str, stream: torch.npu.Stream | None = None) -> bool:
        """Make ``stream`` (default: current stream) wait on ``name``.

        Returns ``True`` when the wait was issued, ``False`` when the slot
        has never been recorded or graph capture disabled the operation.
        """
        event = self._events.get(name)
        if event is None or _is_graph_capturing():
            return False
        if stream is None:
            stream = torch.npu.current_stream()
        stream.wait_event(event)
        return True

    def get_event(self, name: str) -> torch.npu.Event | None:
        """Return the event stored under ``name`` without recording.

        Returns ``None`` when the slot has never been recorded.
        """
        return self._events.get(name)

    def clear(self, name: str | None = None) -> None:
        """Drop one event slot (``name``) or all of them (``name=None``)."""
        if name is None:
            self._events.clear()
        else:
            self._events.pop(name, None)


_LEDGER = EventLedger()


def get_event_ledger() -> EventLedger:
    """Return the process-wide :class:`EventLedger`."""
    return _LEDGER


def reset_event_ledger() -> None:
    """Drop all named event slots. Intended for tests only."""
    _LEDGER.clear()
