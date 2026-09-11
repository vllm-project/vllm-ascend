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
#

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager

import torch

_lock = threading.RLock()
_attention_compute_start_gate: AttentionComputeStartGate | None = None


class AttentionComputeStartGate:
    """Gate that opens when the compute stream reaches attention.

    The attention worker records an NPU event immediately before submitting the
    attention op. MemCache worker threads wait for that event to complete before
    submitting H2D/L2G work, so transfer starts when the compute stream is
    actually at the attention boundary rather than merely after the Python call
    site was reached.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self.on_start: Callable[[], None] | None = None
        self.on_finish: Callable[[], None] | None = None
        self._event: torch.npu.Event | None = None

    def record(
        self,
        stream: torch.npu.Stream | None = None,
    ) -> None:
        stream = stream or torch.npu.current_stream()
        event = torch.npu.Event()
        event.record(stream)
        with self._condition:
            if self._event is None:
                self._event = event
                self._condition.notify_all()
                if self.on_start is not None:
                    self.on_start()

    def cancel(self) -> None:
        """Release gated tasks after their caller has set its transfer abort flag."""
        self.on_start = None
        self.on_finish = None
        self.record()

    def wait(self, timeout: float = 10.0) -> bool:
        with self._condition:
            while self._event is None:
                if not self._condition.wait(timeout=timeout):
                    return False
            event = self._event

        event.synchronize()
        return True


def reset_attention_compute_start_gate() -> AttentionComputeStartGate:
    """Create a new per-layer gate for MemCache work.

    Layerwise prefetch tasks keep a reference to the gate that was current when
    they were submitted. The attention path opens that same gate when attention
    compute is about to be launched.
    """
    global _attention_compute_start_gate
    gate = AttentionComputeStartGate()
    with _lock:
        _attention_compute_start_gate = gate
    return gate


def get_attention_compute_start_gate() -> AttentionComputeStartGate:
    with _lock:
        gate = _attention_compute_start_gate
    if gate is None:
        gate = reset_attention_compute_start_gate()
    return gate


def record_attention_compute_start() -> None:
    """Record the compute-stream boundary immediately before attention."""
    with _lock:
        gate = _attention_compute_start_gate
    if gate is not None:
        gate.record()


@contextmanager
def attention_transfer_window():
    """Finish opted-in range transfers before subsequent communication launches.

    The event is recorded after cache writes and any preceding collectives. The
    host submits the attention kernel, then drains synchronous Mooncake range
    calls while that kernel can execute. A slow transfer delays communication
    rather than competing with it.
    """
    with _lock:
        gate = _attention_compute_start_gate
    if gate is None:
        yield
        return
    try:
        gate.record()
        yield
    finally:
        if gate.on_finish is not None:
            gate.on_finish()
