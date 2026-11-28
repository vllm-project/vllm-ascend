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
_attention_load_gate: AttentionLoadGate | None = None
_attention_save_gate: AttentionSaveGate | None = None


class AttentionLoadGate:
    """Release future-layer loads after the current layer's load has completed.

    MemCache worker threads wait for this compute-stream event before starting
    the next layer's H2D/L2G transfer.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._event: torch.npu.Event | None = None

    def record(
        self,
        stream: torch.npu.Stream | None = None,
    ) -> None:
        with self._condition:
            if self._event is None:
                event = torch.npu.Event()
                event.record(stream or torch.npu.current_stream())
                self._event = event
                self._condition.notify_all()

    def cancel(self) -> None:
        """Release gated tasks after their caller has set its transfer abort flag."""
        self.record()

    def wait(self, timeout: float = 10.0) -> bool:
        with self._condition:
            while self._event is None:
                if not self._condition.wait(timeout=timeout):
                    return False
            event = self._event

        event.synchronize()
        return True


class AttentionSaveGate:
    """Record the existing save event once current-layer KV is ready."""

    def __init__(self) -> None:
        self.event: torch.npu.Event | None = None
        self.on_start: Callable[[], None] | None = None
        self.on_finish: Callable[[], None] | None = None
        self._recorded = False

    def record(self, stream: torch.npu.Stream | None = None) -> None:
        if self._recorded or self.event is None:
            return
        self.event.record(stream or torch.npu.current_stream())
        self._recorded = True
        if self.on_start is not None:
            self.on_start()

    def cancel(self) -> None:
        self.event = None
        self.on_start = None
        self.on_finish = None


def reset_attention_gates() -> tuple[AttentionLoadGate, AttentionSaveGate]:
    """Create independent per-layer load and save gates.

    Layerwise prefetch tasks retain their load gate after the current pair is
    replaced for the next layer.
    """
    global _attention_load_gate, _attention_save_gate
    load_gate = AttentionLoadGate()
    save_gate = AttentionSaveGate()
    with _lock:
        _attention_load_gate = load_gate
        _attention_save_gate = save_gate
    return load_gate, save_gate


def get_attention_load_gate() -> AttentionLoadGate:
    with _lock:
        gate = _attention_load_gate
    if gate is None:
        gate, _ = reset_attention_gates()
    return gate


def record_attention_prefetch_start() -> None:
    """Allow future-layer loads after the current layer's load has completed."""
    with _lock:
        gate = _attention_load_gate
    if gate is not None:
        gate.record()


def record_attention_compute_start() -> None:
    """Open any remaining load gate and record current-layer KV readiness."""
    with _lock:
        load_gate, save_gate = _attention_load_gate, _attention_save_gate
    if load_gate is not None:
        load_gate.record()
    if save_gate is not None:
        save_gate.record()


@contextmanager
def attention_transfer_window():
    """Run the configured transfer completion policy after attention submission.

    The save event is recorded after cache writes and preceding collectives.
    The connector callback applies its completion policy after attention submission.
    """
    with _lock:
        load_gate, save_gate = _attention_load_gate, _attention_save_gate
    if load_gate is None or save_gate is None:
        yield
        return
    try:
        load_gate.record()
        save_gate.record()
        yield
    finally:
        if save_gate.on_finish is not None:
            save_gate.on_finish()
