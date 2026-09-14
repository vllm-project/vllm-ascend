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

import regex as re
import torch

_lock = threading.RLock()
_attention_compute_start_gate: AttentionComputeStartGate | None = None
_gates: list[AttentionComputeStartGate] = []
_layer_indices: dict[str, int] = {}
_record_cursor = 0


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
    global _attention_compute_start_gate, _gates
    gate = AttentionComputeStartGate()
    with _lock:
        _attention_compute_start_gate = gate
        _gates = []
    return gate


def get_attention_compute_start_gate(layer_idx: int | None = None) -> AttentionComputeStartGate | None:
    with _lock:
        if layer_idx is not None:
            return _gates[layer_idx] if 0 <= layer_idx < len(_gates) else None
        gate = _attention_compute_start_gate
    if gate is None:
        gate = reset_attention_compute_start_gate()
    return gate


def record_attention_compute_start(layer_name: str = "") -> None:
    """Open the gate bound to this physical layer's attention boundary."""
    global _record_cursor
    with _lock:
        if not _gates:
            gate = _attention_compute_start_gate
        else:
            if layer_name:
                idx = _layer_indices.get(layer_name)
                if idx is None:
                    match = re.search(r"layers\.(\d+)", layer_name)
                    idx = int(match.group(1)) if match is not None else -1
            else:
                idx = _record_cursor
                _record_cursor += 1
            gate = _gates[idx] if 0 <= idx < len(_gates) else None
    if gate is not None:
        gate.record()


def reset_attention_compute_start_gates(num_layers: int, layer_indices: dict[str, int] | None = None) -> None:
    """Bind one gate per physical layer for an entire forward step."""
    global _gates, _layer_indices, _record_cursor, _attention_compute_start_gate
    with _lock:
        _gates = [AttentionComputeStartGate() for _ in range(num_layers)]
        _layer_indices = dict(layer_indices or {})
        _record_cursor = 0
        _attention_compute_start_gate = None
