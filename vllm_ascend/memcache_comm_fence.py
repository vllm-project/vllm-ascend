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
_gates: list[AttentionComputeStartGate] = []
# Sequential cursor for backends that record without a layer_name (e.g. the
# unified attention_v1 path). Reset to 0 at each step's gate reset.
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


def _extract_layer_index(layer_name: str) -> int | None:
    """Parse the physical layer index from a layer name.

    Mirrors pool_worker._extract_physical_layer_index for base layers. MTP and
    unparseable names return None so their record is ignored (they have no
    per-layer load gate).
    """
    m = re.search(r"layers\.(\d+)", layer_name)
    if m:
        return int(m.group(1))
    return None


def reset_attention_compute_start_gates(num_layers: int) -> None:
    """Recreate one gate per layer at the start of each forward step.

    Each layerwise load task binds the gate of the layer it loads, so a load's
    H2D copy waits for that layer's own attention boundary rather than whichever
    gate happened to be current when the task was submitted.
    """
    global _gates, _record_cursor
    with _lock:
        _gates = [AttentionComputeStartGate() for _ in range(num_layers)]
        _record_cursor = 0


def get_attention_compute_start_gate(layer_idx: int) -> AttentionComputeStartGate | None:
    """Return the gate for ``layer_idx``, or None if out of range."""
    with _lock:
        if 0 <= layer_idx < len(_gates):
            return _gates[layer_idx]
    return None


def record_attention_compute_start(layer_name: str = "") -> None:
    """Record the compute-stream boundary immediately before attention.

    The gate to open is resolved from ``layer_name``; when empty (backends that
    do not thread a name through) a sequential cursor is used, matching the
    per-layer order in which loads were submitted.
    """
    global _record_cursor
    with _lock:
        if not _gates:
            return
        if layer_name:
            idx = _extract_layer_index(layer_name)
            if idx is None:
                return
        else:
            idx = _record_cursor
            _record_cursor += 1
        if 0 <= idx < len(_gates):
            gate = _gates[idx]
        else:
            return
    gate.record()
