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

    Some attention paths need a second host-side gate. For example, SFA records
    the device event immediately before submitting the indexer op, but MemCache
    should not contend with the host runtime until the indexer op has been
    submitted. In that case wait_for_host_submit=True makes wait() block on a
    Python event before synchronizing the device event.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._event: torch.npu.Event | None = None
        self._wait_for_host_submit = False
        self._host_submit_done = threading.Event()

    def record(
        self,
        stream: torch.npu.Stream | None = None,
        wait_for_host_submit: bool = False,
    ) -> None:
        stream = stream or torch.npu.current_stream()
        event = torch.npu.Event()
        event.record(stream)
        with self._condition:
            if self._event is None:
                self._event = event
                self._wait_for_host_submit = wait_for_host_submit
                self._condition.notify_all()

    def record_host_submit_done(self) -> None:
        self._host_submit_done.set()

    def wait(self, timeout: float = 10.0) -> bool:
        with self._condition:
            while self._event is None:
                if not self._condition.wait(timeout=timeout):
                    return False
            event = self._event
            wait_for_host_submit = self._wait_for_host_submit

        if wait_for_host_submit and not self._host_submit_done.wait(timeout=timeout):
            return False

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


def record_attention_compute_start(wait_for_host_submit: bool = False) -> None:
    """Record the compute-stream boundary immediately before attention."""
    with _lock:
        gate = _attention_compute_start_gate
    if gate is not None:
        gate.record(wait_for_host_submit=wait_for_host_submit)


def record_attention_host_submit_done() -> None:
    """Open the host-side gate after critical attention ops are submitted."""
    with _lock:
        gate = _attention_compute_start_gate
    if gate is not None:
        gate.record_host_submit_done()
