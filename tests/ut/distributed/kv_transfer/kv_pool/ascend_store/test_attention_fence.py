# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import attention_fence as fence


@pytest.fixture(autouse=True)
def reset_gate(monkeypatch):
    monkeypatch.setattr(fence, "_attention_compute_start_gate", None)


def test_gate_is_lazy_and_reset_replaces_current_gate():
    fence.record_attention_compute_start()
    torch.npu.Event.assert_not_called()
    first = fence.get_attention_compute_start_gate()
    assert fence.get_attention_compute_start_gate() is first
    second = fence.reset_attention_compute_start_gate()
    assert second is not first
    assert fence.get_attention_compute_start_gate() is second
    fence.record_attention_compute_start()
    torch.npu.Event.return_value.record.assert_called_once_with(torch.npu.current_stream.return_value)


def test_first_recorded_event_remains_authoritative():
    gate = fence.AttentionComputeStartGate()
    stream, first, second = MagicMock(), MagicMock(), MagicMock()
    torch.npu.Event.side_effect = [first, second]
    gate.record(stream)
    gate.record(stream)
    assert gate.wait(timeout=0) is True
    first.record.assert_called_once_with(stream)
    second.record.assert_called_once_with(stream)
    first.synchronize.assert_called_once_with()
    second.synchronize.assert_not_called()


def test_wait_times_out_without_synchronizing(monkeypatch):
    gate = fence.AttentionComputeStartGate()
    wait = MagicMock(return_value=False)
    monkeypatch.setattr(gate._condition, "wait", wait)
    assert gate.wait(timeout=0.01) is False
    wait.assert_called_once_with(timeout=0.01)
    torch.npu.Event.assert_not_called()


def test_wait_rechecks_event_after_spurious_wakeup(monkeypatch):
    gate = fence.AttentionComputeStartGate()
    event = MagicMock()
    wakes = iter([None, event])

    def wake(**kwargs):
        gate._event = next(wakes)
        return True

    wait = MagicMock(side_effect=wake)
    monkeypatch.setattr(gate._condition, "wait", wait)
    assert gate.wait(timeout=0.01) is True
    assert wait.call_count == 2
    event.synchronize.assert_called_once_with()
