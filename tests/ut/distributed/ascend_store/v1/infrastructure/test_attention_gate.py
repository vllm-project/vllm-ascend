"""Cancellation of the shared attention gate is independent of device event health."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import AttentionComputeStartGate


@pytest.mark.parametrize("record_failed", (False, True))
def test_cancel_wakes_host_waiters_without_creating_an_event(monkeypatch, record_failed):
    event = Mock(side_effect=RuntimeError("device event unavailable"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=event, current_stream=lambda: object()), raising=False)
    gate = AttentionComputeStartGate()
    gate.on_start, gate.on_finish = Mock(), Mock()
    if record_failed:
        with pytest.raises(RuntimeError, match="unavailable"):
            gate.record()
    entered = threading.Event()

    def wait():
        entered.set()
        return gate.wait(timeout=5)

    with ThreadPoolExecutor(1) as threads:
        waiting = threads.submit(wait)
        assert entered.wait(2)
        gate.cancel()
        assert waiting.result(timeout=2)
    gate.cancel()
    gate.record()
    assert gate.wait(timeout=0)
    assert gate.on_start is None and gate.on_finish is None
    assert event.call_count == int(record_failed)


def test_normal_gate_wait_still_synchronizes_the_recorded_device_event(monkeypatch):
    event, started = Mock(), Mock()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=lambda: event, current_stream=lambda: 7), raising=False)
    gate = AttentionComputeStartGate()
    gate.on_start = started
    assert not gate.wait(timeout=0)
    gate.record()
    assert gate.wait(timeout=0)
    event.record.assert_called_once_with(7)
    event.synchronize.assert_called_once()
    started.assert_called_once()
    gate.cancel()
    assert gate.wait(timeout=0)
    event.synchronize.assert_called_once()
