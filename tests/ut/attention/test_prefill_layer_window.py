# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, call

import pytest
import torch

from vllm_ascend.attention.context_parallel.prefill_layer_window import PrefillLayerWindow


def test_prefill_window_waits_for_oldest_and_recycles_events(monkeypatch):
    first, second = Mock(), Mock()
    first.query.return_value = False
    second.query.return_value = True
    calls = Mock()
    calls.attach_mock(first, "first")
    calls.attach_mock(second, "second")
    calls.record_event.side_effect = [first, second, first, second]
    monkeypatch.setattr(torch.npu, "current_stream", lambda: calls)
    window = PrefillLayerWindow(2)

    for index in range(4):
        with window.layer():
            calls.submit(index)

    assert calls.mock_calls == [
        call.submit(0),
        call.record_event(None),
        call.submit(1),
        call.record_event(None),
        call.first.query(),
        call.first.synchronize(),
        call.submit(2),
        call.record_event(first),
        call.second.query(),
        call.submit(3),
        call.record_event(second),
    ]


def test_prefill_window_preserves_oldest_event_on_wait_error(monkeypatch):
    event = Mock()
    event.query.return_value = False
    stream = Mock()
    stream.record_event.return_value = event
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stream)
    window = PrefillLayerWindow(1)
    with window.layer():
        pass

    event.synchronize.side_effect = RuntimeError("wait failed")
    with pytest.raises(RuntimeError, match="wait failed"), window.layer():
        pytest.fail("The next layer must not submit after a failed wait")

    event.synchronize.side_effect = None
    with window.layer():
        pass
    assert stream.record_event.call_args_list == [call(None), call(event)]
