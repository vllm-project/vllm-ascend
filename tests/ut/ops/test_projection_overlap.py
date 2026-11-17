# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from vllm_ascend.ops import finegrained_tp as tp


def test_pending_buffers_follow_actual_consumer(monkeypatch):
    trace: list[str | tuple[str, object]] = []
    consumer = object()
    monkeypatch.setattr(torch.npu, "current_stream", lambda: consumer)
    source = SimpleNamespace(record_stream=lambda stream: trace.append(("source", stream)))
    output = SimpleNamespace(record_stream=lambda stream: trace.append(("output", stream)))
    pending = tp.PendingTensor(output, SimpleNamespace(wait=lambda: trace.append("wait")), source)
    assert pending.wait() is output
    assert trace == ["wait", ("source", consumer), ("output", consumer)]


def test_communication_reuses_stream_and_orders_producer_consumer(monkeypatch):
    trace = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def record_event(self):
            trace.append(f"{self.name}:record")
            return self.name

        def wait_event(self, event):
            trace.append(f"{self.name}:wait:{event}")

    main, communication = Stream("main"), Stream("communication")
    current = [main]
    created = []

    def create_stream():
        created.append(communication)
        return communication

    @contextmanager
    def switch(stream):
        previous = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch.npu, "current_stream", lambda: current[0])
    monkeypatch.setattr(torch.npu, "Stream", create_stream)
    monkeypatch.setattr(tp, "npu_stream_switch", switch)
    executor = tp.TPCommunication()
    group = SimpleNamespace(device_group=object())
    source = SimpleNamespace(record_stream=lambda stream: trace.append(f"source:{stream.name}"))
    output = SimpleNamespace(record_stream=lambda stream: trace.append(f"output:{stream.name}"))

    def op(receive, send, *, group, async_op):
        assert current[0] is communication and async_op
        trace.append("launch")
        return SimpleNamespace(wait=lambda: trace.append(f"join:{current[0].name}"))

    for _ in range(2):
        pending = executor._launch(source, output, group, op)
        trace.append("independent-compute")
        assert pending.wait() is output
    assert created == [communication]
    expected = [
        "main:record",
        "source:communication",
        "output:main",
        "communication:wait:main",
        "launch",
        "independent-compute",
        "join:main",
        "source:main",
        "output:main",
    ]
    assert trace == expected * 2
