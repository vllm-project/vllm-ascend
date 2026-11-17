# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Condition
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops import finegrained_tp as tp


class CollectiveWorld:
    """CPU rendezvous with the same owner ordering as the real collectives."""

    def __init__(self, size):
        self.size = size
        self.condition = Condition()
        self.calls = {}

    def communication(self, rank):
        world = self

        class Communication:
            sequence = 0

            def submit(self, kind, source):
                index = self.sequence
                self.sequence += 1
                with world.condition:
                    call = world.calls.setdefault(index, {"kind": kind, "sources": {}})
                    assert call["kind"] == kind
                    call["sources"][rank] = source.clone()
                    if kind == "gather":
                        shape = (source.shape[0] * world.size, *source.shape[1:])
                    else:
                        shape = (source.shape[0] // world.size, *source.shape[1:])
                    output = source.new_empty(shape)
                    world.condition.notify_all()

                def wait():
                    with world.condition:
                        assert world.condition.wait_for(lambda: len(call["sources"]) == world.size, timeout=10)
                        sources = [call["sources"][peer] for peer in range(world.size)]
                        if kind == "gather":
                            value = torch.cat(sources)
                        else:
                            value = torch.stack(sources).sum(0).chunk(world.size)[rank]
                        output.copy_(value)

                return tp.PendingTensor(output, SimpleNamespace(wait=wait), source)

            def gather(self, source, group):
                return self.submit("gather", source)

            def scatter(self, source, group):
                return self.submit("scatter", source)

        return Communication()


@pytest.fixture(autouse=True)
def cpu_streams(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: None)


@pytest.mark.parametrize("capacity", [1, 3, 8])
@pytest.mark.parametrize("size", [2, 4])
def test_chunked_embedding_collectives_preserve_owner_rows(monkeypatch, capacity, size):
    world = CollectiveWorld(size)
    # Include an idle DP owner and unequal live rows, while communication
    # spans and ordering remain identical on every rank.
    lengths = [0] + [max(1, capacity - rank) for rank in range(1, size)]
    vocab, hidden = size * 8, 3
    weight = torch.arange(vocab * hidden, dtype=torch.float32).reshape(vocab, hidden) / 10

    def run(rank):
        group = SimpleNamespace(world_size=size, rank_in_group=rank)
        communication = world.communication(rank)
        rows = lengths[rank]
        executor = tp.ChunkedEmbeddingTP(group)
        ids = (torch.arange(capacity) + rank * 5) % vocab
        shard = vocab // size

        def lookup(ids):
            return weight[ids] * ((ids >= rank * shard) & (ids < (rank + 1) * shard)).unsqueeze(-1)

        layer = SimpleNamespace(_lookup_partition=lookup)
        expected = weight[ids][:rows]
        execute = lambda: executor.forward(layer, ids)[:rows]
        executor.communication = communication
        torch.testing.assert_close(execute(), expected)

    with ThreadPoolExecutor(max_workers=size) as pool:
        list(pool.map(run, range(size)))
    chunks = min(capacity, 2)
    assert [world.calls[index]["kind"] for index in sorted(world.calls)] == ["gather"] * chunks + ["scatter"] * chunks


def test_pending_buffers_follow_actual_consumer(monkeypatch):
    trace: list[str | tuple[str, object]] = []
    consumer = object()
    monkeypatch.setattr(torch.npu, "current_stream", lambda: consumer)
    source = SimpleNamespace(record_stream=lambda stream: trace.append(("source", stream)))
    output = SimpleNamespace(record_stream=lambda stream: trace.append(("output", stream)))
    pending = tp.PendingTensor(output, SimpleNamespace(wait=lambda: trace.append("wait")), source)
    assert pending.wait() is output
    assert trace == ["wait", ("source", consumer), ("output", consumer)]


def test_restore_copies_first_chunk_before_joining_tail(monkeypatch):
    first = torch.ones(2, 3)
    tail = torch.full((1, 3), 2.0)
    trace = []
    copy = torch.Tensor.copy_

    def record_copy(destination, source, **kwargs):
        trace.append("copy")
        return copy(destination, source, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    outputs = [
        tp.PendingTensor(first, SimpleNamespace(wait=lambda: trace.append("wait0")), first),
        tp.PendingTensor(tail, SimpleNamespace(wait=lambda: trace.append("wait1")), tail),
    ]
    torch.testing.assert_close(tp._restore_owner_chunks(outputs, 3), torch.cat((first, tail)))
    assert trace == ["wait0", "copy", "wait1", "copy"]


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
