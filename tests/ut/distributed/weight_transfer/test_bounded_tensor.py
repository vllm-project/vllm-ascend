# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run real planning and copying against deferred CPU stream queues."""

import dataclasses
import types
import weakref
from collections import deque
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.distributed.weight_transfer import bounded_tensor as bounded


@pytest.fixture(autouse=True)
def base_storage_format():
    with patch.object(bounded.torch_npu, "get_npu_format", return_value=2, create=True):
        yield


class QueuedRuntime:
    def __init__(self):
        self.active = None
        self.streams = []
        self.allocations = []
        self.peak_bytes = 0
        self.copy = torch.Tensor.copy_
        self.empty = torch.empty
        self.caller = self.new_stream()

    def new_stream(self):
        runtime = self

        class Stream:
            def __init__(self):
                self.torch_queue = deque()
                self.native_queue = deque()
                self.wait_count = 0
                self.skip_waits = 0

            def synchronize(self):
                self.wait_count += 1
                if self.wait_count <= self.skip_waits:
                    return
                # torch_npu work first reaches the native stream on queue flush.
                self.native_queue.extend(self.torch_queue)
                self.torch_queue.clear()
                while self.native_queue:
                    self.native_queue.popleft()()

            def wait_stream(self, other):
                self.torch_queue.append(other.synchronize)

            def __enter__(self):
                self.previous = runtime.active
                runtime.active = self
                return self

            def __exit__(self, *args):
                runtime.active = self.previous

        stream = Stream()
        self.streams.append(stream)
        return stream

    def allocate(self, *args, **kwargs):
        if str(kwargs.get("device", "")).startswith("npu"):
            kwargs["device"] = "cpu"
        tensor = self.empty(*args, **kwargs)
        self.allocations.append((weakref.ref(tensor), tensor.nbytes))
        self.peak_bytes = max(self.peak_bytes, sum(size for ref, size in self.allocations if ref() is not None))
        return tensor

    def copy_(self, destination, source, *args, **kwargs):
        if self.active is None:
            return self.copy(destination, source, *args, **kwargs)
        self.active.torch_queue.append(lambda: self.copy(destination, source, *args, **kwargs))
        return destination

    def install(self):
        fake = types.SimpleNamespace(
            Stream=self.new_stream,
            current_stream=lambda: self.caller,
            stream=lambda stream: stream,
        )
        return (
            patch.object(torch, "npu", fake, create=True),
            patch.object(torch, "empty", self.allocate),
            patch.object(torch.Tensor, "copy_", lambda dst, src, *a, **kw: self.copy_(dst, src, *a, **kw)),
            patch.object(torch.Tensor, "record_stream", lambda *args: None),
        )


class Wire:
    def __init__(self, runtime, payloads=None):
        self.runtime = runtime
        self.payloads = [] if payloads is None else deque(payloads)
        self.collective_bytes = []

    def broadcast(self, tensor, src, stream):
        self.collective_bytes.append(tensor.nbytes)
        if isinstance(self.payloads, list):
            stream.native_queue.append(lambda: self.payloads.append(tensor.clone()))
        else:
            payload = self.payloads.popleft()
            assert tensor.nbytes == payload.nbytes
            stream.native_queue.append(lambda: self.runtime.copy(tensor, payload))


def weights(step=0):
    output = []
    for index, (dtype, shape) in enumerate(
        [
            (torch.uint8, (3,)),
            (torch.float32, ()),
            (torch.float64, (0, 4)),
            (torch.float64, (37,)),
            (torch.bfloat16, (5, 5)),
            (torch.int16, (1,)),
        ]
    ):
        count = 1
        for size in shape:
            count *= size
        raw = (torch.arange(count * dtype.itemsize) * 37 + step * 13 + index).to(torch.uint8)
        output.append((f"weight{index}", raw.view(dtype).reshape(shape)))
    return output


def metadata(items):
    return iter((name, (tensor.shape, tensor.dtype)) for name, tensor in items)


def produce(items, capacity, slots):
    runtime = QueuedRuntime()
    wire = Wire(runtime)
    npu, empty, copy, record = runtime.install()
    with npu, empty, copy, record:
        bounded.bounded_broadcast_producer(iter(items), metadata(items), wire, 0, capacity, slots)
    assert max(wire.collective_bytes, default=0) <= capacity
    assert runtime.peak_bytes <= slots * capacity
    return wire.payloads


def consume(items, payloads, capacity, slots, configure=None, callback=None):
    runtime = QueuedRuntime()
    wire = Wire(runtime, payloads)
    observed = []

    def load(batch):
        for name, tensor in batch:
            # Emulate a loader that reads asynchronously on its active stream.
            target = torch.empty_like(tensor)
            target.copy_(tensor)
            observed.append((name, target))
        if callback is not None:
            callback(batch)

    original_new_stream = runtime.new_stream

    def new_stream():
        stream = original_new_stream()
        if configure is not None:
            configure(stream, len(runtime.streams) - 1)
        return stream

    runtime.new_stream = new_stream
    npu, empty, copy, record = runtime.install()
    with npu, empty, copy, record:
        bounded.bounded_broadcast_consumer(metadata(items), wire, 0, load, capacity, slots)
    assert not wire.payloads
    assert runtime.peak_bytes <= slots * capacity + max((tensor.nbytes for _, tensor in items), default=0)
    return observed, runtime


def assert_bytes(actual, expected):
    assert [name for name, _ in actual] == [name for name, _ in expected]
    for (_, lhs), (_, rhs) in zip(actual, expected):
        assert lhs.shape == rhs.shape and lhs.dtype == rhs.dtype
        assert torch.equal(lhs.reshape(-1).view(torch.uint8), rhs.reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("capacity", [1, 8, 13, 64])
@pytest.mark.parametrize("slots", [1, 2, 3])
def test_roundtrip_and_storage_bound(capacity, slots):
    for step in range(3):
        items = weights(step)
        payloads = produce(items, capacity, slots)
        actual, _ = consume(items, payloads, capacity, slots)
        assert_bytes(actual, items)


def test_session_streams_reused_with_changed_weights():
    runtime = QueuedRuntime()
    send_streams = tuple(runtime.new_stream() for _ in range(2))
    receive_streams = tuple(runtime.new_stream() for _ in range(2))
    copy_stream = runtime.new_stream()
    npu, empty, copy, record = runtime.install()
    with npu, empty, copy, record:
        for step in range(3):
            items = weights(step)
            sending = Wire(runtime)
            bounded.bounded_broadcast_producer(iter(items), metadata(items), sending, 0, 13, 2, streams=send_streams)
            receiving = Wire(runtime, sending.payloads)
            actual = []

            def load(batch, observed=actual):
                for name, tensor in batch:
                    target = torch.empty_like(tensor)
                    target.copy_(tensor)
                    observed.append((name, target))

            bounded.bounded_broadcast_consumer(
                metadata(items), receiving, 0, load, 13, 2, streams=receive_streams, copy_stream=copy_stream
            )
            assert_bytes(actual, items)
    assert len(runtime.streams) == 6


@pytest.mark.parametrize("items", [[], [("zero", torch.empty(0, 4))]])
def test_zero_element_updates(items):
    payloads = produce(items, 17, 2)
    assert not payloads
    actual, _ = consume(items, payloads, 17, 2)
    assert_bytes(actual, items)


def test_loader_retains_independent_storage():
    items = weights()
    payloads = produce(items, 13, 2)
    retained = []
    # Retention is loader-owned and is measured separately from the transport.
    runtime = QueuedRuntime()
    wire = Wire(runtime, payloads)
    npu, empty, copy, record = runtime.install()
    with npu, empty, copy, record:
        bounded.bounded_broadcast_consumer(metadata(items), wire, 0, retained.extend, 13, 2)
    assert_bytes(retained, items)
    pointers = [tensor.data_ptr() for _, tensor in retained if tensor.numel()]
    assert len(pointers) == len(set(pointers))
    retained[0][1].fill_(0)
    assert_bytes(retained[1:], items[1:])


def test_noncontiguous_source_rejected():
    items = [("transposed", torch.arange(16).reshape(4, 4).T)]
    with pytest.raises(ValueError, match="contiguous"):
        produce(items, 13, 2)


def test_internal_source_format_rejected_before_view():
    items = [("blocked", torch.arange(16).reshape(4, 4))]
    with (
        patch.object(bounded.torch_npu, "get_npu_format", return_value=29),
        patch.object(torch.Tensor, "reshape", side_effect=AssertionError("must not convert the parameter")),
        pytest.raises(ValueError, match="source storage"),
    ):
        produce(items, 13, 2)


def test_complete_small_parameters_share_loader_calls_within_bound():
    items = [("large", torch.arange(256, dtype=torch.uint8))]
    items.extend((f"small{index}", torch.arange(32, dtype=torch.uint8)) for index in range(10))
    payloads = produce(items, 64, 2)
    batches = []
    actual, _ = consume(items, payloads, 64, 2, callback=lambda batch: batches.append([name for name, _ in batch]))
    assert batches == [["large"], [f"small{index}" for index in range(8)], ["small8", "small9"]]
    assert_bytes(actual, items)


def test_short_update_uses_unused_slot_budget_for_complete_inputs():
    items = weights()
    payloads = produce(items, 4096, 2)
    batches = []
    actual, _ = consume(items, payloads, 4096, 2, callback=lambda batch: batches.append([name for name, _ in batch]))
    assert batches == [[name for name, _ in items]]
    assert_bytes(actual, items)


@pytest.mark.parametrize("fault", ["completion", "offset", "reuse"])
def test_negative_controls_detect_corruption(fault):
    items = [("large", torch.arange(256, dtype=torch.uint8))]
    payloads = produce(items, 16, 2)

    def configure(stream, index):
        if fault == "completion" and index in (1, 2):
            stream.skip_waits = 100
        if fault == "reuse" and index == 3:
            # Preserve the final loader wait; remove the preceding reuse waits.
            stream.skip_waits = 16

    original_plan = bounded.plan_transfer

    def corrupt_plan(*args):
        for index, chunk in enumerate(original_plan(*args)):
            if index == 1:
                fragment = dataclasses.replace(chunk.fragments[0], parameter_offset=32)
                chunk = dataclasses.replace(chunk, fragments=(fragment,))
            yield chunk

    plan = corrupt_plan if fault == "offset" else original_plan
    with patch.object(bounded, "plan_transfer", plan):
        actual, _ = consume(items, payloads, 16, 2, configure=configure)
    with pytest.raises(AssertionError):
        assert_bytes(actual, items)


@pytest.mark.parametrize("error_type", [RuntimeError, GeneratorExit])
def test_callback_failure_drains_prefetched_collective(error_type):
    items = [("first", torch.arange(16)), ("second", torch.arange(16) + 20)]
    payloads = produce(items, 128, 2)
    runtime = QueuedRuntime()
    wire = Wire(runtime, payloads)

    def fail(batch):
        raise error_type("loader failed")

    npu, empty, copy, record = runtime.install()
    with npu, empty, copy, record, pytest.raises(error_type, match="loader failed"):
        bounded.bounded_broadcast_consumer(metadata(items), wire, 0, fail, 128, 2)
    assert all(not stream.native_queue and not stream.torch_queue for stream in runtime.streams)


def test_out_of_order_source_rejected():
    items = weights()
    runtime = QueuedRuntime()
    wire = Wire(runtime)
    npu, empty, copy, record = runtime.install()
    with npu, empty, copy, record, pytest.raises(ValueError, match="metadata"):
        bounded.bounded_broadcast_producer(iter(reversed(items)), metadata(items), wire, 0, 13, 2)
