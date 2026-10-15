#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""Unit tests for packed tensor utilities (HCCL broadcast + NPU IPC).

These tests run on CPU hosts by stubbing out the NPU-specific primitives
(``torch.npu.Stream``, ``torch.npu.current_stream``, ``torch.npu.synchronize``)
and ``reduce_tensor``/``rebuild_npu_tensor``. The packing/unpacking logic
itself (``torch.cat``/``torch.split``/``view``) runs on real CPU tensors so
data correctness is verified end-to-end.
"""

from __future__ import annotations

import itertools
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    _get_streams,
    packed_broadcast_consumer,
    packed_broadcast_producer,
    packed_npu_ipc_consumer,
    packed_npu_ipc_producer,
)

_MODULE = "vllm_ascend.distributed.weight_transfer.packed_tensor"


# ---------------------------------------------------------------------------
# Fixtures: stub NPU primitives so the module body runs on a CPU host.
# ---------------------------------------------------------------------------


class _FakeNpuStream:
    """A minimal stand-in for ``torch.npu.Stream``.

    ``synchronize()`` is a no-op; ``__enter__``/``__exit__`` allow the
    ``with torch.npu.stream(s):`` context manager to work.
    """

    def synchronize(self) -> None:
        return None

    def __enter__(self) -> _FakeNpuStream:
        return self

    def __exit__(self, *exc) -> None:
        return None


@pytest.fixture(autouse=True)
def _stub_torch_npu():
    """Patch every ``torch.npu.*`` reference and ``device="npu"`` the module uses.

    ``packed_tensor`` reads ``torch.npu.Stream`` / ``torch.npu.current_stream``
    at call time via attribute access on the ``torch.npu`` submodule, and creates
    buffers with ``torch.empty(..., device="npu")``.  On a pure CPU build neither
    ``torch.npu`` nor the ``"npu"`` device exists, so both must be stubbed:
    ``torch.npu`` is replaced with a fake namespace and ``torch.empty`` is wrapped
    to redirect ``device="npu"`` to ``device="cpu"`` (the packing logic under test
    is dtype/shape/byte-correctness, which is device-agnostic).
    """
    fake_npu = types.SimpleNamespace(
        Stream=_FakeNpuStream,
        current_stream=_FakeNpuStream,
        current_device=lambda: 0,
        synchronize=lambda *a, **kw: None,
        # ``torch.npu.stream(s)`` returns a context manager; the fake stream
        # already implements ``__enter__``/``__exit__`` so just return it.
        stream=lambda s: s,
    )
    original_empty = torch.empty

    def _fake_empty(*args, **kwargs):
        if kwargs.get("device") == "npu":
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    # The packed streams are cached per device for the lifetime of the process, so a
    # test has to start from an empty cache to see its own fake streams.
    _get_streams.cache_clear()
    with patch.object(torch, "npu", fake_npu, create=True), patch.object(torch, "empty", _fake_empty):
        yield
    _get_streams.cache_clear()


# ---------------------------------------------------------------------------
# packed_broadcast_producer
# ---------------------------------------------------------------------------


def _make_group_mock() -> MagicMock:
    """Return a fake process group that records every broadcast payload."""
    group = MagicMock()
    group.broadcast = MagicMock()
    return group


def test_packed_broadcast_producer_single_tensor():
    """A single small tensor is broadcast exactly once with correct bytes."""
    tensor = torch.arange(12, dtype=torch.float32)
    iterator = iter([("w", tensor)])
    group = _make_group_mock()

    packed_broadcast_producer(
        iterator=iterator,
        group=group,
        src=0,
        post_iter_func=lambda item: item[1],
    )

    assert group.broadcast.call_count == 1
    sent = group.broadcast.call_args[0][0]
    assert sent.dtype == torch.uint8
    # 12 float32 = 48 bytes
    assert sent.numel() == 48
    # Bytes match the original tensor when viewed back as float32.
    assert torch.equal(sent.view(torch.float32), tensor)


def test_packed_broadcast_producer_multiple_tensors_one_buffer():
    """Multiple tensors under the buffer size are packed into one broadcast."""
    tensors = [
        ("a", torch.full((4,), 1.0, dtype=torch.float32)),
        ("b", torch.full((8,), 2.0, dtype=torch.float32)),
    ]
    group = _make_group_mock()

    packed_broadcast_producer(
        iterator=iter(tensors),
        group=group,
        src=0,
        post_iter_func=lambda item: item[1],
    )

    assert group.broadcast.call_count == 1
    sent = group.broadcast.call_args[0][0]
    # 4 + 8 = 12 floats = 48 bytes
    assert sent.numel() == 48
    viewed = sent.view(torch.float32)
    assert torch.equal(viewed[:4], torch.full((4,), 1.0))
    assert torch.equal(viewed[4:], torch.full((8,), 2.0))


def test_packed_broadcast_producer_splits_when_exceeding_buffer():
    """Tensors larger than ``buffer_size_bytes`` trigger an extra broadcast."""
    # buffer = 32 bytes (8 float32). Two tensors of 12 floats each (48 bytes)
    # → first tensor fills past the threshold and triggers a broadcast after it.
    tensors = [
        ("a", torch.full((12,), 1.0, dtype=torch.float32)),
        ("b", torch.full((12,), 2.0, dtype=torch.float32)),
    ]
    group = _make_group_mock()

    packed_broadcast_producer(
        iterator=iter(tensors),
        group=group,
        src=0,
        post_iter_func=lambda item: item[1],
        buffer_size_bytes=32,
    )

    assert group.broadcast.call_count == 2
    first_payload = group.broadcast.call_args_list[0].args[0]
    second_payload = group.broadcast.call_args_list[1].args[0]
    # First buffer holds only the first tensor (it already exceeds the 32-byte
    # threshold, so the loop breaks right after appending it).
    assert torch.equal(first_payload.view(torch.float32), tensors[0][1])
    # Second buffer holds the second tensor.
    assert torch.equal(second_payload.view(torch.float32), tensors[1][1])


def test_packed_broadcast_producer_empty_iterator():
    """An empty iterator triggers no broadcasts and does not hang."""
    group = _make_group_mock()
    packed_broadcast_producer(
        iterator=iter([]),
        group=group,
        src=0,
        post_iter_func=lambda item: item[1],
    )
    group.broadcast.assert_not_called()


def test_packed_broadcast_producer_uses_num_buffers_to_rotate():
    """The producer cycles through ``num_buffers`` streams."""
    tensors = [(f"w{i}", torch.zeros(1, dtype=torch.float32)) for i in range(6)]
    group = _make_group_mock()
    # Small buffer so each tensor triggers its own broadcast.
    packed_broadcast_producer(
        iterator=iter(tensors),
        group=group,
        src=0,
        post_iter_func=lambda item: item[1],
        buffer_size_bytes=1,  # 1 byte → every 4-byte tensor overflows immediately
        num_buffers=2,
    )
    # Each tensor is broadcast alone because each one (4 bytes) > 1 byte buffer.
    assert group.broadcast.call_count == 6


def test_packed_broadcast_producer_passes_src_rank():
    """``src`` is forwarded to ``group.broadcast`` unchanged."""
    group = _make_group_mock()
    packed_broadcast_producer(
        iterator=iter([("w", torch.zeros(4, dtype=torch.float32))]),
        group=group,
        src=3,
        post_iter_func=lambda item: item[1],
        buffer_size_bytes=1,
    )
    assert group.broadcast.call_args.kwargs.get("src") == 3


# ---------------------------------------------------------------------------
# packed_broadcast_consumer
# ---------------------------------------------------------------------------


def test_packed_broadcast_consumer_single_tensor():
    """Consumer unpacks a single broadcasted tensor and loads it."""
    # Arrange: producer-style payload (12 float32 = 48 bytes, uint8 view)
    original = torch.arange(12, dtype=torch.float32)
    packed = original.view(torch.uint8).view(-1).clone()
    # Group returns ``packed`` on broadcast.
    group = _make_group_mock()
    group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))

    received: list[tuple[str, torch.Tensor]] = []

    packed_broadcast_consumer(
        iterator=iter([("w", ([12], torch.float32))]),
        group=group,
        src=0,
        post_unpack_func=lambda weights: received.extend(weights),
    )

    assert len(received) == 1
    name, tensor = received[0]
    assert name == "w"
    assert tensor.shape == (12,)
    assert tensor.dtype == torch.float32
    assert torch.equal(tensor.cpu(), original)


def test_packed_broadcast_consumer_multiple_tensors_one_buffer():
    """Consumer unpacks multiple tensors from one packed buffer."""
    a = torch.full((4,), 1.0, dtype=torch.float32)
    b = torch.full((8,), 2.0, dtype=torch.float32)
    packed = torch.cat([a.view(torch.uint8).view(-1), b.view(torch.uint8).view(-1)])
    group = _make_group_mock()
    group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))

    received: list[tuple[str, torch.Tensor]] = []
    packed_broadcast_consumer(
        iterator=iter([("a", ([4], torch.float32)), ("b", ([8], torch.float32))]),
        group=group,
        src=0,
        post_unpack_func=lambda weights: received.extend(weights),
    )

    assert len(received) == 2
    assert received[0][0] == "a"
    assert torch.equal(received[0][1].cpu(), a)
    assert received[1][0] == "b"
    assert torch.equal(received[1][1].cpu(), b)


def test_packed_broadcast_consumer_empty_iterator():
    """Consumer with an empty iterator triggers no broadcasts."""
    group = _make_group_mock()
    received: list = []
    packed_broadcast_consumer(
        iterator=iter([]),
        group=group,
        src=0,
        post_unpack_func=lambda weights: received.extend(weights),
    )
    group.broadcast.assert_not_called()
    assert received == []


def test_packed_broadcast_consumer_passes_src_rank():
    """``src`` is forwarded to ``group.broadcast``."""
    packed = torch.zeros(4, dtype=torch.uint8)
    group = _make_group_mock()
    group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))

    packed_broadcast_consumer(
        iterator=iter([("w", ([1], torch.float32))]),
        group=group,
        src=2,
        post_unpack_func=lambda _: None,
    )
    assert group.broadcast.call_args.kwargs.get("src") == 2


def test_packed_broadcast_consumer_handles_multiple_dtypes():
    """Consumer correctly restores tensors of different dtypes from one buffer."""
    f16 = torch.tensor([1.5, 2.5], dtype=torch.float16)
    f32 = torch.tensor([3.0, 4.0], dtype=torch.float32)
    packed = torch.cat([f16.view(torch.uint8).view(-1), f32.view(torch.uint8).view(-1)])
    group = _make_group_mock()
    group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))

    received: list[tuple[str, torch.Tensor]] = []
    packed_broadcast_consumer(
        iterator=iter([("f16", ([2], torch.float16)), ("f32", ([2], torch.float32))]),
        group=group,
        src=0,
        post_unpack_func=lambda weights: received.extend(weights),
    )

    assert received[0][1].dtype == torch.float16
    assert received[1][1].dtype == torch.float32
    assert torch.equal(received[0][1].cpu(), f16)
    assert torch.equal(received[1][1].cpu(), f32)


# ---------------------------------------------------------------------------
# Stream contract: one stream per buffer slot, carrying its whole chunk
# ---------------------------------------------------------------------------


class _QueuedStream:
    """Stream whose device work only runs when the stream is synchronized.

    NPU kernels are asynchronous, so a CPU test that wants to observe stream
    ordering has to queue the work and run it on ``synchronize()``.
    """

    def __init__(self, fake_npu, name: str, log: list | None = None) -> None:
        self.pending: list = []
        self._fake_npu = fake_npu
        self._name = name
        self._log = log
        self._previous = None

    def synchronize(self) -> None:
        if self._log is not None:
            self._log.append(("sync", self))
        while self.pending:
            self.pending.pop(0)()

    def __enter__(self) -> _QueuedStream:
        self._previous = self._fake_npu.active
        self._fake_npu.active = self
        return self

    def __exit__(self, *exc) -> None:
        self._fake_npu.active = self._previous

    def __repr__(self) -> str:
        return f"<npu stream {self._name}>"


def _queueing_npu(log: list | None = None):
    """Build a fake ``torch.npu`` whose streams queue their work.

    Returns ``(fake_npu, created)``: ``created`` collects the streams the module
    asks for through ``torch.npu.Stream()`` (its buffer slots), while
    ``fake_npu.current_stream()`` is a separate default stream until a
    ``with torch.npu.stream(slot)`` block says otherwise.
    """
    fake_npu = types.SimpleNamespace()
    created: list[_QueuedStream] = []

    def make_stream(*args, **kwargs):
        stream = _QueuedStream(fake_npu, f"s{len(created)}", log)
        created.append(stream)
        return stream

    fake_npu.active = _QueuedStream(fake_npu, "default", log)
    fake_npu.Stream = make_stream
    fake_npu.stream = lambda stream: stream
    fake_npu.current_stream = lambda: fake_npu.active
    fake_npu.current_device = lambda: 0
    return fake_npu, created


def test_packed_broadcast_streams_are_cached_across_calls():
    """A device's packed streams are reused instead of recreated per call.

    The caching allocator keys cached blocks by the stream they were allocated on,
    so a fresh stream per call strands the packed buffers' reserved memory in a pool
    nobody reuses (upstream vLLM caches its NCCL streams for the same reason,
    gh-52950). It also gives a slot one stable stream, which is what its barrier
    means anything against.
    """
    fake_npu, created = _queueing_npu()

    with patch.object(torch, "npu", fake_npu, create=True):
        first = _get_streams(0, 2)
        second = _get_streams(0, 2)
        other_device = _get_streams(1, 2)

    assert first is second, "the streams of a device have to be cached"
    assert first != other_device, "each device gets its own streams"
    assert len(created) == 4, "two streams per device, not two per call"


def test_packed_broadcast_producer_packs_and_broadcasts_on_the_slot_stream():
    """Packing a chunk and its collective both run on that slot's stream.

    A slot's buffer is replaced - and its memory handed back to the allocator - the
    next time the slot comes around, right after that stream is barred. The barrier
    therefore only covers the broadcast if the broadcast ran on the same stream.
    """
    log: list = []
    fake_npu, created = _queueing_npu(log)
    real_cat = torch.cat

    def recording_cat(tensors, dim=0):
        log.append(("pack", fake_npu.active))
        return real_cat(tensors, dim=dim)

    group = _make_group_mock()
    group.broadcast = MagicMock(side_effect=lambda tensor, **kwargs: log.append(("broadcast", kwargs.get("stream"))))
    chunks = [(str(index), torch.full((4,), index, dtype=torch.uint8)) for index in range(3)]

    with patch.object(torch, "npu", fake_npu, create=True), patch.object(torch, "cat", recording_cat):
        packed_broadcast_producer(
            iterator=iter(chunks),
            group=group,
            src=0,
            post_iter_func=lambda item: item[1],
            buffer_size_bytes=1,  # every tensor overflows -> one chunk per tensor
            num_buffers=2,
        )

    assert len(created) == 2
    packs = [entry for entry in log if entry[0] == "pack"]
    broadcasts = [entry for entry in log if entry[0] == "broadcast"]
    assert len(packs) == len(broadcasts) == len(chunks)
    for index, (pack, broadcast) in enumerate(zip(packs, broadcasts)):
        slot_stream = created[index % 2]
        assert pack == ("pack", slot_stream)
        # ``stream=None`` would resolve to a process-wide stream instead.
        assert broadcast == ("broadcast", slot_stream)

    slot0_syncs = [position for position, entry in enumerate(log) if entry == ("sync", created[0])]
    assert slot0_syncs[1] > log.index(broadcasts[0]), "the slot's barrier has to cover its broadcast"
    assert slot0_syncs[1] < log.index(packs[2]), "and has to run before the buffer is reused"
    assert {entry[1] for entry in log[-2:]} == {created[0], created[1]}, "every slot is drained before returning"


def test_packed_broadcast_producer_reuses_a_slot_buffer_only_after_its_broadcast_ran():
    """A slot's packed buffer is only overwritten once its broadcast has run.

    ``torch.cat`` hands the same block back to a slot on every chunk here - which
    is what the caching allocator does with a freed block - and the broadcast reads
    that block whenever the device gets to it. If the packing of a slot's next chunk
    runs while the broadcast of its previous chunk is still queued, the workers
    receive a window of the wrong chunk's bytes.
    """
    num_buffers = 2
    chunks = [torch.arange(4, dtype=torch.uint8) + 10 * index for index in range(2 * num_buffers + 1)]
    arena = [torch.zeros(4, dtype=torch.uint8) for _ in range(num_buffers)]
    arena_calls = itertools.count()
    real_cat = torch.cat
    received: list[torch.Tensor] = []

    fake_npu, _ = _queueing_npu()

    def arena_cat(tensors, dim=0):
        slot = next(arena_calls) % num_buffers
        arena[slot].copy_(real_cat(tensors, dim=dim))
        return arena[slot]

    def queue_broadcast(tensor, **kwargs):
        stream = kwargs.get("stream") or fake_npu.current_stream()

        def read():
            received.append(tensor.clone())

        stream.pending.append(read)

    group = _make_group_mock()
    group.broadcast = MagicMock(side_effect=queue_broadcast)

    with patch.object(torch, "npu", fake_npu, create=True), patch.object(torch, "cat", arena_cat):
        packed_broadcast_producer(
            iterator=iter([(str(index), chunk) for index, chunk in enumerate(chunks)]),
            group=group,
            src=0,
            post_iter_func=lambda item: item[1],
            buffer_size_bytes=1,
            num_buffers=num_buffers,
        )

    assert len(received) == len(chunks)
    assert sorted(tuple(tensor.tolist()) for tensor in received) == sorted(tuple(chunk.tolist()) for chunk in chunks)


def test_packed_broadcast_consumer_receives_and_loads_on_the_slot_stream():
    """Receiving a chunk and loading it both run on that slot's stream.

    The loads of a chunk are queued behind its broadcast, and the next broadcast is
    ordered behind this one by the communicator's own launch order, so the loaders a
    later chunk triggers for an earlier chunk's retained tensors run after those
    tensors were loaded.
    """
    log: list = []
    fake_npu, created = _queueing_npu(log)
    payloads = iter([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])])
    group = _make_group_mock()
    group.broadcast = MagicMock(
        side_effect=lambda tensor, **kwargs: (
            log.append(("broadcast", kwargs.get("stream"))),
            tensor.copy_(next(payloads).view(torch.uint8)),
        )
    )
    loaded: list[torch.Tensor] = []

    def post_unpack(weights):
        log.append(("load", fake_npu.active))
        loaded.extend(tensor for _, tensor in weights)

    with patch.object(torch, "npu", fake_npu, create=True):
        packed_broadcast_consumer(
            iterator=iter([(str(index), ([2], torch.float32)) for index in range(3)]),
            group=group,
            src=0,
            post_unpack_func=post_unpack,
            buffer_size_bytes=1,  # every tensor overflows -> one chunk per tensor
            num_buffers=2,
        )

    assert [tensor.tolist() for tensor in loaded] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert len(created) == 2
    broadcasts = [entry for entry in log if entry[0] == "broadcast"]
    loads = [entry for entry in log if entry[0] == "load"]
    assert len(broadcasts) == len(loads) == 3
    for index, (broadcast, load) in enumerate(zip(broadcasts, loads)):
        slot_stream = created[index % 2]
        assert broadcast == ("broadcast", slot_stream)
        assert load == ("load", slot_stream)

    slot0_syncs = [position for position, entry in enumerate(log) if entry == ("sync", created[0])]
    assert slot0_syncs[1] > log.index(broadcasts[0]), "the slot's barrier has to cover its broadcast"
    assert slot0_syncs[1] < log.index(broadcasts[2]), "and the loads that read its buffer"
    assert {entry[1] for entry in log[-2:]} == {created[0], created[1]}, "every slot is drained before returning"


# ---------------------------------------------------------------------------
# packed_npu_ipc_producer
# ---------------------------------------------------------------------------


def _install_fake_reduce_tensor():
    """Patch ``reduce_tensor`` to return a sentinel func + fake args tuple.

    Returns ``(patcher, args_sentinel)`` so tests can assert what got stored.
    """
    args_sentinel = ("uuid", "size", 0, 0, 0, 0, 7, None)
    fake_reduce = MagicMock(return_value=("rebuild_func", args_sentinel))
    patcher = patch(f"{_MODULE}.reduce_tensor", fake_reduce)
    return patcher, args_sentinel


def test_packed_npu_ipc_producer_single_chunk():
    """A small tensor fits in one chunk; one dict is yielded."""
    patcher, args_sentinel = _install_fake_reduce_tensor()
    with patcher:
        tensor = torch.full((4,), 1.5, dtype=torch.float32)
        chunks = list(
            packed_npu_ipc_producer(
                iterator=iter([("w", tensor)]),
                npu_uuid="node-0",
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
            )
        )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["names"] == ["w"]
    assert chunk["shapes"] == [[4]]
    assert chunk["dtype_names"] == ["float32"]
    assert chunk["tensor_sizes"] == [16]  # 4 float32 = 16 bytes
    assert chunk["ipc_handle"] == {"node-0": args_sentinel}


def test_packed_npu_ipc_producer_empty_iterator():
    """An empty iterator yields no chunks."""
    patcher, _ = _install_fake_reduce_tensor()
    with patcher:
        chunks = list(
            packed_npu_ipc_producer(
                iterator=iter([]),
                npu_uuid="node-0",
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
            )
        )
    assert chunks == []


def test_packed_npu_ipc_producer_splits_when_exceeding_buffer():
    """Tensors that don't fit together are split across multiple chunks."""
    patcher, args_sentinel = _install_fake_reduce_tensor()
    with patcher:
        tensors = [
            ("a", torch.full((4,), 1.0, dtype=torch.float32)),  # 16 bytes
            ("b", torch.full((4,), 2.0, dtype=torch.float32)),  # 16 bytes
        ]
        chunks = list(
            packed_npu_ipc_producer(
                iterator=iter(tensors),
                npu_uuid="node-0",
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=20,  # Only one 16-byte tensor fits per chunk
            )
        )

    assert len(chunks) == 2
    # First chunk holds the first tensor only.
    assert chunks[0]["names"] == ["a"]
    assert chunks[0]["tensor_sizes"] == [16]
    # Second chunk holds the second tensor.
    assert chunks[1]["names"] == ["b"]
    assert chunks[1]["tensor_sizes"] == [16]
    # Every chunk carries the same IPC handle (reusable single buffer).
    assert all(c["ipc_handle"] == {"node-0": args_sentinel} for c in chunks)


def test_packed_npu_ipc_producer_raises_when_single_tensor_exceeds_buffer():
    """A single tensor larger than the buffer raises ``ValueError``."""
    patcher, _ = _install_fake_reduce_tensor()
    with patcher:
        big = torch.zeros(64, dtype=torch.float32)  # 256 bytes
        with pytest.raises(ValueError, match="exceeds buffer_size_bytes"):
            list(
                packed_npu_ipc_producer(
                    iterator=iter([("big", big)]),
                    npu_uuid="node-0",
                    post_iter_func=lambda item: item[1],
                    buffer_size_bytes=128,
                )
            )


def test_packed_npu_ipc_producer_dtype_name_extraction():
    """``dtype_names`` strips the ``torch.`` prefix from ``str(dtype)``."""
    patcher, _ = _install_fake_reduce_tensor()
    with patcher:
        tensors = [
            ("f16", torch.zeros(1, dtype=torch.float16)),
            ("bf16", torch.zeros(1, dtype=torch.bfloat16)),
            ("i64", torch.zeros(1, dtype=torch.int64)),
        ]
        chunks = list(
            packed_npu_ipc_producer(
                iterator=iter(tensors),
                npu_uuid="node-0",
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
            )
        )
    assert len(chunks) == 1
    assert chunks[0]["dtype_names"] == ["float16", "bfloat16", "int64"]


# ---------------------------------------------------------------------------
# packed_npu_ipc_consumer
# ---------------------------------------------------------------------------


def _install_fake_rebuild_npu_tensor(rebuilt: torch.Tensor):
    """Install a fake ``torch_npu.multiprocessing.reductions`` module."""
    fake_mod = types.ModuleType("torch_npu.multiprocessing.reductions")
    fake_mod.rebuild_npu_tensor = MagicMock(return_value=rebuilt)  # type: ignore[attr-defined]
    return patch.dict(
        "sys.modules",
        {
            "torch_npu.multiprocessing": types.ModuleType("torch_npu.multiprocessing"),
            "torch_npu.multiprocessing.reductions": fake_mod,
        },
    )


def test_packed_npu_ipc_consumer_roundtrip():
    """Consumer rebuilds the buffer and slices it into the named tensors."""
    # Two float32 tensors of size 2 and 3 → 8 + 12 = 20 bytes
    a = torch.tensor([1.0, 2.0], dtype=torch.float32)
    b = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
    packed = torch.cat([a.view(torch.uint8).view(-1), b.view(torch.uint8).view(-1)])
    # ``rebuild_npu_tensor`` returns the packed buffer; consumer slices it.
    patcher = _install_fake_rebuild_npu_tensor(packed)
    fake_args = ("uuid", 20, 0, 0, 0, 0, 7, None)

    with patcher:
        weights = packed_npu_ipc_consumer(
            ipc_handle={"node-0": fake_args},
            physical_npu_id="node-0",
            names=["a", "b"],
            shapes=[[2], [3]],
            dtype_names=["float32", "float32"],
            tensor_sizes=[8, 12],
            device_index=2,
        )

    assert len(weights) == 2
    assert weights[0][0] == "a"
    assert torch.equal(weights[0][1], a)
    assert weights[1][0] == "b"
    assert torch.equal(weights[1][1], b)


def test_packed_npu_ipc_consumer_missing_uuid_raises():
    """An unknown NPU UUID raises ``ValueError``."""
    packed = torch.zeros(4, dtype=torch.uint8)
    patcher = _install_fake_rebuild_npu_tensor(packed)
    with patcher, pytest.raises(ValueError, match="IPC handle not found for NPU UUID"):
        packed_npu_ipc_consumer(
            ipc_handle={"other-node": ("x",)},
            physical_npu_id="node-0",
            names=["w"],
            shapes=[[1]],
            dtype_names=["float32"],
            tensor_sizes=[4],
            device_index=0,
        )


def test_packed_npu_ipc_consumer_overwrites_device_index():
    """The receiver's ``device_index`` is written into ``args[6]`` before rebuild."""
    packed = torch.zeros(4, dtype=torch.uint8)
    patcher = _install_fake_rebuild_npu_tensor(packed)

    sender_args = ("uuid", 4, 0, 0, 0, 0, 7, None)
    with patcher:
        packed_npu_ipc_consumer(
            ipc_handle={"node-0": sender_args},
            physical_npu_id="node-0",
            names=["w"],
            shapes=[[1]],
            dtype_names=["float32"],
            tensor_sizes=[4],
            device_index=5,
        )
        # Capture the args passed to rebuild_npu_tensor while patch is active.
        import sys

        fake_mod = sys.modules["torch_npu.multiprocessing.reductions"]
        fake_rebuild = fake_mod.rebuild_npu_tensor  # type: ignore[attr-defined]
        rebuilt_args = fake_rebuild.call_args.args
    # Index 6 must be overwritten from 7 → 5.
    assert rebuilt_args[6] == 5


def test_packed_npu_ipc_consumer_clones_slices():
    """Consumer returns independent storage so producer reuse is safe."""
    packed = torch.zeros(8, dtype=torch.uint8)
    patcher = _install_fake_rebuild_npu_tensor(packed)
    # Args tuple must have at least 7 elements (index 6 = device_index).
    fake_args = ("uuid", 8, 0, 0, 0, 0, 0)
    with patcher:
        weights = packed_npu_ipc_consumer(
            ipc_handle={"node-0": fake_args},
            physical_npu_id="node-0",
            names=["a", "b"],
            shapes=[[1], [1]],
            dtype_names=["float32", "float32"],
            tensor_sizes=[4, 4],
            device_index=0,
        )
    # Each returned tensor must have its own storage (not a view into ``packed``).
    assert weights[0][1].storage().data_ptr() != weights[1][1].storage().data_ptr()


def test_packed_npu_ipc_consumer_truncates_to_content_size():
    """Consumer slices the rebuilt buffer to ``sum(tensor_sizes)``."""
    # Packed buffer has 16 bytes, but only 4 are meaningful.
    packed = torch.zeros(16, dtype=torch.uint8)
    packed[:4].copy_(torch.tensor([1.0], dtype=torch.float32).view(torch.uint8).view(-1))
    patcher = _install_fake_rebuild_npu_tensor(packed)
    # Args tuple must have at least 7 elements (index 6 = device_index).
    fake_args = ("uuid", 16, 0, 0, 0, 0, 0)
    with patcher:
        weights = packed_npu_ipc_consumer(
            ipc_handle={"node-0": fake_args},
            physical_npu_id="node-0",
            names=["w"],
            shapes=[[1]],
            dtype_names=["float32"],
            tensor_sizes=[4],
            device_index=0,
        )
    assert torch.equal(weights[0][1].cpu(), torch.tensor([1.0], dtype=torch.float32))


# ---------------------------------------------------------------------------
# Additional coverage: post_iter_func transforms, edge cases, roundtrip
# ---------------------------------------------------------------------------


def test_packed_broadcast_producer_post_iter_func_transforms_tensor():
    """``post_iter_func`` may transform the tensor before packing.

    The producer passes each ``(name, tensor)`` item through ``post_iter_func``
    and packs the returned tensor.  A non-identity transform (e.g. ``t * 2``)
    must be applied before the bytes hit the wire.
    """
    original = torch.full((4,), 1.0, dtype=torch.float32)
    group = _make_group_mock()

    packed_broadcast_producer(
        iterator=iter([("w", original)]),
        group=group,
        src=0,
        post_iter_func=lambda item: item[1] * 2,
    )

    sent = group.broadcast.call_args[0][0]
    assert torch.equal(sent.view(torch.float32), torch.full((4,), 2.0))


def test_packed_broadcast_producer_handles_non_contiguous_tensor():
    """The producer calls ``.contiguous()`` on each tensor before packing.

    A non-contiguous tensor (e.g. a transposed view) must be packed using its
    logical byte layout, not its strides.
    """
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    non_contig = base.t()  # transpose → non-contiguous
    assert not non_contig.is_contiguous()
    group = _make_group_mock()

    packed_broadcast_producer(
        iterator=iter([("w", non_contig)]),
        group=group,
        src=0,
        post_iter_func=lambda item: item[1],
    )

    sent = group.broadcast.call_args[0][0].view(torch.float32).view(*non_contig.shape)
    assert torch.equal(sent, non_contig.contiguous())


def test_packed_broadcast_producer_tensor_exactly_fills_buffer():
    """A tensor whose byte size equals ``buffer_size_bytes`` is broadcast alone.

    The producer splits when ``packing_tensor_sizes > target_packed_tensor_size``
    (strict greater-than), so a tensor that exactly equals the threshold stays
    in the current buffer.
    """
    tensor = torch.full((4,), 3.0, dtype=torch.float32)  # 16 bytes
    group = _make_group_mock()

    packed_broadcast_producer(
        iterator=iter([("w", tensor)]),
        group=group,
        src=0,
        post_iter_func=lambda item: item[1],
        buffer_size_bytes=16,
    )

    assert group.broadcast.call_count == 1
    sent = group.broadcast.call_args[0][0]
    assert sent.numel() == 16
    assert torch.equal(sent.view(torch.float32), tensor)


def test_packed_broadcast_consumer_unpacks_multiple_dtypes_in_one_buffer():
    """Consumer restores fp16 and int32 tensors from the same packed buffer.

    Mixed dtypes in one buffer must each restore to the correct dtype and shape.
    """
    f16 = torch.tensor([1.5, 2.5], dtype=torch.float16)
    i32 = torch.tensor([10, 20, 30], dtype=torch.int32)
    packed = torch.cat([f16.view(torch.uint8).view(-1), i32.view(torch.uint8).view(-1)])
    group = _make_group_mock()
    group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))

    received: list[tuple[str, torch.Tensor]] = []
    packed_broadcast_consumer(
        iterator=iter([("f16", ([2], torch.float16)), ("i32", ([3], torch.int32))]),
        group=group,
        src=0,
        post_unpack_func=lambda weights: received.extend(weights),
    )

    assert received[0][1].dtype == torch.float16
    assert torch.equal(received[0][1].cpu(), f16)
    assert received[1][1].dtype == torch.int32
    assert torch.equal(received[1][1].cpu(), i32)


def test_packed_broadcast_consumer_restores_multi_dim_shape():
    """Consumer restores a 2-D tensor from its packed 1-D representation.

    ``view(*shape)`` must reconstruct the original multi-dimensional layout.
    """
    original = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    packed = original.view(torch.uint8).view(-1).clone()
    group = _make_group_mock()
    group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))

    received: list[tuple[str, torch.Tensor]] = []
    packed_broadcast_consumer(
        iterator=iter([("w", ([2, 3], torch.float32))]),
        group=group,
        src=0,
        post_unpack_func=lambda weights: received.extend(weights),
    )

    assert received[0][1].shape == (2, 3)
    assert torch.equal(received[0][1].cpu(), original)


def test_packed_broadcast_consumer_handles_bfloat16():
    """Consumer correctly restores bfloat16 tensors.

    bfloat16 has 2-byte elements; the unpacker must use the correct itemsize
    when slicing and the correct dtype when viewing.
    """
    original = torch.tensor([1.5, -0.5, 2.25], dtype=torch.bfloat16)
    packed = original.view(torch.uint8).view(-1).clone()
    group = _make_group_mock()
    group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(packed))

    received: list[tuple[str, torch.Tensor]] = []
    packed_broadcast_consumer(
        iterator=iter([("w", ([3], torch.bfloat16))]),
        group=group,
        src=0,
        post_unpack_func=lambda weights: received.extend(weights),
    )

    assert received[0][1].dtype == torch.bfloat16
    assert torch.equal(received[0][1].cpu(), original)


# ---------------------------------------------------------------------------
# packed_broadcast roundtrip: producer + consumer data integrity
# ---------------------------------------------------------------------------


def test_packed_broadcast_roundtrip_single_tensor():
    """Producer → consumer roundtrip preserves tensor bytes and metadata."""
    original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    group = _make_group_mock()

    captured: list[torch.Tensor] = []
    group.broadcast = MagicMock(side_effect=lambda t, **kw: captured.append(t))
    packed_broadcast_producer(
        iterator=iter([("w", original)]),
        group=group,
        src=0,
        post_iter_func=lambda item: item[1],
    )
    group.broadcast = MagicMock(side_effect=lambda t, **kw: t.copy_(captured[0]))

    received: list[tuple[str, torch.Tensor]] = []
    packed_broadcast_consumer(
        iterator=iter([("w", ([3], torch.float32))]),
        group=group,
        src=0,
        post_unpack_func=lambda weights: received.extend(weights),
    )

    assert received[0][0] == "w"
    assert torch.equal(received[0][1].cpu(), original)


# ---------------------------------------------------------------------------
# packed_npu_ipc_producer — additional coverage
# ---------------------------------------------------------------------------


def test_packed_npu_ipc_producer_post_iter_func_transforms_tensor():
    """``post_iter_func`` may transform the tensor before packing in IPC mode."""
    patcher, _ = _install_fake_reduce_tensor()
    with patcher:
        original = torch.full((4,), 1.0, dtype=torch.float32)
        chunks = list(
            packed_npu_ipc_producer(
                iterator=iter([("w", original)]),
                npu_uuid="node-0",
                post_iter_func=lambda item: item[1] * 3,
                buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
            )
        )

    assert len(chunks) == 1
    # Chunk metadata records the original shape/dtype, not the transformed one.
    assert chunks[0]["shapes"] == [[4]]
    assert chunks[0]["dtype_names"] == ["float32"]


def test_packed_npu_ipc_producer_handles_multiple_tensors_one_chunk():
    """Multiple small tensors fit in one IPC chunk and are yielded together."""
    patcher, _ = _install_fake_reduce_tensor()
    with patcher:
        a = torch.full((2,), 1.0, dtype=torch.float32)
        b = torch.full((3,), 2.0, dtype=torch.float32)
        chunks = list(
            packed_npu_ipc_producer(
                iterator=iter([("a", a), ("b", b)]),
                npu_uuid="node-0",
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
            )
        )

    assert len(chunks) == 1
    assert chunks[0]["names"] == ["a", "b"]
    assert chunks[0]["shapes"] == [[2], [3]]
    assert chunks[0]["tensor_sizes"] == [8, 12]


def test_packed_npu_ipc_producer_ipc_handle_key_matches_npu_uuid():
    """The ``ipc_handle`` dict key equals the ``npu_uuid`` argument.

    The consumer looks up the handle by its own UUID, so the producer must
    store it under the UUID it was given.
    """
    patcher, args_sentinel = _install_fake_reduce_tensor()
    with patcher:
        chunks = list(
            packed_npu_ipc_producer(
                iterator=iter([("w", torch.zeros(4, dtype=torch.float32))]),
                npu_uuid="physical-npu-7",
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
            )
        )

    assert "physical-npu-7" in chunks[0]["ipc_handle"]
    assert chunks[0]["ipc_handle"]["physical-npu-7"] == args_sentinel


# ---------------------------------------------------------------------------
# packed_npu_ipc_consumer — additional coverage
# ---------------------------------------------------------------------------


def test_packed_npu_ipc_consumer_restores_multi_dim_shape():
    """Consumer restores a 2-D tensor from the packed IPC buffer."""
    original = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    packed = original.view(torch.uint8).view(-1).clone()
    patcher = _install_fake_rebuild_npu_tensor(packed)
    fake_args = ("uuid", 24, 0, 0, 0, 0, 0)

    with patcher:
        weights = packed_npu_ipc_consumer(
            ipc_handle={"node-0": fake_args},
            physical_npu_id="node-0",
            names=["w"],
            shapes=[[2, 3]],
            dtype_names=["float32"],
            tensor_sizes=[24],
            device_index=0,
        )

    assert weights[0][1].shape == (2, 3)
    assert torch.equal(weights[0][1].cpu(), original)


def test_packed_npu_ipc_consumer_handles_bfloat16():
    """Consumer correctly restores bfloat16 tensors via ``getattr(torch, dn)``."""
    original = torch.tensor([1.5, -0.5, 2.25], dtype=torch.bfloat16)
    packed = original.view(torch.uint8).view(-1).clone()
    patcher = _install_fake_rebuild_npu_tensor(packed)
    fake_args = ("uuid", 6, 0, 0, 0, 0, 0)

    with patcher:
        weights = packed_npu_ipc_consumer(
            ipc_handle={"node-0": fake_args},
            physical_npu_id="node-0",
            names=["w"],
            shapes=[[3]],
            dtype_names=["bfloat16"],
            tensor_sizes=[6],
            device_index=0,
        )

    assert weights[0][1].dtype == torch.bfloat16
    assert torch.equal(weights[0][1].cpu(), original)


def test_packed_npu_ipc_consumer_clones_each_slice_independently():
    """Every returned tensor has distinct storage, even sharing one buffer.

    The producer reuses one IPC buffer across chunks, so the consumer must
    ``.clone()`` each slice to give it independent storage.
    """
    packed = torch.zeros(12, dtype=torch.uint8)
    patcher = _install_fake_rebuild_npu_tensor(packed)
    fake_args = ("uuid", 12, 0, 0, 0, 0, 0)

    with patcher:
        weights = packed_npu_ipc_consumer(
            ipc_handle={"node-0": fake_args},
            physical_npu_id="node-0",
            names=["a", "b", "c"],
            shapes=[[1], [1], [1]],
            dtype_names=["float32", "float32", "float32"],
            tensor_sizes=[4, 4, 4],
            device_index=0,
        )

    ptrs = [w[1].storage().data_ptr() for w in weights]
    assert len(set(ptrs)) == 3, f"expected 3 distinct storage ptrs, got {ptrs}"
