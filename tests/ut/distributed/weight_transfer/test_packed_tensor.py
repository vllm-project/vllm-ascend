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

import sys
import types
import weakref
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
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
        synchronize=lambda *a, **kw: None,
        # ``torch.npu.stream(s)`` returns a context manager; the fake stream
        # already implements ``__enter__``/``__exit__`` so just return it.
        stream=lambda s: s,
    )
    original_empty = torch.empty

    def _fake_empty(*args, **kwargs):
        device = kwargs.get("device")
        if getattr(device, "type", None) == "npu" or (isinstance(device, str) and device.split(":", 1)[0] == "npu"):
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    with patch.object(torch, "npu", fake_npu, create=True), patch.object(torch, "empty", _fake_empty):
        yield


@pytest.mark.parametrize(
    "device",
    ["npu", "npu:0", types.SimpleNamespace(type="npu")],
    ids=["unindexed-string", "indexed-string", "device-type"],
)
def test_cpu_fixture_redirects_all_npu_device_forms(device):
    """Guard test infrastructure against indexed/device-object allocations."""
    assert torch.empty(1, device=device).device.type == "cpu"


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


def _assert_packed_roundtrip(backend, tensors, buffer_size):
    """Check values and ownership after overwriting the transport storage."""
    buffers: list[torch.Tensor] = []
    received: list[tuple[str, torch.Tensor]] = []
    if backend == "hccl":
        group = _make_group_mock()
        packed_broadcast_producer(iter(tensors), group, 0, lambda item: item[1], buffer_size)
        payloads = iter(call.args[0].clone() for call in group.broadcast.call_args_list)
        chunk_count = group.broadcast.call_count

        def receive(packed, **kwargs):
            packed.copy_(next(payloads))
            buffers.append(packed)

        packed_broadcast_consumer(
            iter((name, (list(tensor.shape), tensor.dtype)) for name, tensor in tensors),
            types.SimpleNamespace(broadcast=receive),
            0,
            received.extend,
            buffer_size,
        )
    else:

        def export(tensor):
            buffers.append(tensor)
            return None, (0,) * 7

        with patch(f"{_MODULE}.reduce_tensor", side_effect=export):
            chunks = list(packed_npu_ipc_producer(iter(tensors), "node-0", lambda item: item[1], buffer_size))
        assert len(chunks) == 1  # IPC callers here deliberately fit in one chunk.
        chunk_count = len(chunks)
        chunk = chunks[0]
        assert isinstance(chunk, dict)  # Keep the existing IPC wire representation.
        with _install_fake_rebuild_npu_tensor(buffers[0]):
            received = packed_npu_ipc_consumer(
                chunk["ipc_handle"],
                "node-0",
                chunk["names"],
                chunk["shapes"],
                chunk["dtype_names"],
                chunk["tensor_sizes"],
                0,
            )
    for buffer in buffers:
        buffer.zero_()
    assert [name for name, _ in received] == [name for name, _ in tensors]
    for (_, actual), (_, expected) in zip(received, tensors):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
    return chunk_count


@pytest.mark.parametrize("buffer_size", [1, 8, 128])
def test_packed_broadcast_roundtrip_mixed_scalar_empty_and_unaligned(buffer_size):
    tensors = [
        ("byte", torch.tensor([7], dtype=torch.uint8)),
        ("scalar", torch.tensor(2.5)),
        ("empty", torch.empty(0, 3, dtype=torch.float16)),
        ("matrix", torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).t()),
    ]
    _assert_packed_roundtrip("hccl", tensors, buffer_size)


@pytest.mark.parametrize("empty_only", [False, True])
def test_packed_ipc_roundtrip_scalar_empty_and_unaligned(empty_only):
    tensors = [("empty", torch.empty(0, 3, dtype=torch.float16))]
    if not empty_only:
        tensors = [("byte", torch.tensor([7], dtype=torch.uint8)), ("scalar", torch.tensor(2.5)), *tensors]
    _assert_packed_roundtrip("ipc", tensors, 128)


@pytest.mark.parametrize("producer", [False, True])
def test_packed_transfer_drains_storage_before_propagating_error(producer):
    import vllm_ascend.distributed.weight_transfer.packed_tensor as packed_module

    events: list[str] = []
    pinned: list[torch.Tensor] = []

    class TrackingStream(_FakeNpuStream):
        buffer = None
        pending = None

        def synchronize(self):
            if self.pending is not None:
                events.append("drain")
                self.pending = None

    stream = TrackingStream()

    def broadcast(packed, **kwargs):
        stream.buffer = weakref.ref(packed)
        if producer:
            stream.pending = stream.buffer
            events.append("error")
            raise ValueError("transfer rejected")

    def reject_load(weights):
        # Receive-before-unpack synchronization must not satisfy this check:
        # only finally can drain work queued by the failing loader.
        stream.pending = stream.buffer
        events.append("error")
        raise ValueError("transfer rejected")

    group = types.SimpleNamespace(broadcast=broadcast)
    with (
        patch.object(torch.npu, "Stream", return_value=stream),
        patch.object(packed_module, "_UNSYNCHRONIZED_BUFFERS", pinned),
        pytest.raises(ValueError, match="transfer rejected"),
    ):
        if producer:
            packed_broadcast_producer(iter([("w", torch.ones(1))]), group, 0, lambda item: item[1], 4, 1)
        else:
            packed_broadcast_consumer(
                iter([("w", ([1], torch.float32))]),
                group,
                0,
                reject_load,
                4,
                1,
            )
    assert events == ["error", "drain"]
    assert pinned == []


def test_packed_consumer_waits_for_receive_before_unpacking():
    """An external HCCL receive must finish before torch queues unpack clones."""
    events = []
    original = torch.tensor([2.5])

    class ReceiveStream(_FakeNpuStream):
        pending = None

        def synchronize(self):
            if self.pending is not None:
                self.pending()
                self.pending = None
                events.append("ready")

    stream = ReceiveStream()

    def receive(packed, **kwargs):
        assert kwargs["stream"] is stream
        stream.pending = lambda: packed.copy_(original.view(torch.uint8))
        events.append("receive")

    def load(weights):
        assert events == ["receive", "ready"]
        assert torch.equal(weights[0][1], original)
        events.append("load")

    with patch.object(torch.npu, "Stream", return_value=stream):
        packed_broadcast_consumer(
            iter([("w", ([1], torch.float32))]), types.SimpleNamespace(broadcast=receive), 0, load, 4, 1
        )
    assert events == ["receive", "ready", "load"]


@pytest.mark.parametrize("backend", ["hccl", "ipc"])
@pytest.mark.parametrize("change", ["dtype", "numel", "none"])
def test_packed_producer_rejects_invalid_callback_result(backend, change):
    original = torch.ones(2)
    returned = {"dtype": original.to(torch.int32), "numel": original[:1], "none": None}[change]
    message = " returned.*expected dtype=.*numel=2" if change in ("dtype", "numel") else ".*must return a torch.Tensor"
    group = _make_group_mock()
    patcher, _ = _install_fake_reduce_tensor()
    with patcher, pytest.raises(ValueError, match=f"post_iter_func{message}"):
        if backend == "hccl":
            packed_broadcast_producer(iter([("w", original)]), group, 0, lambda _: returned, 16)
        else:
            list(packed_npu_ipc_producer(iter([("w", original)]), "node-0", lambda _: returned, 16))
    group.broadcast.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"device_index": -1},
        {"device_index": None},
        {"device_index": "0"},
        {"device_index": 0.0},
        {"device_index": True},
        {"ipc_handle": {"node-0": (0,)}},
        {"dtype_names": ["not_a_dtype"]},
        {"dtype_names": ["Tensor"]},
        {"names": ["w", "extra"]},
        {"shapes": [[-1]]},
        {"tensor_sizes": [-4]},
        {"tensor_sizes": [8]},
    ],
    ids=[
        "device",
        "none-device",
        "string-device",
        "float-device",
        "bool-device",
        "handle",
        "unknown-dtype",
        "non-dtype",
        "lengths",
        "shape",
        "negative-size",
        "byte-count",
    ],
)
def test_packed_ipc_consumer_rejects_invalid_metadata_before_rebuild(changes):
    kwargs: dict[str, Any] = dict(
        ipc_handle={"node-0": (0,) * 7},
        physical_npu_id="node-0",
        names=["w"],
        shapes=[[1]],
        dtype_names=["float32"],
        tensor_sizes=[4],
        device_index=0,
    )
    kwargs.update(changes)
    with _install_fake_rebuild_npu_tensor(torch.zeros(8, dtype=torch.uint8)):
        rebuild = sys.modules["torch_npu.multiprocessing.reductions"].rebuild_npu_tensor
        with pytest.raises(ValueError):
            packed_npu_ipc_consumer(**kwargs)
        rebuild.assert_not_called()


def test_packed_hccl_consumer_rejects_negative_shape_before_broadcast():
    group = _make_group_mock()
    with pytest.raises(ValueError, match="negative shape"):
        packed_broadcast_consumer(iter([("w", ([-1], torch.float32))]), group, 0, MagicMock(), 4)
    group.broadcast.assert_not_called()


@pytest.mark.parametrize("producer", [False, True])
@pytest.mark.parametrize(
    "logging_error,num_buffers",
    [
        (None, 1),
        (RuntimeError("logging failed"), 1),
        (KeyboardInterrupt(), 1),
        (SystemExit(2), 1),
        (None, 2),
    ],
)
def test_packed_drain_failure_keeps_storage_and_original_error(producer, logging_error, num_buffers):
    import vllm_ascend.distributed.weight_transfer.packed_tensor as packed_module

    class FailedStream(_FakeNpuStream):
        failed = False

        def synchronize(self):
            if self.failed:
                raise RuntimeError("drain failed")

    stream = FailedStream()
    pinned: list[torch.Tensor] = []
    original_error = ValueError("transfer rejected")

    def broadcast(*args, **kwargs):
        stream.failed = True
        raise original_error

    expected_type = ValueError if logging_error is None or isinstance(logging_error, Exception) else type(logging_error)
    with (
        patch.object(torch.npu, "Stream", return_value=stream),
        patch.object(packed_module, "_UNSYNCHRONIZED_BUFFERS", pinned),
        patch.object(packed_module.logger, "error", side_effect=logging_error) as log_error,
        pytest.raises(expected_type) as error,
    ):
        group = types.SimpleNamespace(broadcast=broadcast)
        if producer:
            packed_broadcast_producer(iter([("w", torch.ones(1))]), group, 0, lambda item: item[1], 4, num_buffers)
        else:
            packed_broadcast_consumer(iter([("w", ([1], torch.float32))]), group, 0, MagicMock(), 4, num_buffers)
    if logging_error is None or isinstance(logging_error, Exception):
        assert error.value is original_error
    else:
        assert error.value is logging_error
    assert len(pinned) == 1
    assert pinned[0].numel() == 4
    log_error.assert_called_once()
    message, retained_slots, cleanup_error = log_error.call_args.args
    assert "(slot, bytes)" in message
    assert retained_slots == [(0, 4)]
    assert str(cleanup_error) == "drain failed"
    assert not log_error.call_args.kwargs


def test_packed_ipc_consumer_rejects_payload_larger_than_rebuilt_buffer():
    with (
        _install_fake_rebuild_npu_tensor(torch.zeros(3, dtype=torch.uint8)),
        pytest.raises(ValueError, match="metadata declares 4 bytes, buffer has 3"),
    ):
        packed_npu_ipc_consumer({"node-0": (0,) * 7}, "node-0", ["w"], [[1]], ["float32"], [4], 0)


@pytest.mark.parametrize("backend", ["hccl-producer", "hccl-consumer", "ipc-producer"])
@pytest.mark.parametrize("capacity", [0, -1])
def test_packed_transfer_rejects_nonpositive_capacity(backend, capacity):
    with pytest.raises(ValueError, match="positive"):
        if backend == "hccl-producer":
            packed_broadcast_producer(iter([]), _make_group_mock(), 0, lambda item: item[1], capacity)
        elif backend == "hccl-consumer":
            packed_broadcast_consumer(iter([]), _make_group_mock(), 0, MagicMock(), capacity)
        else:
            list(packed_npu_ipc_producer(iter([]), "node-0", lambda item: item[1], capacity))


@pytest.mark.parametrize("producer", [False, True])
@pytest.mark.parametrize("num_buffers", [0, -1])
def test_packed_hccl_rejects_nonpositive_buffer_count(producer, num_buffers):
    with pytest.raises(ValueError, match="positive"):
        if producer:
            packed_broadcast_producer(iter([]), _make_group_mock(), 0, lambda item: item[1], 4, num_buffers)
        else:
            packed_broadcast_consumer(iter([]), _make_group_mock(), 0, MagicMock(), 4, num_buffers)


@pytest.mark.parametrize("num_buffers", [1, 2, 3])
def test_packed_producer_passes_packing_stream_to_broadcast(num_buffers):
    pending: list[_FakeNpuStream] = []

    class BroadcastStream(_FakeNpuStream):
        def synchronize(self):
            if pending == [self]:
                pending.clear()

    streams = [BroadcastStream() for _ in range(num_buffers)]
    active: list[_FakeNpuStream] = []

    class StreamContext:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            active.append(self.stream)

        def __exit__(self, *exc):
            active.pop()

    def broadcast(tensor, *, src, stream):
        assert active == [stream]
        assert not pending, "previous broadcast must complete before the next collective"
        pending.append(stream)

    group = types.SimpleNamespace(broadcast=MagicMock(side_effect=broadcast))
    with (
        patch.object(torch.npu, "Stream", side_effect=streams),
        patch.object(torch.npu, "stream", side_effect=StreamContext) as context,
    ):
        packed_broadcast_producer(
            iter((f"w{i}", torch.ones(1)) for i in range(7)), group, 0, lambda item: item[1], 1, num_buffers
        )
    expected = [streams[i % num_buffers] for i in range(7)]
    assert [call.kwargs["stream"] for call in group.broadcast.call_args_list] == expected
    assert [call.args[0] for call in context.call_args_list[:7]] == expected
    assert active == []
    assert pending == []


@pytest.mark.parametrize("producer", [False, True])
@pytest.mark.parametrize(
    "body_error,second_error",
    [
        (None, None),
        (None, KeyboardInterrupt()),
        (None, SystemExit(2)),
        (ValueError("body failed"), KeyboardInterrupt()),
        (ValueError("body failed"), SystemExit(2)),
        (KeyboardInterrupt(), None),
        (SystemExit(2), None),
    ],
)
def test_packed_cleanup_drains_all_slots_reports_retention_and_prioritizes_signals(producer, body_error, second_error):
    import vllm_ascend.distributed.weight_transfer.packed_tensor as packed_module

    failure = RuntimeError("cleanup failed")
    failures = [failure, second_error, None]
    drained = []
    armed = False

    class CleanupStream(_FakeNpuStream):
        def __init__(self, index):
            self.index = index

        def synchronize(self):
            if armed:
                drained.append(self.index)
                if failures[self.index] is not None:
                    raise failures[self.index]

    streams = [CleanupStream(index) for index in range(3)]
    pinned: list[torch.Tensor] = []

    def finish(*args, **kwargs):
        nonlocal armed
        armed = True
        if body_error is not None:
            raise body_error

    errors = [error for error in (body_error, *failures) if error is not None]
    expected = next((error for error in errors if not isinstance(error, Exception)), errors[0])
    group = types.SimpleNamespace(broadcast=finish if producer else lambda *args, **kwargs: None)
    with (
        patch.object(torch.npu, "Stream", side_effect=streams),
        patch.object(packed_module, "_UNSYNCHRONIZED_BUFFERS", pinned),
        patch.object(packed_module.logger, "error") as log_error,
        pytest.raises(type(expected)) as error,
    ):
        # One tensor below the threshold reaches finally without another slot wait.
        if producer:
            packed_broadcast_producer(iter([("w", torch.ones(1))]), group, 0, lambda item: item[1], 8, 3)
        else:
            packed_broadcast_consumer(iter([("w", ([1], torch.float32))]), group, 0, finish, 8, 3)
    assert error.value is expected
    assert drained == [0, 1, 2]
    assert len(pinned) == 1
    assert pinned[0].numel() == 4
    log_error.assert_called_once()
    assert log_error.call_args.args[1] == [(0, 4)]


@pytest.mark.parametrize("producer", [False, True])
def test_packed_drain_failure_does_not_retain_unused_empty_slots(producer):
    import vllm_ascend.distributed.weight_transfer.packed_tensor as packed_module

    class FailedStream(_FakeNpuStream):
        def synchronize(self):
            raise RuntimeError("stream unavailable")

    pinned: list[torch.Tensor] = []
    with (
        patch.object(torch.npu, "Stream", return_value=FailedStream()),
        patch.object(packed_module, "_UNSYNCHRONIZED_BUFFERS", pinned),
        pytest.raises(RuntimeError, match="stream unavailable"),
    ):
        if producer:
            packed_broadcast_producer(iter([]), _make_group_mock(), 0, lambda item: item[1], 4, 2)
        else:
            packed_broadcast_consumer(iter([]), _make_group_mock(), 0, MagicMock(), 4, 2)
    assert pinned == []


@pytest.mark.parametrize("backend", ["hccl", "ipc"])
def test_packed_roundtrip_odd_bf16_followed_by_unaligned_int64(backend):
    tensors = [("bf16", torch.tensor([1, 2, 3], dtype=torch.bfloat16)), ("i64", torch.tensor([7], dtype=torch.int64))]
    # Three BF16 elements put int64 at byte offset 6, not divisible by 8.
    assert tensors[0][1].numel() * tensors[0][1].element_size() == 6
    assert _assert_packed_roundtrip(backend, tensors, 32) == 1
