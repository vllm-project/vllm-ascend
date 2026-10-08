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

import types
import weakref
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm_ascend.distributed.weight_transfer.packed_tensor as packed_tensor_module
from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    PackedIpcChunk,
    pack_tensors,
    packed_broadcast_consumer,
    packed_broadcast_producer,
    packed_npu_ipc_consumer,
    packed_npu_ipc_producer,
    unpack_tensor,
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
        device=lambda *_args, **_kwargs: _FakeNpuStream(),
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

    with (
        patch.object(torch, "npu", fake_npu, create=True),
        patch.object(torch, "empty", _fake_empty),
        patch.object(torch.accelerator, "current_device_index", return_value=0),
    ):
        yield


def test_missing_npu_context_fails_explicitly():
    """A missing NPU context must not silently switch the transfer to CPU."""
    with (
        patch.object(torch, "npu", types.SimpleNamespace(), create=True),
        pytest.raises(AttributeError, match="device"),
    ):
        packed_tensor_module._get_streams.cache_clear()
        packed_tensor_module._get_streams(0, 1)


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
    """A small tensor fits in one chunk; one typed chunk is yielded."""
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
    assert isinstance(chunk, PackedIpcChunk)
    assert chunk.names == ["w"]
    assert chunk.shapes == [[4]]
    assert chunk.dtype_names == ["float32"]
    assert chunk.tensor_sizes == [16]  # 4 float32 = 16 bytes
    assert chunk.ipc_handle == {"node-0": args_sentinel}


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
    assert chunks[0].names == ["a"]
    assert chunks[0].tensor_sizes == [16]
    # Second chunk holds the second tensor.
    assert chunks[1].names == ["b"]
    assert chunks[1].tensor_sizes == [16]
    # Every chunk carries the same IPC handle (reusable single buffer).
    assert all(c.ipc_handle == {"node-0": args_sentinel} for c in chunks)


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
    assert chunks[0].dtype_names == ["float16", "bfloat16", "int64"]


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


def test_packed_npu_ipc_consumer_does_not_clone_owned_slices_twice():
    """The consumer returns the owned tensors produced by ``unpack_tensor``."""
    owned = MagicMock()
    importer = MagicMock()
    importer.rebuild.return_value = torch.zeros(4, dtype=torch.uint8)
    fake_args = ("uuid", 4, 0, 0, 0, 0, 0)

    with patch(f"{_MODULE}.unpack_tensor", return_value=[("w", owned)]):
        weights = packed_npu_ipc_consumer(
            ipc_handle={"node-0": fake_args},
            physical_npu_id="node-0",
            names=["w"],
            shapes=[[1]],
            dtype_names=["float32"],
            tensor_sizes=[4],
            device_index=0,
            importer=importer,
        )

    assert weights == [("w", owned)]
    owned.clone.assert_not_called()


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
    assert chunks[0].shapes == [[4]]
    assert chunks[0].dtype_names == ["float32"]


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
    assert chunks[0].names == ["a", "b"]
    assert chunks[0].shapes == [[2], [3]]
    assert chunks[0].tensor_sizes == [8, 12]


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

    assert "physical-npu-7" in chunks[0].ipc_handle
    assert chunks[0].ipc_handle["physical-npu-7"] == args_sentinel


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


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (torch.float32, 1.25),
        (torch.bfloat16, -2.5),
        (torch.int64, 7),
    ],
)
def test_pack_tensors_roundtrip_scalar(dtype, value):
    original = torch.tensor(value, dtype=dtype)
    chunk = pack_tensors(
        iter([("scalar", original)]),
        lambda item: item[1],
        buffer_size_bytes=64,
    )

    assert chunk is not None
    assert chunk.shapes == [[]]
    restored = unpack_tensor(
        chunk.packed_tensor,
        chunk.names,
        chunk.shapes,
        chunk.dtypes,
        chunk.tensor_sizes,
    )[0][1]
    assert restored.shape == torch.Size([])
    assert restored.dtype == dtype
    assert torch.equal(restored.cpu(), original.cpu())


def test_pack_tensors_rejects_post_iter_dtype_change():
    original = torch.ones(1, dtype=torch.float32)
    with pytest.raises(ValueError, match="materialized with dtype"):
        pack_tensors(
            iter([("w", original)]),
            lambda item: item[1].to(torch.int32),
            buffer_size_bytes=64,
        )


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (torch.float32, 1.25),
        (torch.bfloat16, -2.5),
        (torch.int64, 7),
    ],
)
def test_packed_npu_ipc_roundtrip_scalar(dtype, value):
    original = torch.tensor(value, dtype=dtype)
    captured = {}

    def fake_reduce(tensor):
        captured["buffer"] = tensor
        return "rebuild", ("uuid", 64, 0, 0, 0, 0, 0, None)

    with patch(f"{_MODULE}.reduce_tensor", side_effect=fake_reduce):
        chunks = list(
            packed_npu_ipc_producer(
                iterator=iter([("scalar", original)]),
                npu_uuid="node-0",
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=64,
            )
        )

    importer = MagicMock()
    importer.rebuild.return_value = captured["buffer"]
    weights = packed_npu_ipc_consumer(
        ipc_handle=chunks[0].ipc_handle,
        physical_npu_id="node-0",
        names=chunks[0].names,
        shapes=chunks[0].shapes,
        dtype_names=chunks[0].dtype_names,
        tensor_sizes=chunks[0].tensor_sizes,
        device_index=0,
        importer=importer,
    )

    restored = weights[0][1]
    assert restored.shape == torch.Size([])
    assert restored.dtype == dtype
    assert torch.equal(restored.cpu(), original.cpu())


def test_packed_broadcast_consumer_retains_each_slot_until_stream_sync():
    class TrackingStream(_FakeNpuStream):
        def __init__(self):
            self.last_broadcast_ref = None
            self.pending_unpack_ref = None
            self.protected_checks = 0

        def synchronize(self):
            if self.pending_unpack_ref is not None:
                assert self.pending_unpack_ref() is not None
                self.protected_checks += 1
                self.pending_unpack_ref = None

    class LifetimeGroup:
        @staticmethod
        def broadcast(tensor, *, src, stream):
            del src
            stream.last_broadcast_ref = weakref.ref(tensor)

    def tracked_unpack(packed, *args):
        del args
        stream = next(
            stream
            for stream in streams
            if stream.last_broadcast_ref is not None and stream.last_broadcast_ref() is packed
        )
        stream.pending_unpack_ref = weakref.ref(packed)
        return []

    streams = (TrackingStream(), TrackingStream())
    metadata = [(f"w{i}", ([1], torch.float32)) for i in range(3)]
    with (
        patch(f"{_MODULE}._get_streams", return_value=streams),
        patch(f"{_MODULE}.unpack_tensor", new=tracked_unpack),
    ):
        packed_broadcast_consumer(
            iterator=iter(metadata),
            group=LifetimeGroup(),
            src=0,
            post_unpack_func=lambda _: None,
            buffer_size_bytes=1,
            num_buffers=2,
        )

    assert streams[0].protected_checks >= 2
    assert streams[1].protected_checks >= 1


def test_packed_broadcast_consumer_drains_buffer_before_reraising_loader_error():
    class TrackingStream(_FakeNpuStream):
        def __init__(self):
            self.pending_unpack_ref = None
            self.completed_unpack = False

        def synchronize(self):
            if self.pending_unpack_ref is not None:
                assert self.pending_unpack_ref() is not None
                self.completed_unpack = True
                self.pending_unpack_ref = None

    stream = TrackingStream()

    def tracked_unpack(packed, *args):
        del args
        stream.pending_unpack_ref = weakref.ref(packed)
        return []

    with (
        patch(f"{_MODULE}._get_streams", return_value=(stream,)),
        patch(f"{_MODULE}.unpack_tensor", new=tracked_unpack),
        pytest.raises(ValueError, match="loader rejected weights"),
    ):
        packed_broadcast_consumer(
            iterator=iter([("w", ([1], torch.float32))]),
            group=MagicMock(),
            src=0,
            post_unpack_func=MagicMock(side_effect=ValueError("loader rejected weights")),
            buffer_size_bytes=4,
            num_buffers=1,
        )

    assert stream.completed_unpack


def test_drain_preserves_original_error_when_diagnostic_logging_fails():
    class FailingCleanupStream:
        def __init__(self):
            self.calls = 0

        def synchronize(self):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("stream drain failed")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    stream = FailingCleanupStream()
    group = MagicMock()
    group.broadcast.side_effect = ValueError("loader rejected weights")
    packed_tensor_module._UNSYNCHRONIZED_BUFFERS.clear()
    try:
        with (
            patch(f"{_MODULE}._get_streams", return_value=(stream,)),
            patch.object(
                packed_tensor_module.logger,
                "warning",
                side_effect=RuntimeError("logging failed"),
            ),
            pytest.raises(ValueError) as exc_info,
        ):
            packed_broadcast_producer(
                iterator=iter([("w", torch.ones(1))]),
                group=group,
                src=0,
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=4,
                num_buffers=1,
            )

        assert str(exc_info.value) == "loader rejected weights"
        assert len(packed_tensor_module._UNSYNCHRONIZED_BUFFERS) == 1
    finally:
        packed_tensor_module._UNSYNCHRONIZED_BUFFERS.clear()
