# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU boundary adapters around the installed vLLM packed loops.

Only the packed module's torch reference and two helper hooks are patched;
the real torch.cuda namespace and producer/consumer functions stay unchanged.
This is an NPU-process adaptation, not support for mixing CUDA and NPU weight
transfer engines in one process. Remove the module proxy when vLLM exposes
device-neutral stream operations and an owning-unpack extension point.
"""

from contextvars import ContextVar
from functools import cache, wraps
from threading import Lock
from typing import Any

import torch
from vllm.distributed.weight_transfer import packed_tensor as upstream

from vllm_ascend.distributed.weight_transfer.packed_tensor import _drain_packed_streams

# Per-context state prevents callbacks in other threads from sharing a transfer.
# vLLM caches streams per device; a per-device lock prevents simultaneous use.
_ACTIVE: ContextVar[Any] = ContextVar("ascend_packed_transfer", default=None)
_INSTALL_LOCK = Lock()


class _NpuTorch:
    """Module-local device routing, without mutating the torch module."""

    @property
    def cuda(self):
        return torch.npu

    @property
    def Event(self):
        return torch.npu.Event

    def __getattr__(self, name):
        return getattr(torch, name)


@cache
def _device_lock(device: int) -> Any:
    return Lock()


def _install() -> None:
    with _INSTALL_LOCK:
        if isinstance(upstream.torch, _NpuTorch):
            return
        get_streams = upstream._get_streams
        unpack = upstream.unpack_tensor

        @wraps(get_streams)
        def streams_for_transfer(device_idx, num_buffers):
            streams = get_streams(device_idx, num_buffers)
            state = _ACTIVE.get()
            if state is not None:
                state.streams = list(streams)
                state.buffers = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in streams]
                state.readers = (*streams, state.caller)
                state.events = [torch.npu.Event() for _ in streams] if state.receiving else []
                for stream in streams:
                    stream.wait_stream(state.caller)
            return streams

        def owning_unpack(packed, names, shapes, dtypes, sizes):
            state = _ACTIVE.get()
            if state is None:
                return unpack(packed, names, shapes, dtypes, sizes)
            stream = torch.npu.current_stream()
            if state.previous_ready is not None:
                stream.wait_event(state.previous_ready)
            weights = [
                (name, raw.clone().view(dtype).reshape(shape))
                for name, shape, dtype, raw in zip(names, shapes, dtypes, packed.split(sizes))
            ]
            for _, tensor in weights:
                for reader in state.readers:
                    tensor.record_stream(reader)
            return weights

        upstream.torch = _NpuTorch()
        upstream._get_streams = streams_for_transfer
        upstream.unpack_tensor = owning_unpack


class _Transfer:
    def __init__(self, group, callback, receiving):
        self.group = group
        self.callback = callback
        self.receiving = receiving
        self.caller = torch.npu.current_stream()
        self.streams = []
        self.buffers = []
        self.readers = ()
        self.events = []
        self.previous_ready = None
        self.previous_broadcast = None

    def slot(self, stream):
        return self.streams.index(stream)

    def broadcast(self, tensor, src, **kwargs):
        stream = torch.npu.current_stream()
        slot = self.slot(stream)
        self.buffers[slot] = tensor
        if not self.receiving and self.previous_broadcast is not None and self.previous_broadcast != stream:
            self.previous_broadcast.synchronize()
        # torch_npu packing/allocation must reach the native stream before HCCL.
        stream.synchronize()
        self.group.broadcast(tensor, src=src, stream=stream)
        self.previous_broadcast = stream
        if self.receiving:
            stream.synchronize()
            # Protect raw storage on both main and release; clones are tracked
            # separately by owning_unpack, before invoking the model callback.
            for reader in self.readers:
                tensor.record_stream(reader)

    def load(self, weights):
        self.callback(weights)
        stream = torch.npu.current_stream()
        self.previous_ready = self.events[self.slot(stream)]
        self.previous_ready.record(stream)


def broadcast(iterator, group, src, callback, buffer_size_bytes, num_buffers, *, receiving):
    """Delegate the actual loop to the installed (main or release) vLLM."""
    if buffer_size_bytes <= 0 or num_buffers <= 0:
        raise ValueError("buffer_size_bytes and num_buffers must be positive")
    if _ACTIVE.get() is not None:
        raise RuntimeError("nested packed transfers cannot share vLLM's cached streams")
    _install()
    device = torch.npu.current_device()
    # functools.cache can evaluate a missing key twice on concurrent first use.
    # Serialize lookup too, so both transfers obtain the same device lock.
    with _INSTALL_LOCK:
        lock = _device_lock(device)
    with lock:
        state = _Transfer(group, callback, receiving)
        token = _ACTIVE.set(state)
        error = None
        try:
            if receiving:

                def checked_metadata():
                    for name, (shape, dtype) in iterator:
                        if any(dimension < 0 for dimension in shape):
                            raise ValueError(f"tensor {name!r} has a negative shape: {shape}")
                        yield name, (shape, dtype)

                upstream.packed_nccl_broadcast_consumer(
                    checked_metadata(),
                    state,
                    src,
                    state.load,
                    buffer_size_bytes,
                    num_buffers,
                    device=f"npu:{device}",
                )
            else:

                def checked(item):
                    tensor = callback(item)
                    original = item[1]
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError(f"post_iter_func for {item[0]!r} must return a torch.Tensor")
                    if tensor.dtype != original.dtype or tensor.numel() != original.numel():
                        raise ValueError(
                            f"post_iter_func returned tensor {item[0]!r} with dtype={tensor.dtype}, "
                            f"numel={tensor.numel()}; expected dtype={original.dtype}, numel={original.numel()}"
                        )
                    return tensor

                upstream.packed_nccl_broadcast_producer(
                    iterator,
                    state,
                    src,
                    checked,
                    buffer_size_bytes,
                    num_buffers,
                )
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                _drain_packed_streams(state.streams, state.buffers, error)
            finally:
                _ACTIVE.reset(token)
