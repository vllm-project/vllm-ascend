# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Use PTA native stream/event bindings only inside common HiSparse modules.

PTA 2.10's generic event wait does not flush capture propagation before CANN
workspace allocation; the native event binding does. Keep common event points
and algorithms intact, using the platform bindings through a module-local
facade. Remove this adapter when PTA's generic bindings have equivalent graph
capture behavior. The actual torch module and model-runner timing are untouched.
"""

from __future__ import annotations

import importlib
import threading

import torch as _torch
from vllm.platforms import current_platform

_COMMON_MODULES = ("vllm.v1.hisparse.runtime", "vllm.v1.attention.backends.mla.index_group")
_facade = None
_original_group_current_stream = None


def _is_npu(device=None) -> bool:
    if device is None or isinstance(device, int):
        return current_platform.device_type == "npu"
    return _torch.device(device).type == "npu"


def _native_stream(stream):
    if stream is None or isinstance(stream, _torch.npu.Stream):
        return stream
    if not isinstance(stream, _torch.Stream) or stream.device.type != "npu":
        raise TypeError("A HiSparse native NPU event requires an NPU stream")
    # Reconstruct the same pooled c10 stream, never an ExternalStream.
    return _torch.npu.Stream(
        stream_id=stream.stream_id, device_index=stream.device_index, device_type=stream.device_type
    )


class HiSparseStream(_torch.npu.Stream):
    """Native stream with the context-manager interface used by common code."""

    def __new__(cls, device=None, priority=0, **kwargs):
        if not _is_npu(device):
            return _torch.Stream(device=device, priority=priority, **kwargs)
        stream = super().__new__(cls, device=device, priority=priority, **kwargs)
        stream._hisparse_contexts = threading.local()
        return stream

    def __enter__(self):
        context = _torch.npu.stream(self)
        context.__enter__()
        contexts = getattr(self._hisparse_contexts, "stack", None)
        if contexts is None:
            contexts = self._hisparse_contexts.stack = []
        contexts.append(context)
        return self

    def __exit__(self, *exc):
        return self._hisparse_contexts.stack.pop().__exit__(*exc)

    def wait_stream(self, stream):
        return super().wait_stream(_native_stream(stream))


class HiSparseEvent(_torch.npu.Event):
    """Native event; never reinterpret a generic Stream as a native object."""

    def __new__(cls, *args, **kwargs):
        if not _is_npu():
            return _torch.Event(*args, **kwargs)
        return super().__new__(cls, *args, **kwargs)

    def record(self, stream=None):
        return super().record(_native_stream(stream))

    def wait(self, stream=None):
        return super().wait(_native_stream(stream))


def native_current_stream(device=None):
    if _is_npu(device):
        return _torch.npu.current_stream(device)
    return _torch.accelerator.current_stream(device)


def create_event(*args, **kwargs):
    return HiSparseEvent(*args, **kwargs)


def _group_current_stream():
    if _is_npu():
        return native_current_stream()
    return _original_group_current_stream()


class _AcceleratorFacade:
    current_stream = staticmethod(native_current_stream)

    def __getattr__(self, name):
        return getattr(_torch.accelerator, name)


class _TorchFacade:
    Stream = HiSparseStream
    Event = HiSparseEvent
    accelerator = _AcceleratorFacade()

    def __getattr__(self, name):
        return getattr(_torch, name)


def install() -> bool:
    """Cold activation before the first index-group layer is registered."""
    global _facade, _original_group_current_stream
    if not _is_npu():
        return False
    modules = [importlib.import_module(name) for name in _COMMON_MODULES]
    index_module = modules[1]
    if _facade is not None:
        if (
            not all(module.torch is _facade for module in modules)
            or index_module.current_stream is not _group_current_stream
        ):
            raise RuntimeError("HiSparse stream facade changed after installation")
        return True
    # Install only before cache/graph construction. Leave the actual Torch
    # module and all algorithm/event points intact; do not unwrap other owners.
    if not callable(getattr(index_module, "current_stream", None)):
        raise RuntimeError("HiSparse requires Core's current-stream interface")
    if any(module.torch is not _torch for module in modules):
        raise RuntimeError("HiSparse common torch binding was already replaced")
    facade = _TorchFacade()
    _original_group_current_stream = index_module.current_stream
    for module in modules:
        module.torch = facade
    index_module.current_stream = _group_current_stream
    _facade = facade
    return True
