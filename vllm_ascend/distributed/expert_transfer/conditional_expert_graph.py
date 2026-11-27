# SPDX-License-Identifier: Apache-2.0
"""CANN conditional subgraphs for expert-offloaded decode graphs."""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch_npu

from .expert_weight_transfer import ACL_MEMCPY_HOST_TO_DEVICE

if TYPE_CHECKING:
    from .expert_offload_manager import LayerOffloadBinding

_CAPTURE_THREAD_LOCAL = 1
_COND_RESET_EACH_REPLAY = 1
_COND_IF = 0
_HANDLE = ctypes.c_void_p
_HANDLE_OUT = ctypes.POINTER(_HANDLE)


class _ConditionalTask(ctypes.Structure):
    _fields_ = [
        ("handle", _HANDLE),
        ("type", ctypes.c_int),
        ("size", ctypes.c_uint32),
        ("models", _HANDLE_OUT),
    ]


def _check_acl(name: str, status: int) -> None:
    if status:
        raise RuntimeError(f"{name} failed: ACL error {status}")


def _bind_acl(library, name, *argument_types):
    function = getattr(library, name)
    function.argtypes = argument_types
    function.restype = ctypes.c_int

    def call(*args):
        _check_acl(name, function(*args))

    return call


class ConditionalExpertGraph:
    """Own one serial capture stream and the child workspaces for a runner's graphs."""

    @classmethod
    def create_if_supported(cls, device: torch.device) -> ConditionalExpertGraph | None:
        try:
            library = ctypes.CDLL("libascendcl.so")
            return cls(device, library)
        except (OSError, AttributeError, ImportError):
            # Fixed-group capture is available without the conditional graph runtime.
            return None

    def __init__(self, device: torch.device, library: ctypes.CDLL):
        import acl  # type: ignore[import-not-found]

        self.acl = acl
        self._capture_info = _bind_acl(
            library, "aclmdlRICaptureGetInfo", _HANDLE, ctypes.POINTER(ctypes.c_int), _HANDLE_OUT
        )
        self._create_condition = _bind_acl(
            library, "aclmdlRICondHandleCreate", _HANDLE, ctypes.c_uint32, ctypes.c_int, _HANDLE_OUT
        )
        self._condition_pointer = _bind_acl(library, "aclmdlRICondHandleGetCondPtr", _HANDLE, _HANDLE_OUT)
        self._add_task = _bind_acl(library, "aclmdlRIAddCondTask", _ConditionalTask, _HANDLE, ctypes.c_uint32)
        self._begin_capture = _bind_acl(library, "aclmdlRICaptureToModelRIBegin", _HANDLE, _HANDLE, ctypes.c_int)
        self._end_capture = _bind_acl(library, "aclmdlRICaptureEnd", _HANDLE, _HANDLE_OUT)
        # Finish host-side operator submission before ending native capture.
        self.capture_stream = torch_npu.npu.SyncLaunchStream(device=device)
        self.child_pool = torch_npu.npu.memory.MemPool()
        self.overflow_flag = torch.ones(1, dtype=torch.int64, device="cpu", pin_memory=True)

    def register(self, binding: LayerOffloadBinding, num_topk_ids: int, cut_ids: int) -> tuple[int, int]:
        stream = torch.npu.current_stream(binding.device)
        parent, status = _HANDLE(), ctypes.c_int()
        self._capture_info(stream.npu_stream, ctypes.byref(status), ctypes.byref(parent))

        condition, address = _HANDLE(), _HANDLE()
        self._create_condition(parent, 0, _COND_RESET_EACH_REPLAY, ctypes.byref(condition))
        self._condition_pointer(condition, ctypes.byref(address))
        torch.npu._launch_host_func(stream, self._prepare_and_select, (binding, num_topk_ids, cut_ids, address.value))

        # CANN returns the true and false child model handles in that order.
        branches = (_HANDLE * 2)()
        self._add_task(_ConditionalTask(condition, _COND_IF, len(branches), branches), stream.npu_stream, 0)
        overflow, fit = branches
        return fit, overflow

    @torch.inference_mode()
    def _prepare_and_select(self, args: tuple[LayerOffloadBinding, int, int, int]) -> None:
        binding, num_topk_ids, cut_ids, address = args
        try:
            if binding.load_first_graph_group(num_topk_ids, cut_ids):
                # CANN resets the flag to zero at replay. Only overflow needs a write.
                stream = binding.manager.graph_load_stream
                size = self.overflow_flag.element_size()
                _check_acl(
                    "Expert graph condition H2D",
                    self.acl.rt.memcpy_async(
                        address, size, self.overflow_flag.data_ptr(), size, ACL_MEMCPY_HOST_TO_DEVICE, stream.npu_stream
                    ),
                )
                stream.synchronize()
        except Exception:
            binding.fail_graph_callback()

    @contextmanager
    def capture_branch(self, branch: int):
        # Child temporaries stay alive in a dedicated pool. The child branches
        # run serially and write into outputs owned by the parent graph.
        with torch.npu.stream(self.capture_stream), torch_npu.npu.memory.use_mem_pool(self.child_pool):
            self._begin_capture(self.capture_stream.npu_stream, branch, _CAPTURE_THREAD_LOCAL)
            try:
                yield
            finally:
                self._end_capture(self.capture_stream.npu_stream, ctypes.byref(_HANDLE()))
