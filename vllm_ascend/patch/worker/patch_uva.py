# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/block_table.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import os
from collections.abc import Callable, Sequence
from importlib.metadata import version

import numpy as np
import torch
import vllm.v1.worker.gpu.buffer_utils
from vllm.logger import logger


def check_triton_ascend_version_valid() -> bool:
    """
    Check triton-ascend version and warn about UVA feature disablement.
    If the installed version isn't affected by the UVA issue, return True.
    """
    # Triton Ascend versions affected by the UVA pointer validation issue.
    UVA_INCOMPATIBLE_VERSIONS = ("3.2.1", "3.2.2")
    installed_version = version("triton-ascend")
    if installed_version in UVA_INCOMPATIBLE_VERSIONS:
        logger.warning(
            "triton-ascend %s disables the UVA feature.\n"
            "Related bug issue: https://github.com/triton-lang/triton-ascend/issues/783",
            installed_version,
        )
        return False
    return True


def is_uva_available() -> bool:
    """check if uva feature is supported in this environment"""
    # FIXME(chenboxun): Some triton-ascend versions reject pinned CPU tensors.
    # Thus UVA is disabled for affected versions.
    # (Related bug issue link: https://github.com/triton-lang/triton-ascend/issues/783)
    return (
        "pinned_mem_register:True" in os.environ.get("PYTORCH_NPU_ALLOC_CONF", {})
        and check_triton_ascend_version_valid()
    )


class _DirtyRows:
    """Track which rows of the host shadow need a device refresh.

    A buffer has one row per request slot, so ``size`` follows ``max_num_reqs``
    (``max_num_reqs * num_dims`` for the M-RoPE position table) and the mask
    stays in the kilobyte range. Recording the dirty rows as ``set(range(n))``
    and calling a Python callback once per row made a single ``np[:n] = x``
    cost hundreds of milliseconds on the host, and that prefix write runs on
    every step.

    The common write pattern is a contiguous prefix (``buf.np[:n] = x``), which
    is tracked in O(1). Anything scattered falls back to a vectorised boolean
    mask, so the per-write cost stays in NumPy instead of Python.
    """

    __slots__ = ("_size", "_prefix", "_mask", "_mask_pending")

    def __init__(self, size: int) -> None:
        self._size = size
        # Rows in [0, _prefix) are known to be dirty. The prefix only grows
        # through writes that cover it from row 0, so no per-row bookkeeping
        # is needed to prove the coverage.
        self._prefix = 0
        # The mask is allocated on the first scattered write and then kept, so
        # that one scattered write does not make every later sync re-zero the
        # row buffer. This flag is what keeps the scan off the hot path: it is
        # the only reason take() has to look at the mask at all.
        self._mask: np.ndarray | None = None
        self._mask_pending = False

    def mark(self, key) -> None:
        """Record the rows touched by an ``arr[key] = value`` write.

        NumPy has already performed the write by the time this runs, so the
        only job here is to classify the key: whole buffer, a slice, a single
        row, or an explicit row list/mask.
        """
        if isinstance(key, tuple):
            if not key:
                # ``arr[()] = value`` addresses the whole buffer in NumPy.
                self._mark_whole()
                return
            # The buffers are two-dimensional and writes are row-wise, so only
            # the first axis selects which rows change.
            key = key[0]
        if key is None or key is Ellipsis:
            # ``arr[None] = v`` broadcasts over the whole buffer, and so does
            # ``arr[...] = v``.
            self._mark_whole()
        elif isinstance(key, slice):
            self._mark_slice(key)
        elif isinstance(key, (bool, np.bool_)):
            # NumPy reads a scalar boolean as a mask over the whole array.
            self._mark_whole()
        elif isinstance(key, (int, np.integer)):
            self._mark_index(int(key))
        else:
            self._mark_rows(key if isinstance(key, np.ndarray) else np.asarray(key))

    def _mark_slice(self, key: slice) -> None:
        """Rows selected by a slice, in both directions and when empty."""
        start, stop, step = key.indices(self._size)
        if step < 0:
            # A reversed slice can never extend the prefix, and its empty form
            # is ``stop >= start`` - the opposite of the forward case.
            if stop < start:
                self._mark_rows(np.arange(start, stop, step))
            return
        if stop <= start:
            return
        if step == 1 and start <= self._prefix:
            # Joining [0, _prefix) with [start, stop) is still a prefix, so the
            # whole run is covered without touching the mask.
            if stop > self._prefix:
                self._prefix = stop
            return
        self._mark_rows(np.arange(start, stop, step))

    def _mark_index(self, index: int) -> None:
        """A single row, resolved against negative indices like NumPy does."""
        row = index + self._size if index < 0 else index
        if row < 0 or row >= self._size:
            raise IndexError(f"row index {index} out of [0, {self._size})")
        if row == self._prefix:
            # Appending to the prefix keeps it contiguous.
            self._prefix += 1
        elif row > self._prefix:
            # A row below the prefix is already covered by it. Set the bit
            # directly: the 310P staged-write path lands here once per request
            # and tensor, so a single-row write must not allocate an index
            # array of its own.
            if self._mask is None:
                self._mask = np.zeros(self._size, dtype=bool)
            self._mask[row] = True
            self._mask_pending = True

    def _mark_whole(self) -> None:
        self._prefix = self._size

    def _mark_rows(self, rows: np.ndarray) -> None:
        if rows.size == 0:
            # ``arr[[]] = v`` writes nothing. NumPy accepts the empty list (it
            # infers an integer index), but ``np.asarray([])`` here is float64
            # and would raise on the mask; an empty multi-dimensional boolean
            # mask selects no elements either, so both are no-ops.
            return
        if rows.dtype == np.bool_ and rows.ndim > 1:
            # ``arr[arr > 0] = v`` selects elements, not rows, and NumPy has
            # already performed that write by the time we get here. Refresh
            # the whole buffer instead of raising IndexError on the 1-D mask.
            self._mark_whole()
            return
        if self._mask is None:
            self._mask = np.zeros(self._size, dtype=bool)
        self._mask[rows] = True
        self._mask_pending = True

    def take(self) -> list[tuple[int, int]]:
        """Return the coalesced ``[start, stop)`` runs to copy, then reset."""
        runs: list[tuple[int, int]] = []
        covered = 0
        if self._prefix:
            runs.append((0, self._prefix))
            covered = self._prefix
        if self._mask_pending:
            mask = self._mask
            assert mask is not None  # _mark_rows allocates before the flag
            index = np.flatnonzero(mask[covered:])
            mask[:] = False
            self._mask_pending = False
            if index.size:
                index += covered
                breaks = np.flatnonzero(np.diff(index) != 1) + 1
                for run in np.split(index, breaks):
                    start, stop = int(run[0]), int(run[-1]) + 1
                    if runs and runs[-1][1] == start:
                        # The first scattered run can touch the prefix; one
                        # H2D copy covers both.
                        runs[-1] = (runs[-1][0], stop)
                    else:
                        runs.append((start, stop))
        self._prefix = 0
        return runs


class MonitoredNumPyArray:
    """A NumPy mirror that reports the rows a write touches.

    ``__setitem__`` is the only write the dirty tracker can observe, so every
    other way of writing is either refused or treated as a whole-buffer write.
    Handing out a writable view would reach the mirror without a callback and
    leave the device copy silently stale, and the tracker has no way to notice
    afterwards.
    """

    def __init__(self, array: np.ndarray, callback: Callable):
        self._array = array
        self._callback = callback

    def _seal(self, array: np.ndarray) -> np.ndarray:
        """Make a view read-only so a write cannot bypass ``__setitem__``."""
        if not np.may_share_memory(array, self._array):
            # An independent copy: writing to it cannot reach the mirror.
            return array
        if array is self._array:
            # Never seal the mirror itself; it has to stay writable here.
            array = array.view()
        array.flags.writeable = False
        return array

    def __setitem__(self, key, value):
        self._array[key] = value
        self._callback(key)

    def __array__(self, dtype=None, copy=None):
        # NumPy reaches a mirror through this path (``np.asarray``,
        # ``np.copyto``, ``np.put``, ``out=``). Seal it like any other view;
        # the protocol attributes are blocked in ``__getattr__`` so NumPy
        # cannot take a writable shortcut around it.
        array = np.asarray(self._array, dtype=dtype)
        if copy:
            array = array.copy()
        return self._seal(array)

    def __getitem__(self, key):
        result = self._array[key]
        return self._seal(result) if isinstance(result, np.ndarray) else result

    def __getattr__(self, name):
        if name in ("__array_struct__", "__array_interface__"):
            # Either of these would hand NumPy a writable view of the mirror.
            raise AttributeError(name)
        attr = getattr(self._array, name)
        if name in ("flat", "data", "ctypes", "base"):
            # These expose the memory without a NumPy view we can seal, so the
            # access itself has to publish the whole buffer.
            self._callback(slice(None))
        if isinstance(attr, np.ndarray):
            return self._seal(attr)
        if not callable(attr):
            return attr

        def _call(*args, **kwargs):
            result = attr(*args, **kwargs)
            if result is None:
                # ndarray's in-place methods (fill, put, sort, partition, ...)
                # mutate the receiver and return None.
                self._callback(slice(None))
            elif isinstance(result, np.ndarray):
                return self._seal(result)
            return result

        return _call


class MonitoredTorchTensor:
    """A torch mirror that reports the rows a write touches.

    ``__setitem__`` and the in-place methods (torch's trailing-underscore
    convention) are the writes the tracker can observe. In-place methods mark
    the whole buffer, which is slower but never stale; ``.numpy()`` hands out a
    read-only array and ``.data`` is refused so neither can write behind the
    tracker's back.

    torch has no read-only views, so a write through a view (``t.cpu[0:2]``,
    ``t.cpu.view(-1)``) still cannot be observed. Use ``.np`` or a
    ``__setitem__`` on the mirror for writes; the in-repo readers only read.
    """

    def __init__(self, tensor: torch.Tensor, callback: Callable):
        self._tensor = tensor
        self._callback = callback

    def __setitem__(self, key, value):
        self._tensor[key] = value
        self._callback(key)

    def __getitem__(self, key):
        # See the class docstring: a view write is the one remaining way to
        # reach the mirror unnoticed, and readers need a real tensor here.
        return self._tensor[key]

    def __getattr__(self, name):
        if name == "data":
            raise AttributeError("'.data' bypasses the UVA dirty tracking; write through '.np' or the tensor itself")
        attr = getattr(self._tensor, name)
        if name == "numpy":

            def _numpy(*args, **kwargs):
                array = attr(*args, **kwargs)
                array.flags.writeable = False
                return array

            return _numpy
        if name in ("storage", "untyped_storage"):
            # Raw storage access can write any element; publish everything.
            self._callback(slice(None))
            return attr
        if callable(attr) and name.endswith("_") and not name.startswith("__"):

            def _inplace(*args, **kwargs):
                # torch's in-place methods always mutate the receiver, so the
                # whole buffer is dirty whichever rows they touch.
                self._callback(slice(None))
                return attr(*args, **kwargs)

            return _inplace
        return attr


def _zeros_on_npu(cpu: torch.Tensor) -> torch.Tensor:
    """Zero an NPU buffer matching ``cpu``.

    ``aclnnInplaceZero`` rejects ``DT_UINT64``. Prompt-embeds pointer tables
    use that dtype, so zero an int64 allocation and view it back. The zero
    bit pattern is identical, and callers still see ``uint64``.
    """
    if cpu.dtype == torch.uint64:
        return torch.zeros(cpu.shape, dtype=torch.int64, device="npu").view(torch.uint64)
    return torch.zeros_like(cpu, device="npu")


def _int64_storage(tensor: torch.Tensor) -> torch.Tensor:
    """View a uint64 tensor as int64 so NPU copy kernels accept it."""
    if tensor.dtype == torch.uint64:
        return tensor.view(torch.int64)
    return tensor


class UvaBufferWrapper:
    """
    Ascend NPU doesn't support UVA tensors directly.
    This is a wrapper class that provides CPU and NPU views of a UVA tensor.
    However if users add environment parameter below, UVA feature is Supported.
    os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'pinned_mem_register:True'
    """

    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        self._cpu: torch.Tensor = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self._np: np.ndarray = self._cpu.numpy()
        self._dirty = _DirtyRows(self._cpu.size(0))
        requested_real_uva = is_uva_available()
        if requested_real_uva:
            import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401

        view = torch.ops._C_ascend.get_npu_view_from_cpu_tensor(self._cpu) if requested_real_uva else None
        self._use_real_uva = view is not None
        if requested_real_uva and not self._use_real_uva:
            logger.warning_once("Pinned CPU memory is not mapped for NPU UVA; using the async H2D fallback")
        self._uva: torch.Tensor = view if view is not None else _zeros_on_npu(self._cpu)
        # Views are either raw (real UVA) or monitored wrappers (emulated path),
        # so both attributes are explicitly typed as a union.
        self._cpu_view: torch.Tensor | MonitoredTorchTensor
        self._np_view: np.ndarray | MonitoredNumPyArray
        if self._use_real_uva:
            self._cpu_view = self._cpu
            self._np_view = self._np
        else:
            self._cpu_view = MonitoredTorchTensor(self._cpu, self._dirty.mark)
            self._np_view = MonitoredNumPyArray(self._np, self._dirty.mark)

    @property
    def cpu(self):
        return self._cpu_view

    @property
    def np(self):
        return self._np_view

    def _sync_dirty_rows(self) -> None:
        if self._use_real_uva:
            return
        runs = self._dirty.take()
        for index, (start, stop) in enumerate(runs):
            # A slice of the pinned host buffer is still pinned, so the copy
            # stays asynchronous without an intermediate staged tensor.
            try:
                _int64_storage(self._uva[start:stop]).copy_(_int64_storage(self._cpu[start:stop]), non_blocking=True)
            except Exception:
                # take() has already reset the tracker, so put the ranges that
                # were not refreshed back to keep a failed sync retryable
                # instead of leaving the device copy silently stale.
                for pending_start, pending_stop in runs[index:]:
                    self._dirty.mark(slice(pending_start, pending_stop))
                raise

    def uva(self, n: int | None = None) -> torch.Tensor:
        """Get the device data of the buffer.

        vLLM main (#56908) turned UvaBuffer.uva into a method taking an
        optional row count; the pool and StagedWriteTensor call it.
        """
        self._sync_dirty_rows()
        return self._uva if n is None else self._uva[:n]


vllm.v1.worker.gpu.buffer_utils.UvaBuffer = UvaBufferWrapper
