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
        if not (np.issubdtype(rows.dtype, np.integer) or rows.dtype == np.bool_):
            # NumPy accepted a key we cannot read as rows, for example an
            # object that only implements ``__index__``. The write already
            # happened, so publish everything rather than drop it.
            self._mark_whole()
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

    ``__setitem__`` is the only write the tracker can observe, so it is the
    only write the mirror accepts. Every other access goes through a read-only
    alias of the same memory: a write through a view, ``np.asarray``,
    ``np.put``, ``flat``/``data`` or an in-place method raises instead of
    reaching the mirror with no callback and leaving the device copy stale.

    What stays outside the tracker cannot be closed here: the raw pointer
    (``.ctypes``) and a tensor built with ``torch.from_numpy`` over any of these
    arrays, because torch ignores NumPy's writeable flag. The proxy is also not
    an ``ndarray``, so arithmetic, comparison and the ``np.*`` helpers are not
    forwarded; read through ``np.asarray(buf.np)`` when a real array is needed.
    """

    def __init__(self, array: np.ndarray, callback: Callable):
        self._array = array
        # ``np.frombuffer`` over a read-only memoryview, not ``array.view()``:
        # a plain read-only view can have its writeable flag turned back on
        # whenever the base stays writeable, which it has to here.
        self._readonly = np.frombuffer(memoryview(array).toreadonly(), dtype=array.dtype).reshape(array.shape)
        self._callback = callback

    _OWN_FIELDS = ("_array", "_readonly", "_callback")

    def __setattr__(self, name, value):
        if name not in self._OWN_FIELDS:
            # Without this, ``buf.np.shape = ...`` would quietly land in the
            # instance dictionary and shadow the forwarding instead of failing.
            raise AttributeError(f"{type(self).__name__}.{name} cannot be assigned; write through buf.np[...] = value")
        object.__setattr__(self, name, value)

    def __setitem__(self, key, value):
        self._array[key] = value
        self._callback(key)

    def __len__(self) -> int:
        return len(self._array)

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self._readonly, dtype=dtype)
        if copy:
            array = array.copy()
        return array

    def __getitem__(self, key):
        return self._readonly[key]

    def __getattr__(self, name):
        return getattr(self._readonly, name)


class MonitoredTorchTensor:
    """A torch mirror that reports the rows a write touches.

    ``__setitem__`` and the in-place methods (torch's trailing-underscore
    convention) are the writes the tracker can observe. In-place methods mark
    the whole buffer, which is slower but never stale; delegated calls and
    attributes that hand back a tensor over the mirror's storage (``to`` with
    the same dtype, ``detach``, ``view``, ``T``, ``split``, ...) mark it too.
    ``.numpy()`` returns a read-only array and ``.data`` is refused.

    torch has no read-only views, so two writes stay outside the tracker: one
    through a view returned by ``__getitem__`` (``t.cpu[0:2].fill_(9)``), and one
    through a tensor that ``torch.from_numpy`` built over ``.numpy()`` (torch
    only warns about the writeable flag). Wrapping ``__getitem__`` would make
    every read pay for a proxy, and the in-repo callers only read; write through
    ``.np`` or ``__setitem__`` instead.
    """

    def __init__(self, tensor: torch.Tensor, callback: Callable):
        self._tensor = tensor
        self._callback = callback
        self._storage_ptr = tensor.untyped_storage().data_ptr()

    _OWN_FIELDS = ("_tensor", "_callback", "_storage_ptr")

    def __setattr__(self, name, value):
        if name not in self._OWN_FIELDS:
            # Same shadowing risk as the NumPy proxy: ``buf.cpu.shape = ...``
            # would otherwise be accepted and change nothing.
            raise AttributeError(f"{type(self).__name__}.{name} cannot be assigned; write through buf.cpu[...] = value")
        object.__setattr__(self, name, value)

    def __setitem__(self, key, value):
        self._tensor[key] = value
        self._callback(key)

    def __len__(self) -> int:
        return len(self._tensor)

    def __getitem__(self, key):
        # See the class docstring: a view write is the one remaining way to
        # reach the mirror unnoticed, and readers need a real tensor here.
        return self._tensor[key]

    def _is_mirror_alias(self, value) -> bool:
        """Whether ``value`` is a tensor, or a tuple of tensors, over our memory."""
        if isinstance(value, torch.Tensor):
            return value.untyped_storage().data_ptr() == self._storage_ptr
        if isinstance(value, tuple) and value:
            return self._is_mirror_alias(value[0])
        return False

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
        if isinstance(attr, torch.Tensor) and self._is_mirror_alias(attr):
            # ``.T``/``.mT``/``.H``/``.real`` hand out a writable view of the
            # mirror, so a write through one has to be published too.
            self._callback(slice(None))
        if not callable(attr):
            return attr

        def _call(*args, **kwargs):
            inplace = name.endswith("_") and not name.startswith("__")
            if inplace:
                # torch's in-place convention: the receiver always changes, so
                # the whole buffer is dirty whichever rows it touches.
                self._callback(slice(None))
            result = attr(*args, **kwargs)
            if not inplace and self._is_mirror_alias(result):
                # ``to``/``detach``/``view``/``split``... handed out an alias
                # of the mirror; a write through it has to be published.
                self._callback(slice(None))
            return result

        return _call


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
