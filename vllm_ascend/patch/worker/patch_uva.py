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

    These buffers are sized like the flattened block table, so ``dim_size`` can
    reach millions of rows. Recording the dirty rows as
    ``set(range(dim_size))`` (what an arbitrary key used to do) and calling a
    Python callback once per row made a single ``np[:n] = x`` cost hundreds of
    milliseconds on the host.

    The common write pattern is a contiguous prefix (``buf.np[:n] = x``), which
    is tracked in O(1). Anything scattered falls back to a vectorised boolean
    mask, so the per-write cost stays in NumPy instead of Python.
    """

    __slots__ = ("_size", "_prefix", "_count", "_mask")

    def __init__(self, size: int) -> None:
        self._size = size
        # Rows in [0, _prefix) are known to be dirty. _count may over-count
        # (a row can be marked twice), but it never under-counts, which is all
        # the "prefix is fully covered" proof needs.
        self._prefix = 0
        self._count = 0
        self._mask: np.ndarray | None = None

    def mark(self, key) -> None:
        """Record the rows touched by an ``arr[key] = value`` write."""
        if isinstance(key, tuple):
            if not key:
                # ``arr[()] = value`` addresses the whole buffer in NumPy.
                self._mark_whole()
                return
            rows = key[0]
        else:
            rows = key
        if rows is None or rows is Ellipsis:
            self._mark_whole()
            return
        if isinstance(rows, slice):
            start, stop, step = rows.indices(self._size)
            if step < 0:
                # A negative step can never extend the prefix, and an empty
                # one must not be confused with a reversed range.
                if stop >= start:
                    return
                self._mark_rows(np.arange(start, stop, step))
                return
            if stop <= start:
                return
            if step == 1 and start <= self._prefix:
                # Extends (or stays inside) the dirty prefix: O(1).
                self._count += stop - start
                if stop > self._prefix:
                    self._prefix = stop
                return
            self._mark_rows(np.arange(start, stop, step))
            return
        if isinstance(rows, (int, np.integer)) and not isinstance(rows, bool):
            index = int(rows)
            if index < 0:
                index += self._size
            if index < 0 or index >= self._size:
                raise IndexError(f"row index {rows} out of [0, {self._size})")
            if index == self._prefix:
                self._prefix += 1
                self._count += 1
            elif index < self._prefix:
                self._count += 1
            else:
                self._mark_rows(np.array([index], dtype=np.intp))
            return
        self._mark_rows(rows if isinstance(rows, np.ndarray) else np.asarray(rows))

    def _mark_whole(self) -> None:
        self._count += self._size
        self._prefix = self._size

    def _mark_rows(self, rows: np.ndarray) -> None:
        if self._mask is None:
            self._mask = np.zeros(self._size, dtype=bool)
        self._mask[rows] = True

    def take(self) -> list[tuple[int, int]]:
        """Return the contiguous ``[start, stop)`` runs to copy, then reset."""
        runs: list[tuple[int, int]] = []
        covered = 0
        if self._prefix:
            # _count >= _prefix, so every row below _prefix was marked.
            runs.append((0, self._prefix))
            covered = self._prefix
        if self._mask is not None:
            index = np.flatnonzero(self._mask[covered:])
            self._mask[:] = False
            if index.size:
                index += covered
                breaks = np.flatnonzero(np.diff(index) != 1) + 1
                for run in np.split(index, breaks):
                    runs.append((int(run[0]), int(run[-1]) + 1))
        self._prefix = 0
        self._count = 0
        return runs


class MonitoredNumPyArray:
    """A wrapper around a NumPy array that monitors modifications."""

    def __init__(self, array: np.ndarray, callback: Callable):
        self._array = array
        self._callback = callback

    def __setitem__(self, key, value):
        self._array[key] = value
        self._callback(key)

    def __getitem__(self, key):
        return self._array[key]

    def __getattr__(self, name):
        return getattr(self._array, name)


class MonitoredTorchTensor:
    """A wrapper around a torch tensor that monitors modifications."""

    def __init__(self, tensor: torch.Tensor, callback: Callable):
        self._tensor = tensor
        self._callback = callback

    def __setitem__(self, key, value):
        self._tensor[key] = value
        self._callback(key)

    def __getitem__(self, key):
        return self._tensor[key]

    def __getattr__(self, name):
        return getattr(self._tensor, name)


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

    def _mark_cpu_modified(self, key) -> None:
        self._dirty.mark(key)

    @property
    def cpu(self):
        return self._cpu_view

    @property
    def np(self):
        return self._np_view

    def _sync_dirty_rows(self) -> None:
        if self._use_real_uva:
            return
        for start, stop in self._dirty.take():
            src = self._cpu[start:stop]
            dst = self._uva[start:stop]
            # A slice of the pinned host buffer is still pinned, so the copy
            # stays asynchronous without an intermediate staged tensor.
            if src.dtype == torch.uint64:
                dst.view(torch.int64).copy_(src.view(torch.int64), non_blocking=True)
            else:
                dst.copy_(src, non_blocking=True)

    def uva(self, n: int | None = None) -> torch.Tensor:
        """Get the device data of the buffer.

        vLLM main (#56908) turned UvaBuffer.uva into a method taking an
        optional row count; the pool and StagedWriteTensor call it.
        """
        self._sync_dirty_rows()
        return self._uva if n is None else self._uva[:n]


vllm.v1.worker.gpu.buffer_utils.UvaBuffer = UvaBufferWrapper
