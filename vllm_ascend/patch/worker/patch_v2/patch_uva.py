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
    Check triton-ascend version and warn about UVA feature disablement in 3.2.1.
    If version isn't 3.2.1, return True.
    """
    # Target version that disables UVA feature
    DISABLE_UVA_VERSION = "3.2.1"
    installed_version = version("triton-ascend")
    # Check if current version is the one with UVA disabled
    if installed_version == DISABLE_UVA_VERSION:
        logger.warning(
            "triton-ascend %s disables the UVA feature.\n"
            "Related bug issue: https://github.com/triton-lang/triton-ascend/issues/783",
            DISABLE_UVA_VERSION,
        )
        return False
    return True


def is_uva_available() -> bool:
    """check if uva feature is supported in this environment"""
    # FIXME(chenboxun): There is an issue with using the UVA feature alongside triton-ascend3.2.1.
    # Thus UVA feature is disabled when using version 3.2.1
    # (Related bug issue link: https://github.com/triton-lang/triton-ascend/issues/783)
    return (
        "pinned_mem_register:True" in os.environ.get("PYTORCH_NPU_ALLOC_CONF", {})
        and check_triton_ascend_version_valid()
    )


def get_row_indices_from_key(key: int | slice | tuple, dim_size: int) -> set[int]:
    """get the set of row indices involved in the given key."""
    if isinstance(key, int):
        # parse index such as np[1]
        key = key if key >= 0 else dim_size + key
        # handle negative index
        if key < 0 or key >= dim_size:
            raise IndexError(f"row index {key} out of [0, {dim_size})")
        return {key}
    elif isinstance(key, slice):
        # parse slice such as np[1:3]
        start, stop, step = key.indices(dim_size)
        return set(range(start, stop, step))
    elif isinstance(key, tuple):
        # parse row slice such as np[1,:100]
        if len(key) == 0:
            return set(range(dim_size))
        return get_row_indices_from_key(key[0], dim_size)
    else:
        # for other types such as list/ndarray, we return all rows.
        return set(range(dim_size))


class MonitoredNumPyArray:
    """A wrapper around a NumPy array that monitors modifications."""

    def __init__(self, array: np.ndarray, callback: Callable):
        self._array = array
        self._callback = callback

    def __setitem__(self, key, value):
        self._array[key] = value
        dim_size = self._array.shape[0]
        row_indices = get_row_indices_from_key(key, dim_size)
        for row in row_indices:
            self._callback(row)

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
        dim_size = self._tensor.size(0)
        row_indices = get_row_indices_from_key(key, dim_size)
        for row in row_indices:
            self._callback(row)

    def __getitem__(self, key):
        return self._tensor[key]

    def __getattr__(self, name):
        return getattr(self._tensor, name)


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
        self._modified_indices: set[int] = set()
        self._uva: torch.Tensor = self._cpu if is_uva_available() else torch.zeros_like(self._cpu, device="npu")

    def _mark_cpu_modified(self, key: int):
        self._modified_indices.add(key)

    @property
    def cpu(self):
        return self._cpu if is_uva_available() else MonitoredTorchTensor(self._cpu, self._mark_cpu_modified)

    @property
    def np(self):
        return self._np if is_uva_available() else MonitoredNumPyArray(self._np, self._mark_cpu_modified)

    @property
    def uva(self):
        """Get the device data of the buffer."""
        if not is_uva_available() and self._modified_indices:
            # Sort for better memory access locality
            dirty_rows = sorted(self._modified_indices)
            # can't use copy_ method, because copy_ for index tensor
            #  will malloc new memory.
            self._uva[dirty_rows] = self._cpu[dirty_rows].to(device="npu", non_blocking=True)
            self._modified_indices.clear()
        return self._uva


vllm.v1.worker.gpu.buffer_utils.UvaBuffer = UvaBufferWrapper
