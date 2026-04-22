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
from collections.abc import Callable, Sequence

import numpy as np
import torch
import vllm.v1.worker.gpu.buffer_utils


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
    """Ascend NPU doesn't support UVA tensors directly. This is a wrapper class
    that provides CPU and NPU views of a UVA tensor."""

    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        self._cpu: torch.Tensor = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self._np = self._cpu.numpy()
        self._uva: torch.Tensor = torch.zeros_like(self._cpu, device="npu")
        self._modified_indices: set[int] = set()

    def _mark_cpu_modified(self, key: int):
        self._modified_indices.add(key)

    @property
    def cpu(self):
        return MonitoredTorchTensor(self._cpu, self._mark_cpu_modified)

    @property
    def np(self):
        return MonitoredNumPyArray(self._np, self._mark_cpu_modified)

    @property
    def uva(self):
        """Get the device data of the buffer."""
        if self._modified_indices:
            # Sort for better memory access locality
            dirty_rows = sorted(self._modified_indices)
            # can't use copy_ method, because copy_ for index tensor
            #  will malloc new memory.
            self._uva[dirty_rows] = self._cpu[dirty_rows].to(device="npu", non_blocking=True)
            self._modified_indices.clear()
        return self._uva


vllm.v1.worker.gpu.buffer_utils.UvaBuffer = UvaBufferWrapper
