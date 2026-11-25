# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import prod

import torch


def scratch_allocator(device, storage=None):
    offset = 0

    def allocate(shape, dtype=torch.bfloat16):
        nonlocal offset
        if storage is None:
            return torch.empty(shape, dtype=dtype, device=device)
        start = (offset + 511) // 512 * 512
        offset = start + prod(shape) * dtype.itemsize
        if offset > storage.numel():
            raise ValueError("Insufficient SFA decode workspace")
        return storage[start:offset].view(dtype).view(shape)

    return allocate
