#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
# mypy: ignore-errors
# Adapted from vllm-project/vllm/vllm/v1/worker/mamba_utils.py
# Replace the CUDA batch_memcpy kernel with a Triton kernel for Ascend NPU.

import vllm
from vllm.triton_utils import tl, triton


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        curr_src_ptr = (src_ptr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst_ptr = (dst_ptr + i + offsets).to(tl.pointer_type(tl.uint8))

        data = tl.load(curr_src_ptr, mask=mask)
        tl.store(curr_dst_ptr, data, mask=mask)


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    # NOTE: BLOCK_SIZE must be a power-of-2 constexpr for Triton.
    # 128 bytes per tile provides a reasonable balance between parallelism
    # and register pressure on Ascend NPU; adjust if profiling suggests
    # a different value.
    BLOCK_SIZE = 128
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


vllm.v1.worker.mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
vllm.v1.worker.mamba_utils.batch_memcpy = batch_memcpy
