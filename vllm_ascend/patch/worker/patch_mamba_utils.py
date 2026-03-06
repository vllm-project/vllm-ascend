# mypy: ignore-errors

import torch

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
    BLOCK_SIZE = 1
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


vllm.v1.worker.mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
vllm.v1.worker.mamba_utils.batch_memcpy = batch_memcpy