# mypy: ignore-errors


from vllm.v1.worker import mamba_utils

from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    # using larger block_size to accelerate copy.
    BLOCK_SIZE = 8192
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
mamba_utils.batch_memcpy = batch_memcpy
