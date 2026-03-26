# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/mamba_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.triton_utils import tl, triton


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    # We need to mv pointer_type cast outside the loop.
    # Otherwise it causes potential bugs.
    src_ptr = src_ptr.to(tl.pointer_type(tl.uint8))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.uint8))

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        curr_src_ptr = src_ptr + i + offsets
        curr_dst_ptr = dst_ptr + i + offsets

        # cache_modifier=".cg" bypasses L1 cache for streaming data.
        data = tl.load(curr_src_ptr, mask=mask, cache_modifier=".cg")
        tl.store(curr_dst_ptr, data, mask=mask, cache_modifier=".cg")
