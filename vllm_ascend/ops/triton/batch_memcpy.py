# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/mamba_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.triton_utils import tl, triton


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_addr = tl.load(src_ptrs + pid)
    dst_addr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    # FIX (EZ9999 "MTE accesses an invalid GM address"): restore upstream's
    # same-buffer overlap guard, which was dropped when this kernel was ported.
    # mamba_cache_mode="align" keeps every mamba state inside ONE pool, so a
    # copy's source and destination ranges can overlap. Without the barrier, a
    # lower-address lane's store destroys source bytes that another lane still
    # has to read -- a program-order race whose outcome depends on device
    # scheduling, which is why the corruption surfaces later, in an unrelated
    # kernel, on many cores at once, and why any host sync hides it.
    # Upstream: vllm/v1/worker/mamba_utils.py batch_memcpy_kernel.
    is_left_overlap = (dst_addr < src_addr) & (dst_addr + size > src_addr)

    # We need to mv pointer_type cast outside the loop.
    # Otherwise it causes potential bugs.
    src_ptr = src_addr.to(tl.pointer_type(tl.uint8))
    dst_ptr = dst_addr.to(tl.pointer_type(tl.uint8))

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        curr_src_ptr = src_ptr + i + offsets
        curr_dst_ptr = dst_ptr + i + offsets

        # cache_modifier=".cg" bypasses L1 cache for streaming data.
        data = tl.load(curr_src_ptr, mask=mask, cache_modifier=".cg")
        if is_left_overlap:
            # Preserve each lane's source before a lower-address lane stores
            # over it. The condition is uniform within the program, so for the
            # (overwhelmingly common) non-overlapping pairs this is a no-op.
            tl.debug_barrier()
        tl.store(curr_dst_ptr, data, mask=mask, cache_modifier=".cg")
