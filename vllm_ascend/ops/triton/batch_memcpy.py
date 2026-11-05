# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/mamba_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from dataclasses import dataclass

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import LaunchSpec, TritonWarmupTensor
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel


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


class BatchMemcpyKernel(AscendTritonJitKernel["BatchMemcpyKernel.CompileKey"]):
    """Compile-only warmup for KDA's aligned INT64 byte-pointer vectors."""

    kernel = batch_memcpy_kernel

    @dataclass(frozen=True)
    class CompileKey:
        block_size: int

    def dispatch(self, *, block_size: int) -> CompileKey:
        return self.CompileKey(block_size=block_size)

    def get_warmup_keys(self, *, block_size: int) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(block_size=block_size)

    def warmup_inputs(self, compile_key: CompileKey) -> dict:
        return dict(
            src_ptrs=TritonWarmupTensor(torch.int64),
            dst_ptrs=TritonWarmupTensor(torch.int64),
            sizes=TritonWarmupTensor(torch.int64),
            block_size=compile_key.block_size,
            selected=1,
        )

    @kernel_launcher
    def __call__(self, src_ptrs, dst_ptrs, sizes, *, block_size: int, selected: int) -> LaunchSpec:
        return (selected,), dict(BLOCK_SIZE=block_size)


_BATCH_MEMCPY_KERNEL = BatchMemcpyKernel()
