# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Fused mamba "align" block-table gather on Ascend (port of vLLM #58737).

``mamba_get_block_table_tensor``'s "align" branch takes the last
``1 + num_speculative_blocks`` block-table entries of each request. As PyTorch
ops this is seven kernel launches over a ``(num_reqs,)`` tensor -- a single
element in a one-request decode -- so almost all of its cost is launch
overhead rather than work. On 910B the seven eager ops measure ~370 us device
/ ~247 us host per call while one fused Triton kernel takes ~215 us / ~121 us
(-42% device, -51% host), flat across batch sizes.

The fused kernel cannot be inherited from upstream: vLLM gates it with
``if not block_table.is_cuda``, so NPU always falls back to the seven-op torch
expression regardless of the vLLM version. This module ships the same kernel
for triton-ascend and mirrors upstream's semantics (num_reqs derived from
the block table, asserted equal to the seq_lens rows).
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


@triton.jit(do_not_specialize_on_alignment=["out_ptr", "block_table_ptr", "seq_lens_ptr"])
def _mamba_align_block_table_kernel(
    out_ptr,
    block_table_ptr,
    block_table_stride_0: tl.int64,
    block_table_stride_1,
    seq_lens_ptr,
    seq_lens_stride,
    block_size,
    n_out,
    BLOCK: tl.constexpr,
):
    """Gather the last ``n_out`` block-table entries of each request.

    One program per request; both strides are passed explicitly because
    callers may hold non-contiguous views of the block table. The two
    annotations above are load-bearing: ``block_table_stride_0: tl.int64``
    keeps the row-offset product from wrapping past 2**31 elements, and the
    pointer alignment de-specialization keeps row-sliced views from
    recompiling mid-serve.
    """
    req_idx = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + req_idx * seq_lens_stride)
    # A zero-length request in a graph has an invalid block table; clamp
    # to 0 so it reads a valid row, exactly as the previous clamp_ did.
    start = tl.maximum((seq_len - 1) // block_size, 0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_out
    vals = tl.load(
        block_table_ptr + req_idx * block_table_stride_0 + (start + offs) * block_table_stride_1,
        mask=mask,
        other=0,
    )
    # `out` is freshly allocated by the caller and therefore contiguous, so its
    # row stride is exactly n_out.
    tl.store(out_ptr + req_idx * n_out + offs, vals, mask=mask)


def mamba_get_block_table_tensor(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_spec: KVCacheSpec,
    mamba_cache_mode: str,
) -> torch.Tensor:
    """Ascend counterpart of vLLM's ``mamba_get_block_table_tensor``.

    Identical contract, except the fused Triton kernel is taken on NPU (the
    upstream kernel is CUDA-gated).
    """
    if mamba_cache_mode == "none":
        return block_table
    assert isinstance(kv_cache_spec, MambaSpec)
    num_reqs = block_table.shape[0]
    n_out = 1 + kv_cache_spec.num_speculative_blocks
    assert seq_lens.shape[0] == num_reqs, f"seq_lens has {seq_lens.shape[0]} rows, block_table has {num_reqs}"
    assert seq_lens.device == block_table.device, (
        f"seq_lens is on {seq_lens.device}, block_table on {block_table.device}"
    )
    out = block_table.new_empty((num_reqs, n_out))
    if num_reqs == 0:
        return out
    _mamba_align_block_table_kernel[(num_reqs,)](
        out,
        block_table,
        block_table.stride(0),
        block_table.stride(1),
        seq_lens,
        seq_lens.stride(0),
        kv_cache_spec.block_size,
        n_out,
        BLOCK=triton.next_power_of_2(n_out),
    )
    return out
