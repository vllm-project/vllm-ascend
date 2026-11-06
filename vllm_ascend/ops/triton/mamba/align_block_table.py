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
for triton-ascend plus a CPU fallback, and mirrors upstream's semantics
(num_reqs derived from the block table, asserted equal to the seq_lens rows).
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

    One program per request. This replaces a five-statement PyTorch expression
    that expanded to seven kernels over a ``(num_reqs,)`` tensor -- one element
    in a single-request decode -- so almost all of its cost was launch
    overhead rather than work.

    Both block-table strides are passed explicitly. The expression this
    replaces was ``torch.gather(block_table, 1, ...)``, which honours any
    layout, and this is a shared utility -- ``gdn_attn``, ``mamba_attn``,
    ``short_conv_attn`` and ``linear_attn`` all call it -- so assuming a
    column stride of 1 would silently read the wrong slots for a caller
    holding a non-contiguous view.

    Two annotations above are load-bearing and easy to drop by accident:

    ``block_table_stride_0: tl.int64``
        Without it ``req_idx * block_table_stride_0`` is computed in i32 --
        ``program_id`` is i32 and Triton specializes a stride below 2^31 to
        i32 -- so the product wraps once a row offset reaches 2^31 elements.
        ``torch.gather``, which this replaces, indexes in 64-bit and has no
        such limit, so narrowing it would be a regression.

    ``do_not_specialize_on_alignment`` on the three pointers
        Triton puts pointer alignment in the specialization key, and callers
        can hand this kernel row-sliced views (``block_table[:num_reqs]``,
        ``seq_lens[:num_reqs]``). When a slice starts at a request index that
        is not a multiple of 4 the view is differently aligned and a fresh
        variant JIT-compiles mid-step. The torch expression this replaced
        never compiled anything, so that recompile would be a regression
        introduced by this change.
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
    upstream kernel is CUDA-gated) and every other device keeps the plain
    torch expression, which the kernel must match bit-for-bit.
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
    if block_table.device.type != "npu":
        # Triton runs on the accelerator only, and this helper IS called with
        # CPU tensors: the GDN and linear-attention metadata builders are
        # unit-tested with DEVICE = torch.device("cpu"). The expression the
        # kernel replaces was plain torch and handled that, so keep it rather
        # than narrowing the contract.
        start_indices = (seq_lens - 1) // kv_cache_spec.block_size
        start_indices = start_indices.clamp(min=0)
        offsets = torch.arange(n_out, device=block_table.device, dtype=torch.int32)
        return torch.gather(block_table, 1, (start_indices.unsqueeze(1) + offsets).to(torch.int64))
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
