#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""NPU-compatible linear attention operators for BailingMoE.

This module provides NPU-compatible replacements for GPU-only Triton kernels
used in BailingMoELinearAttention:
  - ``linear_decode_forward_npu``: replaces ``linear_decode_forward_triton``
  - ``LightningAttentionKernelNPU``: replaces ``MiniMaxText01LinearKernel``
"""

import torch
from einops import rearrange
from vllm.triton_utils import tl, triton


@triton.jit
def _fwd_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
):
    # This kernel computes the diagonal blocks of the attention matrix
    # Each diagonal block represents attention
    # where queries attend to keys in the same block
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK  # batch-head index
    off_block = off % NUM_BLOCK  # block index within the sequence
    off_cblock = tl.program_id(1)  # sub-block index within a block

    off_h = off_bh % h  # head index

    # Calculate base offsets for the current batch and head
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    # Calculate offsets for the current block
    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    # Calculate offsets for the current sub-block
    cblock_offset = off_cblock * CBLOCK
    q_cblock_offset = cblock_offset * d
    o_cblock_offset = cblock_offset * e

    # Calculate pointers to the query, key, value, and output tensors
    Q_block_ptr = (
        Q + qk_offset + qk_block_offset + q_cblock_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    K_block_ptr = K + qk_offset + qk_block_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    V_block_ptr = V + v_offset + v_block_offset + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
    O_block_ptr = (
        Out + o_offset + o_block_offset + o_cblock_offset + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
    )

    # Load the decay rate for the current head
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    i = off_cblock
    q_index = tl.arange(0, CBLOCK) + i * CBLOCK

    # Load query values
    q = tl.load(Q_block_ptr, mask=block_offset + q_index[:, None] < n, other=0.0).to(tl.float32)
    # Initialize output accumulator
    qkv = tl.zeros([CBLOCK, e], dtype=tl.float32)

    # Process all sub-blocks up to and
    # including the current one (causal attention)
    for j in range(i + 1):
        kv_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = q_index[:, None] - kv_index[None, :]
        s_index = s * diff
        # Apply causal mask: only attend to positions before the current one
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        decay = tl.exp(s_index)

        # Load key and value
        k = tl.load(
            K_block_ptr,
            mask=block_offset + kv_index[:, None] < n,
            other=0.0,
        ).to(tl.float32)

        v = tl.load(
            V_block_ptr,
            mask=block_offset + kv_index[:, None] < n,
            other=0.0,
        ).to(tl.float32)

        # Compute attention scores and apply decay
        qk = tl.dot(q, k.trans()) * decay
        # Compute weighted values and accumulate
        qkv += tl.dot(qk, v)

        # Move to the next sub-block
        K_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    tl.store(
        O_block_ptr,
        qkv.to(O_block_ptr.dtype.element_ty),
        mask=block_offset + q_index[:, None] < n,
    )


@triton.jit
def _fwd_kv_parallel(
    K,
    V,
    K_decay,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    # This kernel computes the key-value outer
    # products for each block in parallel
    off_bh = tl.program_id(0)  # batch-head index
    off_block = tl.program_id(1)  # block index within the sequence
    off_e = tl.program_id(2)  # e-dimension tile index for UB overflow prevention

    off_h = off_bh % h  # head index

    block_offset = off_block * BLOCK

    # e-dimension tile offset: each program handles E_FBLOCK columns of e
    e_offset = off_e * E_FBLOCK

    # Calculate offsets for the current block
    k_block_offset = block_offset * d
    v_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    # Calculate base offsets for the current batch and head
    k_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * NUM_BLOCK * d * e

    # Calculate pointers to the key, value, and key-value tensors
    # K does not depend on e_offset (K is [n, d], not [n, e])
    K_block_ptr = K + k_offset + k_block_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, D_FBLOCK)[None, :]
    # V is offset by e_offset to select the current E_FBLOCK columns
    V_block_ptr = (
        V + v_offset + v_block_offset + tl.arange(0, CBLOCK)[:, None] * e + e_offset + tl.arange(0, E_FBLOCK)[None, :]
    )
    # KV is offset by e_offset to write into the correct E_FBLOCK columns
    KV_block_ptr = (
        KV
        + kv_offset
        + kv_block_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + e_offset
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    # Load the decay factors for the current head and block
    k_decay_ptr = K_decay + off_h * BLOCK + tl.arange(0, CBLOCK)

    kv_index = tl.arange(0, CBLOCK)

    # Initialize the key-value outer product accumulator
    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)

    # Handle the last block which might be smaller than BLOCK
    split_n = n - (NUM_BLOCK - 1) * BLOCK if off_block == NUM_BLOCK - 1 else BLOCK
    left_shift = tl.cdiv(split_n, CBLOCK) * CBLOCK - split_n
    num_blocks = min(tl.cdiv(split_n, CBLOCK), NUM_CBLOCK)
    k_decay_ptr += (NUM_CBLOCK - num_blocks) * CBLOCK

    # Process all sub-blocks in the current block
    for j in range(num_blocks):
        left_bound = (1 - j) * left_shift
        # Load key and value, handling boundary conditions
        k_block = tl.load(
            K_block_ptr - left_shift * d,
            mask=(kv_index[:, None] >= left_bound),
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr - left_shift * e,
            mask=(kv_index[:, None] >= left_bound),
            other=0.0,
        ).to(tl.float32)

        # Load decay factor and compute weighted key-value outer product
        k_decay = tl.load(k_decay_ptr)
        k_trans = tl.trans(k_block)
        # NOTE: Need to add the extra dim here due to AMD MLIR lowering error.
        # Please don't move it back until issue is resolved.
        # Issue: https://github.com/ROCm/triton/issues/907
        k_decay = k_decay[None, :]

        kv += tl.dot(k_trans * k_decay, v)

        # Move to the next sub-block
        K_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e
        k_decay_ptr += CBLOCK
    # Store the result
    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


@triton.jit
def _fwd_kv_reduce(
    S,
    KV,
    KV_HISTORY,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
):
    # This kernel reduces the key-value outer products
    # across blocks and updates the KV history
    off_bh = tl.program_id(0)  # batch-head index
    off_e = tl.program_id(1)  # e-dimension tile index for UB overflow prevention
    off_h = off_bh % h  # head index

    # e-dimension tile offset: each program handles E_FBLOCK columns of e
    e_offset = off_e * E_FBLOCK

    kv_offset = off_bh * NUM_BLOCK * d * e

    # Calculate pointer to the key-value tensor, offset by e_offset
    KV_block_ptr = KV + kv_offset + tl.arange(0, D_FBLOCK)[:, None] * e + e_offset + tl.arange(0, E_FBLOCK)[None, :]

    # Load the decay rate for the current head
    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    # Calculate pointer to the key-value history tensor, offset by e_offset
    kv_history_offset = off_bh * d * e
    KV_HISTORY_block_ptr = (
        KV_HISTORY
        + kv_history_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + e_offset
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    # Load the previous key-value history
    kv_pre = tl.load(KV_HISTORY_block_ptr).to(tl.float32)

    # Process all blocks in reverse order to compute the prefix sum
    for i in range(NUM_BLOCK):
        block_size = min(n - i * BLOCK, BLOCK)
        # Compute decay factor for the current block
        block_decay = tl.exp(-s.to(tl.float32) * block_size)

        # Load the current key-value outer product
        kv_cur = tl.load(KV_block_ptr).to(tl.float32)
        # Store the previous key-value history to the current block
        tl.store(KV_block_ptr, kv_pre.to(KV_block_ptr.dtype.element_ty))

        # Update the key-value history with the current block
        kv_pre = block_decay * kv_pre + kv_cur
        KV_block_ptr += d * e
    # Store the updated key-value history
    tl.store(KV_HISTORY_block_ptr, kv_pre)


@triton.jit
def _fwd_none_diag_kernel(
    Q,
    Out,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    E_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    # This kernel computes the non-diagonal blocks of the attention matrix
    # Each non-diagonal block represents attention
    # where queries attend to keys in different blocks
    off_bh = tl.program_id(0)  # batch-head index
    off_h = off_bh % h  # head index

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK  # block index
    off_c = off_nc % NUM_CBLOCK  # sub-block index
    off_e = tl.program_id(2)  # output feature block index

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    e_offset = off_e * E_FBLOCK
    block_offset = n_offset + c_offset

    # Calculate offsets for the current batch, head, and block
    q_offset = off_bh * n * d + (n_offset + c_offset) * d
    o_offset = off_bh * n * e + (n_offset + c_offset) * e + e_offset
    kv_offset = off_bh * NUM_BLOCK * d * e + off_n * d * e + e_offset

    # Calculate pointers to the query, output, and key-value tensors
    Q_block_ptr = Q + q_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    O_block_ptr = Out + o_offset + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    KV_block_ptr = KV + kv_offset + tl.arange(0, d)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]

    # Load the decay rate for the current head
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    c_array = tl.arange(0, CBLOCK)

    # Load the key-value outer product for the current block
    kv = tl.load(KV_block_ptr).to(tl.float32)
    q_index = block_offset + tl.arange(0, CBLOCK)

    # Load query values
    q = tl.load(Q_block_ptr, mask=q_index[:, None] < n, other=0.0).to(tl.float32)

    # Compute decay factors for the current sub-block
    q_decay = tl.exp(-s.to(tl.float32) * (off_c * CBLOCK + c_array[:, None]))

    # Compute non-diagonal attention output
    qkv_none_diag = tl.dot(q, kv) * q_decay

    # Load diagonal attention output (computed by _fwd_diag_kernel)
    qkv_diag = tl.load(O_block_ptr, mask=q_index[:, None] < n, other=0.0).to(tl.float32)

    # Combine diagonal and non-diagonal attention outputs
    qkv = qkv_diag + qkv_none_diag
    # Store the result
    tl.store(O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty), mask=q_index[:, None] < n)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s, kv_history):
        # Forward pass of the lightning attention algorithm
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()

        # Get input dimensions
        b, h, n, d = q.shape
        e = v.shape[-1]

        # Initialize output tensor
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        # ------------------------------------------------------------------ #
        # Tiling parameters (NPU UB-safe, all kernels share the same BLOCK)  #
        #                                                                     #
        # BLOCK      = 256  : sequence tile size, unified across all kernels  #
        #                     to keep diag / non-diag semantics consistent.   #
        # CBLOCK_D   = 32   : sub-tile for _fwd_diag_kernel                  #
        #                     UB ≈ 72 KB  (< 192 KB limit)                   #
        # CBLOCK_KV  = 64   : sub-tile for _fwd_kv_parallel /                #
        #                     _fwd_none_diag_kernel                           #
        #                     UB ≈ 112 KB (< 192 KB limit)                   #
        # E_FBLOCK   = e//2 : split the e-dimension into two tiles so that   #
        #                     _fwd_kv_parallel UB stays within limits.        #
        #                     The full e range is covered via grid dim-2      #
        #                     (NUM_EFBLOCK tiles), NOT by truncation.         #
        # ------------------------------------------------------------------ #
        BLOCK = 256
        NUM_BLOCK = triton.cdiv(n, BLOCK)

        # Step 1: Compute diagonal blocks of attention
        # Each program handles CBLOCK_D rows of Q within one BLOCK-sized tile.
        # UB breakdown (fp32): q[32,d] + k[32,d] + v[32,e] +
        #                      qk[32,32] + decay[32,32] + qkv[32,e]
        #                    = 4*(2*32*128 + 2*32*128 + 2*32*32) ≈ 72 KB
        CBLOCK_D = 32
        NUM_CBLOCK_D = BLOCK // CBLOCK_D
        assert BLOCK % CBLOCK_D == 0, "BLOCK must be a multiple of CBLOCK_D"

        grid_diag = (b * h * NUM_BLOCK, NUM_CBLOCK_D)
        _fwd_diag_kernel[grid_diag](
            q,
            k,
            v,
            o,
            s,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            CBLOCK=CBLOCK_D,
            NUM_BLOCK=NUM_BLOCK,
            multibuffer=True,
            limit_auto_multi_buffer_only_for_local_buffer=False,
            set_workspace_multibuffer=4,
            tile_mix_vector_loop=2,
            tile_mix_cube_loop=2,
        )

        # Compute decay factors for keys (shape: [h, BLOCK])
        array = torch.arange(0, BLOCK, device=q.device) + 1
        k_decay = torch.exp(-s * (BLOCK - array.reshape(1, -1)))

        # Feature-dimension tiling:
        # D_FBLOCK covers the full d dimension (no split needed for d).
        # E_FBLOCK splits e into NUM_EFBLOCK tiles; each tile is processed
        # by a separate program (grid dim-2) so the full e is always covered.
        D_FBLOCK = d  # process all d columns in one shot
        E_FBLOCK = e // 2  # half of e per program instance
        NUM_EFBLOCK = e // E_FBLOCK  # = 2 tiles to cover full e
        assert e % E_FBLOCK == 0, "e must be divisible by E_FBLOCK"

        CBLOCK_KV = 64
        NUM_CBLOCK_KV = BLOCK // CBLOCK_KV
        assert BLOCK % CBLOCK_KV == 0, "BLOCK must be a multiple of CBLOCK_KV"

        # Step 2: Compute key-value outer products for each block in parallel.
        # Grid dim-2 (NUM_EFBLOCK) ensures the full e dimension is covered
        # without UB overflow.
        # UB breakdown (fp32): kv[d,E_FBLOCK] + k[CBLOCK_KV,d] +
        #                      v[CBLOCK_KV,E_FBLOCK] + k_trans*decay[d,CBLOCK_KV]
        #                    = 4*(128*64 + 64*128 + 64*64 + 128*64) ≈ 112 KB
        kv = torch.empty((b, h, NUM_BLOCK, d, e), dtype=torch.float32, device=q.device)
        grid_kv = (b * h, NUM_BLOCK, NUM_EFBLOCK)
        _fwd_kv_parallel[grid_kv](
            k,
            v,
            k_decay,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
            NUM_FBLOCK=NUM_EFBLOCK,
            CBLOCK=CBLOCK_KV,
            NUM_CBLOCK=NUM_CBLOCK_KV,
        )

        # Step 3: Reduce key-value outer products across blocks and update
        # KV history. Grid dim-1 (NUM_EFBLOCK) covers the full e dimension.
        # UB breakdown (fp32): kv_pre[d,E_FBLOCK] + kv_cur[d,E_FBLOCK]
        #                    = 2*4*128*64 = 64 KB
        grid_reduce = (b * h, NUM_EFBLOCK)
        _fwd_kv_reduce[grid_reduce](
            s,
            kv,
            kv_history,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK,
            E_FBLOCK=E_FBLOCK,
        )

        # Step 4: Compute non-diagonal blocks of attention.
        # Grid dim-2 (NUM_EFBLOCK) covers the full e dimension.
        # UB breakdown (fp32): kv[d,E_FBLOCK] + q[CBLOCK_KV,d] +
        #                      qkv_none[CBLOCK_KV,E_FBLOCK] +
        #                      qkv_diag[CBLOCK_KV,E_FBLOCK] + q_decay[CBLOCK_KV,1]
        #                    = 4*(128*64 + 64*128 + 64*64 + 64*64 + 64) ≈ 96 KB
        grid_none_diag = (b * h, NUM_BLOCK * NUM_CBLOCK_KV, NUM_EFBLOCK)
        _fwd_none_diag_kernel[grid_none_diag](
            q,
            o,
            s,
            kv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            E_FBLOCK=E_FBLOCK,
            CBLOCK=CBLOCK_KV,
            NUM_CBLOCK=NUM_CBLOCK_KV,
        )

        # Save tensors for backward pass
        ctx.save_for_backward(q, k, v, s, kv)
        ctx.BLOCK = BLOCK

        return o, torch.cat([kv, kv_history.unsqueeze(2)], dim=2)


lightning_attention_npu_ = _attention.apply


def lightning_attention_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ed: torch.Tensor,
    block_size: int,
    kv_history: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """lightning attention forward pass (NPU-friendly)."""
    d = q.shape[-1]
    e = v.shape[-1]

    if ed.dim() == 1:
        ed = ed.view(1, -1, 1, 1)

    # Split the computation into chunks for better parallelism
    m = 128 if d >= 128 else 64
    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)
    n = len(arr)
    output = 0

    # Initialize or clone key-value history
    if kv_history is None:
        kv_history = torch.zeros((q.shape[0], q.shape[1], d, e), dtype=torch.float32, device=q.device)
    else:
        kv_history = kv_history.clone().contiguous()

    # Process each chunk and accumulate results
    for i in range(n - 1):
        s = arr[i]
        e = arr[i + 1]
        q1 = q[..., s:e]
        k1 = k[..., s:e]
        o, kv = lightning_attention_npu_(q1, k1, v, ed, kv_history)
        output = output + o
    return output, kv


class AscendLightningAttentionKernel:
    """NPU-friendly lightning attention kernel for BailingMoE prefill.

    Replaces ``MiniMaxText01LinearKernel`` by providing an NPU-friendly
    implementation of the prefill forward pass
    """

    @staticmethod
    def jit_linear_forward_prefix(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_caches: torch.Tensor,
        slope_rate: torch.Tensor,
        block_size: int,
        layer_idx: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        slope_rate = slope_rate.to(torch.float32)
        should_squeeze = q.dim() == 3
        if should_squeeze:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        b, h, n, d = q.shape
        e = v.shape[-1]
        kv_history = kv_caches.reshape(1, h, d, e).contiguous()
        output, kv_history = lightning_attention_npu(
            q,
            k,
            v,
            slope_rate,
            block_size=block_size,
            kv_history=kv_history,
        )
        kv_caches.copy_(kv_history[:, :, -1, :, :].reshape(h, d, e))
        assert output.shape[0] == 1, "batch size must be 1"
        return rearrange(output.squeeze(0), "h n d -> n (h d)")
