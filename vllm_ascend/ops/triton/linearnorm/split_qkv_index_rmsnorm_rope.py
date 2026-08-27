#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""Fused MiniMax-M3 sparse prepare: split + Gemma RMSNorm + NeoX RoPE.

Concat layout ``[q | k | v | index_q | index_k]``.
With ``attn_out_fp8=True``, main Q/K/V are clamped to +-448 and stored as
e4m3; ``indexer_out_fp8=True`` does the same for indexer Q/K.

Determinism note
----------------
An earlier version gathered cos/sin rows inside the Triton kernel using a
scalar ``get_element`` / ``insert_slice`` loop, i.e. for each token the
position index was read with ``get_element`` and the corresponding
cos/sin row was loaded via indirect addressing (``tl.load(ptr + pos *
stride)``) then written back with ``insert_slice``.

This approach was **non-deterministic** under real inference workloads
(even in eager mode, without torch.compile or graph capture). Root cause:
Ascend NPU hardware scheduling of scalar indirect-address loads is
sensitive to concurrent memory traffic from other operators. When other
ops compete for memory bandwidth / DMA channels, the completion order of
the per-token scalar loads may vary across runs, producing tiny
floating-point differences that propagate and amplify through
autoregressive generation (greedy decoding diverges after enough tokens).

Key evidence from ablation and reproduction:
1. Ablation B (tl.sum RMSNorm + C++ RoPE) = deterministic; Ablation A
   (C++ RMSNorm + Triton RoPE with get_element) = non-deterministic.
2. Standalone single-op test = deterministic (no concurrent traffic).
3. Standalone test with a background memory-heavy stream = reproduces
   non-determinism, and the probability increases with num_tokens (more
   scalar loop iterations).

Fix: cos/sin rows are pre-gathered in Python via a single
``cos_sin_cache[positions]`` (one ``aclnnIndex`` op) before the kernel.
The kernel then reads cos/sin with a **vectorized contiguous offset**
(``ptr + block_offset``), which is immune to hardware scheduling jitter.
The entire gathered tensor is passed directly (no extra Slice ops) and
the kernel selects cos from offset 0 and sin from ``HALF_CACHE``.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from vllm.triton_utils import tl, triton
from triton.language import constexpr
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import (
    extract_slice,
    get_vectorcore_num,
    insert_slice,
)

# A5 / 910_95 / 950 physical UB = 256KB, compiler reserves 8KB.
_A5_UB_RESERVE = 8 * 1024
_UB_KB_A2 = 192
_UB_KB_A5 = 256

# FP8 E4M3 max value for clamping.
_FP8_E4M3_MAX = constexpr(448.0)


@lru_cache(maxsize=1)
def _ub_size_bytes() -> int:
    """Get available UB size in bytes for the current NPU.

    A2 / 910B / 910_93: 192KB.
    A5 / 910_95 / 950: 256KB - 8KB compiler reserve.
    Prefers Triton runtime ``ub_size_in_kbytes`` (same source as compiler).
    """
    kb: int | None = None
    try:
        from triton.backends.ascend.runtime import utils as npu_utils

        kb = int(npu_utils.ub_size_in_kbytes)
    except Exception:
        kb = None
    if kb is None:
        name = ""
        try:
            name = str(torch.npu.get_device_name(0) or "")
        except Exception:
            name = ""
        arch = name.lower()
        is_a5 = any(
            m in arch
            for m in ("910_95", "91095", "ascend950", "950pr", "950dt", "dav-c310")
        )
        kb = _UB_KB_A5 if is_a5 else _UB_KB_A2
    nbytes = kb * 1024
    if kb >= _UB_KB_A5:
        nbytes -= _A5_UB_RESERVE
    return nbytes


def _tokens_per_iter(elem_size: int, elems_per_token: int, *, cap: int = 2) -> int:
    """Estimate token tile size based on 1/4 UB (three loops share UB)."""
    n = int((_ub_size_bytes() // 4) / max(elem_size, 1)) // max(int(elems_per_token), 1)
    return max(1, min(cap, n))


@triton.jit
def split_qkv_index_rmsnorm_rope_kernel(
    input_gm_ptr,  # concat QKV input [q|k|v|index_q|index_k]
    q_gm_ptr,  # main Q output
    k_gm_ptr,  # main K output
    v_gm_ptr,  # main V output (split only, no norm/RoPE)
    index_q_gm_ptr,  # indexer Q output
    index_k_gm_ptr,  # indexer K output (shared single head)
    q_weight_ptr,  # main Q Gemma RMSNorm weight (1+w)
    q_bias_ptr,  # main Q optional bias (unused when BIAS=False)
    k_weight_ptr,  # main K Gemma RMSNorm weight (1+w)
    k_bias_ptr,  # main K optional bias (unused when BIAS=False)
    index_q_weight_ptr,  # indexer Q Gemma RMSNorm weight (1+w)
    index_k_weight_ptr,  # indexer K Gemma RMSNorm weight (1+w)
    cos_gm_ptr,  # pre-gathered cos [batch, max(ATTN_HALF, IDX_HALF)]
    sin_gm_ptr,  # pre-gathered sin [batch, max(ATTN_HALF, IDX_HALF)]
    batch_size,  # number of tokens
    q_hidden_size: tl.constexpr,  # q_size = q_head_num * HEAD_DIM
    kv_hidden_size: tl.constexpr,  # kv_size = kv_head_num * HEAD_DIM
    index_q_size: tl.constexpr,  # index_q_head_num * IDX_HEAD_DIM
    total_hidden_size: tl.constexpr,  # concat last dim total width
    index_offset: tl.constexpr,  # index_q start in concat = q + 2*kv
    index_qk_hidden: tl.constexpr,  # index_q_size + IDX_HEAD_DIM
    eps: tl.constexpr,  # RMSNorm epsilon
    BIAS: tl.constexpr,  # whether to apply bias to main Q/K
    HEAD_DIM: tl.constexpr,  # main head dimension; RoPE views by this
    IDX_HEAD_DIM: tl.constexpr,  # indexer head dimension; RMSNorm reduces by this
    ROPE_DIM: tl.constexpr,  # cache last dim (cos||sin concat length) [unused]
    HALF_CACHE: tl.constexpr,  # ROPE_DIM/2, sin starts here [unused]
    ATTN_HALF: tl.constexpr,  # main partial RoPE half width = attn_rope_dim/2
    IDX_HALF: tl.constexpr,  # indexer partial RoPE half width = idx_rope_dim/2
    num_vectorcore: tl.constexpr,  # number of Vector Cores (grid)
    batch_size_per_iter_per_vec: tl.constexpr,  # main QK loop token tile per iter
    qk_head_nums_per_iter_per_vec: tl.constexpr,  # tile * (q_head+kv_head), for reshape
    q_head_num: tl.constexpr,  # main Q head count
    kv_head_num: tl.constexpr,  # main KV head count
    qk_head_num_sum: tl.constexpr,  # q_head_num + kv_head_num
    v_batch_size_per_iter_per_vec: tl.constexpr,  # V copy loop token tile per iter
    idx_batch_size_per_iter_per_vec: tl.constexpr,  # indexer loop token tile per iter
    idx_qk_heads_per_iter: tl.constexpr,  # tile * index_qk_head_num, for reshape
    index_q_head_num: tl.constexpr,  # indexer Q head count
    index_qk_head_num: tl.constexpr,  # indexer Q head count + 1 (shared index_k)
    ATTN_OUT_FP8: tl.constexpr,  # main Q/K/V clamp -> e4m3
    INDEX_OUT_FP8: tl.constexpr,  # indexer output clamp -> e4m3
):
    row_pid = tl.program_id(0)

    # Load Gemma 1+w (and optional bias) into registers, shared across all loops
    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    index_q_weight_values = tl.load(index_q_weight_ptr + tl.arange(0, IDX_HEAD_DIM)).to(
        tl.float32
    )
    index_k_weight_values = tl.load(index_k_weight_ptr + tl.arange(0, IDX_HEAD_DIM)).to(
        tl.float32
    )
    if BIAS:
        q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
        k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)

    # Partition tokens across Vector Cores; QK / V / indexer each have own tile
    batch_size_per_vec = tl.cdiv(batch_size, num_vectorcore)
    iter_num_per_vec = tl.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)
    v_iter_num_per_vec = tl.cdiv(batch_size_per_vec, v_batch_size_per_iter_per_vec)
    idx_iter_num_per_vec = tl.cdiv(batch_size_per_vec, idx_batch_size_per_iter_per_vec)
    input_batch_offset = row_pid * batch_size_per_vec
    input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)

    # --- Section 1: main QK — load [q|k] -> Gemma RMSNorm -> NeoX RoPE ---
    mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(0, q_hidden_size + kv_hidden_size)
    nmask = nblk_idx < total_hidden_size
    output_q_nblk_idx = tl.arange(0, q_hidden_size)
    output_q_nmask = output_q_nblk_idx < q_hidden_size
    output_kv_nblk_idx = tl.arange(0, kv_hidden_size)
    output_kv_nmask = output_kv_nblk_idx < kv_hidden_size

    for iter in tl.range(iter_num_per_vec):
        pos_offset = iter * batch_size_per_iter_per_vec
        mmask = (mblk_idx + pos_offset) < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        # T * hidden can exceed int32 at 64x16k; use int64 for GM offsets
        row64 = (mblk_idx + pos_offset).to(tl.int64)
        idx = row64[:, None] * total_hidden_size + nblk_idx[None, :]
        values_tmp1 = tl.load(input_gm_ptr + idx, mask=mask).reshape(
            qk_head_nums_per_iter_per_vec, HEAD_DIM
        )

        # Load pre-gathered cos/sin (deterministic, no scalar loop)
        cos_qk_range = tl.arange(0, ATTN_HALF)
        cos = tl.load(cos_gm_ptr + row64[:, None] * ROPE_DIM + cos_qk_range[None, :],
                      mask=mmask[:, None]).to(tl.float32)
        sin = tl.load(sin_gm_ptr + row64[:, None] * ROPE_DIM + HALF_CACHE + cos_qk_range[None, :],
                      mask=mmask[:, None]).to(tl.float32)
        cos = cos.reshape(batch_size_per_iter_per_vec, 1, ATTN_HALF)
        sin = sin.reshape(batch_size_per_iter_per_vec, 1, ATTN_HALF)

        # Gemma RMSNorm: reduce over HEAD_DIM, Q/K heads computed together
        x32 = values_tmp1.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x32 * x32, axis=1) / HEAD_DIM + eps).reshape(
            qk_head_nums_per_iter_per_vec, 1
        )
        normalized_values = (x32 * rstd).reshape(
            batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM
        )

        # Q: multiply by (1+w) -> NeoX RoPE [x1*cos-x2*sin | x2*cos+x1*sin]
        q_heads = extract_slice(
            normalized_values,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )
        if BIAS:
            q_heads = q_heads * q_weight_values + q_bias_values
        else:
            q_heads = q_heads * q_weight_values

        q_x1 = extract_slice(
            q_heads,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        q_x2 = extract_slice(
            q_heads,
            offsets=(0, 0, ATTN_HALF),
            sizes=(batch_size_per_iter_per_vec, q_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        q_heads = insert_slice(
            q_heads,
            q_x1 * cos - q_x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        q_heads = insert_slice(
            q_heads,
            q_x2 * cos + q_x1 * sin,
            offsets=(0, 0, ATTN_HALF),
            sizes=(batch_size_per_iter_per_vec, q_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        # FP8: clamp after RoPE, cast on store
        if ATTN_OUT_FP8:
            q_heads = tl.minimum(tl.maximum(q_heads, -_FP8_E4M3_MAX), _FP8_E4M3_MAX)
        q_output_idx = output_q_nblk_idx[None, :] + row64[:, None] * q_hidden_size
        q_store_mask = (mmask[:, None]) & (output_q_nmask[None, :])
        tl.store(
            q_gm_ptr + q_output_idx,
            q_heads.reshape(batch_size_per_iter_per_vec, q_hidden_size).to(
                q_gm_ptr.dtype.element_ty
            ),
            mask=q_store_mask,
        )

        # K: shares same norm result, sliced from q_head_num onward
        k_heads = extract_slice(
            normalized_values,
            offsets=(0, q_head_num, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )
        if BIAS:
            k_heads = k_heads * k_weight_values + k_bias_values
        else:
            k_heads = k_heads * k_weight_values

        k_x1 = extract_slice(
            k_heads,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        k_x2 = extract_slice(
            k_heads,
            offsets=(0, 0, ATTN_HALF),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        k_heads = insert_slice(
            k_heads,
            k_x1 * cos - k_x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        k_heads = insert_slice(
            k_heads,
            k_x2 * cos + k_x1 * sin,
            offsets=(0, 0, ATTN_HALF),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, ATTN_HALF),
            strides=(1, 1, 1),
        )
        if ATTN_OUT_FP8:
            k_heads = tl.minimum(tl.maximum(k_heads, -_FP8_E4M3_MAX), _FP8_E4M3_MAX)
        kv_output_idx = output_kv_nblk_idx[None, :] + row64[:, None] * kv_hidden_size
        k_store_mask = (mmask[:, None]) & (output_kv_nmask[None, :])
        tl.store(
            k_gm_ptr + kv_output_idx,
            k_heads.reshape(batch_size_per_iter_per_vec, kv_hidden_size).to(
                k_gm_ptr.dtype.element_ty
            ),
            mask=k_store_mask,
        )

    # --- Section 2: V — copy from concat middle, no norm / no RoPE ---
    mblk_idx = tl.arange(0, v_batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = (q_hidden_size + kv_hidden_size) + tl.arange(0, kv_hidden_size)
    nmask = nblk_idx < total_hidden_size
    out_nblk_idx = tl.arange(0, kv_hidden_size)
    out_nmask = out_nblk_idx < kv_hidden_size
    for _ in tl.range(v_iter_num_per_vec):
        mmask = mblk_idx < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        row64 = mblk_idx.to(tl.int64)
        idx = row64[:, None] * total_hidden_size + nblk_idx[None, :]
        values = tl.load(input_gm_ptr + idx, mask=mask)
        # V has no RoPE; still clamp for FP8 before cast on store
        if ATTN_OUT_FP8:
            values = tl.minimum(tl.maximum(values.to(tl.float32), -_FP8_E4M3_MAX), _FP8_E4M3_MAX)
        out_idx = row64[:, None] * kv_hidden_size + out_nblk_idx[None, :]
        out_mask = (mmask[:, None]) & (out_nmask[None, :])
        tl.store(
            v_gm_ptr + out_idx,
            values.to(v_gm_ptr.dtype.element_ty),
            mask=out_mask,
        )
        mblk_idx += v_batch_size_per_iter_per_vec

    # --- Section 3: indexer — concat tail [index_q | index_k], index_k is shared single head ---
    idx_mblk = tl.arange(0, idx_batch_size_per_iter_per_vec) + input_batch_offset
    idx_nblk = index_offset + tl.arange(0, index_qk_hidden)
    idx_nmask = idx_nblk < total_hidden_size
    out_iq_nblk = tl.arange(0, index_q_size)
    out_iq_nmask = out_iq_nblk < index_q_size
    out_ik_nblk = tl.arange(0, IDX_HEAD_DIM)
    out_ik_nmask = out_ik_nblk < IDX_HEAD_DIM

    # indexer: load -> Gemma RMSNorm(IDX_HEAD_DIM) -> NeoX RoPE -> optional FP8 clamp
    for iter in tl.range(idx_iter_num_per_vec):
        pos_offset = iter * idx_batch_size_per_iter_per_vec
        mmask = (idx_mblk + pos_offset) < input_batch_offset_end
        mask = (mmask[:, None]) & (idx_nmask[None, :])
        row64 = (idx_mblk + pos_offset).to(tl.int64)
        idx = row64[:, None] * total_hidden_size + idx_nblk[None, :]
        values_idx = tl.load(input_gm_ptr + idx, mask=mask).reshape(
            idx_qk_heads_per_iter, IDX_HEAD_DIM
        )

        # Load pre-gathered cos/sin for indexer (deterministic)
        cos_idx_range = tl.arange(0, IDX_HALF)
        cos = tl.load(cos_gm_ptr + row64[:, None] * ROPE_DIM + cos_idx_range[None, :],
                      mask=mmask[:, None]).to(tl.float32)
        sin = tl.load(sin_gm_ptr + row64[:, None] * ROPE_DIM + HALF_CACHE + cos_idx_range[None, :],
                      mask=mmask[:, None]).to(tl.float32)
        cos = cos.reshape(idx_batch_size_per_iter_per_vec, 1, IDX_HALF)
        sin = sin.reshape(idx_batch_size_per_iter_per_vec, 1, IDX_HALF)

        # Gemma RMSNorm: index_q multi-head + index_k single head, reduce over IDX_HEAD_DIM
        x32 = values_idx.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x32 * x32, axis=1) / IDX_HEAD_DIM + eps).reshape(
            idx_qk_heads_per_iter, 1
        )
        normalized_idx = (x32 * rstd).reshape(
            idx_batch_size_per_iter_per_vec, index_qk_head_num, IDX_HEAD_DIM
        )

        # index_q: multiply by (1+w) + NeoX RoPE
        iq_heads = extract_slice(
            normalized_idx,
            offsets=(0, 0, 0),
            sizes=(idx_batch_size_per_iter_per_vec, index_q_head_num, IDX_HEAD_DIM),
            strides=(1, 1, 1),
        )
        iq_heads = iq_heads * index_q_weight_values
        iq_x1 = extract_slice(
            iq_heads,
            offsets=(0, 0, 0),
            sizes=(idx_batch_size_per_iter_per_vec, index_q_head_num, IDX_HALF),
            strides=(1, 1, 1),
        )
        iq_x2 = extract_slice(
            iq_heads,
            offsets=(0, 0, IDX_HALF),
            sizes=(idx_batch_size_per_iter_per_vec, index_q_head_num, IDX_HALF),
            strides=(1, 1, 1),
        )
        iq_heads = insert_slice(
            iq_heads,
            iq_x1 * cos - iq_x2 * sin,
            offsets=(0, 0, 0),
            sizes=(idx_batch_size_per_iter_per_vec, index_q_head_num, IDX_HALF),
            strides=(1, 1, 1),
        )
        iq_heads = insert_slice(
            iq_heads,
            iq_x2 * cos + iq_x1 * sin,
            offsets=(0, 0, IDX_HALF),
            sizes=(idx_batch_size_per_iter_per_vec, index_q_head_num, IDX_HALF),
            strides=(1, 1, 1),
        )

        # index_k: shared single head, RoPE uses same cos/sin as index_q
        ik_heads = extract_slice(
            normalized_idx,
            offsets=(0, index_q_head_num, 0),
            sizes=(idx_batch_size_per_iter_per_vec, 1, IDX_HEAD_DIM),
            strides=(1, 1, 1),
        )
        ik_heads = ik_heads * index_k_weight_values
        ik_x1 = extract_slice(
            ik_heads,
            offsets=(0, 0, 0),
            sizes=(idx_batch_size_per_iter_per_vec, 1, IDX_HALF),
            strides=(1, 1, 1),
        )
        ik_x2 = extract_slice(
            ik_heads,
            offsets=(0, 0, IDX_HALF),
            sizes=(idx_batch_size_per_iter_per_vec, 1, IDX_HALF),
            strides=(1, 1, 1),
        )
        ik_heads = insert_slice(
            ik_heads,
            ik_x1 * cos - ik_x2 * sin,
            offsets=(0, 0, 0),
            sizes=(idx_batch_size_per_iter_per_vec, 1, IDX_HALF),
            strides=(1, 1, 1),
        )
        ik_heads = insert_slice(
            ik_heads,
            ik_x2 * cos + ik_x1 * sin,
            offsets=(0, 0, IDX_HALF),
            sizes=(idx_batch_size_per_iter_per_vec, 1, IDX_HALF),
            strides=(1, 1, 1),
        )

        # FP8: clamp after RoPE, cast on store
        if INDEX_OUT_FP8:
            iq_heads = tl.minimum(tl.maximum(iq_heads, -_FP8_E4M3_MAX), _FP8_E4M3_MAX)
            ik_heads = tl.minimum(tl.maximum(ik_heads, -_FP8_E4M3_MAX), _FP8_E4M3_MAX)

        iq_idx = out_iq_nblk[None, :] + row64[:, None] * index_q_size
        ik_idx = out_ik_nblk[None, :] + row64[:, None] * IDX_HEAD_DIM
        iq_mask = (mmask[:, None]) & (out_iq_nmask[None, :])
        ik_mask = (mmask[:, None]) & (out_ik_nmask[None, :])
        tl.store(
            index_q_gm_ptr + iq_idx,
            iq_heads.reshape(idx_batch_size_per_iter_per_vec, index_q_size).to(
                index_q_gm_ptr.dtype.element_ty
            ),
            mask=iq_mask,
        )
        tl.store(
            index_k_gm_ptr + ik_idx,
            ik_heads.reshape(idx_batch_size_per_iter_per_vec, IDX_HEAD_DIM).to(
                index_k_gm_ptr.dtype.element_ty
            ),
            mask=ik_mask,
        )


def split_qkv_index_rmsnorm_rope_impl(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    index_q_size: int,
    head_dim: int,
    idx_head_dim: int,
    eps: float,
    attn_out_fp8: bool = False,
    indexer_out_fp8: bool = False,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused split -> Gemma RMSNorm -> NeoX RoPE (attn + indexer).

    Concat layout ``[q | k | v | index_q | index_k]``.
    With ``attn_out_fp8=True``, main Q/K/V are clamped to +-448 and stored
    as e4m3; ``indexer_out_fp8=True`` does the same for indexer Q/K.
    Cos/sin are pre-gathered in Python for deterministic RoPE.
    """
    input = input.contiguous()
    positions = positions.contiguous()
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    index_q_weight = index_q_weight.contiguous()
    index_k_weight = index_k_weight.contiguous()
    cos_sin_cache = cos_sin_cache.contiguous()

    num_vectorcore = get_vectorcore_num()
    batch_size = input.shape[0]
    cache_dim = int(cos_sin_cache.shape[-1])
    attn_rope_dim = min(cache_dim, int(head_dim))
    idx_rope_dim = min(cache_dim, int(idx_head_dim))

    # ---- Determinism-critical section --------------------------------
    # Pre-gather cos/sin rows by position in Python (one aclnnIndex op).
    #
    # The original implementation gathered cos/sin *inside* the kernel using
    # a scalar loop:
    #
    #   for i in tl.range(batch):
    #       pos = get_element(positions, (i,))
    #       cache_rows = insert_slice(cache_rows,
    #           tl.load(pos * ROPE_DIM + cos_offset[:, None])...)
    #
    # This caused non-deterministic output under real inference workloads
    # because Ascend NPU hardware scheduling of scalar indirect-address
    # loads is unstable when other operators compete for memory bandwidth.
    # The effect is probabilistic: more tokens (more scalar iterations) →
    # higher chance of triggering non-determinism.
    #
    # Fix: gather here in Python via tensor indexing (aclnnIndex, a single
    # deterministic op), then pass the *entire* gathered tensor to the
    # kernel.  The kernel reads cos from offset 0 and sin from HALF_CACHE
    # using a vectorized contiguous load (``ptr + block_offset``), which is
    # immune to hardware scheduling jitter.
    #
    # We pass the full tensor instead of pre-slicing cos/sin separately to
    # avoid two extra Slice + contiguous ops (only 1 aclnnIndex remains).
    # ------------------------------------------------------------------
    cos_sin_gathered = cos_sin_cache[positions].contiguous()  # [batch, cache_dim]
    bias = q_bias is not None
    attn_dtype = torch.float8_e4m3fn if attn_out_fp8 else input.dtype
    index_dtype = torch.float8_e4m3fn if indexer_out_fp8 else input.dtype

    q_out = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=attn_dtype)
    k_out = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=attn_dtype)
    v_out = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=attn_dtype)
    index_q_out = torch.empty(batch_size, index_q_size, device=input.device, dtype=index_dtype)
    index_k_out = torch.empty(batch_size, idx_head_dim, device=input.device, dtype=index_dtype)

    q_head_num = q_hidden_size // head_dim
    kv_head_num = kv_hidden_size // head_dim
    index_q_head_num = index_q_size // idx_head_dim
    index_qk_head_num = index_q_head_num + 1  # +1 = shared index_k
    index_qk_hidden = index_qk_head_num * idx_head_dim
    index_offset = q_hidden_size + 2 * kv_hidden_size
    total_hidden_size = index_offset + index_qk_hidden
    qk_head_num_sum = q_head_num + kv_head_num

    elem = input.element_size()
    qk_factor = 5 * q_hidden_size + 3 * kv_hidden_size + cache_dim * 4 + q_head_num * attn_rope_dim
    idx_factor = (
        5 * index_q_size
        + 3 * idx_head_dim
        + cache_dim * 4
        + index_q_head_num * idx_rope_dim
    )
    batch_tile = _tokens_per_iter(elem, qk_factor)
    idx_batch_tile = _tokens_per_iter(elem, idx_factor)
    v_batch_tile = _tokens_per_iter(elem, kv_hidden_size + 1, cap=4)

    dummy = q_weight
    q_bias = q_bias.contiguous() if q_bias is not None else dummy
    k_bias = k_bias.contiguous() if k_bias is not None else dummy

    grid = (num_vectorcore,)
    split_qkv_index_rmsnorm_rope_kernel[grid](
        input,
        q_out,
        k_out,
        v_out,
        index_q_out,
        index_k_out,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        index_q_weight,
        index_k_weight,
        cos_sin_gathered,
        cos_sin_gathered,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        index_q_size,
        total_hidden_size,
        index_offset,
        index_qk_hidden,
        eps,
        bias,
        head_dim,
        idx_head_dim,
        cache_dim,
        cache_dim // 2,
        attn_rope_dim // 2,
        idx_rope_dim // 2,
        num_vectorcore,
        int(batch_tile),
        int(batch_tile * qk_head_num_sum),
        q_head_num,
        kv_head_num,
        qk_head_num_sum,
        int(v_batch_tile),
        int(idx_batch_tile),
        int(idx_batch_tile * index_qk_head_num),
        index_q_head_num,
        index_qk_head_num,
        attn_out_fp8,
        indexer_out_fp8,
    )
    return q_out, k_out, v_out, index_q_out, index_k_out


def split_qkv_index_rmsnorm_rope_impl_fake(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    index_q_weight: torch.Tensor,
    index_k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    index_q_size: int,
    head_dim: int,
    idx_head_dim: int,
    eps: float,
    attn_out_fp8: bool = False,
    indexer_out_fp8: bool = False,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = input.shape[0]
    attn_dtype = torch.float8_e4m3fn if attn_out_fp8 else input.dtype
    index_dtype = torch.float8_e4m3fn if indexer_out_fp8 else input.dtype
    q_output = torch.empty(batch_size, int(q_hidden_size), device=input.device, dtype=attn_dtype)
    k_output = torch.empty(batch_size, int(kv_hidden_size), device=input.device, dtype=attn_dtype)
    v_output = torch.empty(batch_size, int(kv_hidden_size), device=input.device, dtype=attn_dtype)
    index_q_output = torch.empty(
        batch_size, int(index_q_size), device=input.device, dtype=index_dtype
    )
    index_k_output = torch.empty(
        batch_size, int(idx_head_dim), device=input.device, dtype=index_dtype
    )
    return q_output, k_output, v_output, index_q_output, index_k_output


direct_register_custom_op(
    op_name="qkv_index_rmsnorm_rope",
    op_func=split_qkv_index_rmsnorm_rope_impl,
    fake_impl=split_qkv_index_rmsnorm_rope_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
