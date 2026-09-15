#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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


import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import extract_slice, get_vectorcore_num, insert_slice

# Per-head-group tile sizing: target elements per (head_group, head_size) tile.
_HEAD_TILE_ELEMS = 4096
# Unified-buffer budget (bytes) for the v passthrough block size.
_UB_BUDGET_BYTES = 87040
_MAX_V_BLOCK_T = 32
# Token block size for the position loads and inline cos/sin tiles.
_COS_BLOCK = 8


def _pow2_floor(value: int, lo: int, hi: int) -> int:
    block = lo
    while block * 2 <= value and block * 2 <= hi:
        block *= 2
    return block


@triton.jit(
    do_not_specialize=["num_tokens", "positions_stride_0", "positions_stride_1", "tokens_per_core"]
)
def split_qkv_rmsnorm_mrope_kernel(
    in_qkv_ptr: torch.Tensor,
    q_weight_ptr: torch.Tensor,
    q_bias_ptr: torch.Tensor,
    k_weight_ptr: torch.Tensor,
    k_bias_ptr: torch.Tensor,
    cos_sin_ptr: torch.Tensor,
    positions_ptr: torch.Tensor,
    inv_freq_ptr: torch.Tensor,
    out_q_ptr: torch.Tensor,
    out_k_ptr: torch.Tensor,
    out_v_ptr: torch.Tensor,
    out_gate_ptr: torch.Tensor,
    num_tokens,
    positions_stride_0,
    positions_stride_1,
    tokens_per_core,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    eps: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    has_bias: tl.constexpr,
    is_interleaved: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    INLINE_COS_SIN: tl.constexpr,
    RMS_WEIGHT_OFFSET: tl.constexpr,
    gate_size: tl.constexpr,
    in_width: tl.constexpr,
    q_head_tile: tl.constexpr,
    kv_head_tile: tl.constexpr,
block_v: tl.constexpr,
    cos_block: tl.constexpr,
):
    core_idx = tl.program_id(0)
    token_begin = core_idx * tokens_per_core
    core_tokens = tl.minimum(num_tokens - token_begin, tokens_per_core)

    # Section masks are loop-invariant: compute them once per core.
    cos_offsets = tl.arange(0, half_rope_dim)
    cos_offsets_fp32 = cos_offsets.to(tl.float32)
    if is_interleaved:
        axis = cos_offsets - (cos_offsets // 3) * 3
        h_mask = (axis == 1) & (cos_offsets_fp32 < 3.0 * mrope_section_h)
        w_mask = (axis == 2) & (cos_offsets_fp32 < 3.0 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_mask = cos_offsets_fp32 < mrope_section_t
        h_mask = (mrope_section_t <= cos_offsets_fp32) & (cos_offsets_fp32 < mrope_section_t + mrope_section_h)
        w_mask = (mrope_section_t + mrope_section_h <= cos_offsets_fp32) & (
            cos_offsets_fp32 < mrope_section_t + mrope_section_h + mrope_section_w
        )

    q_rmsnorm_weight = tl.load(q_weight_ptr + tl.arange(0, head_size)).to(tl.float32) + RMS_WEIGHT_OFFSET
    k_rmsnorm_weight = tl.load(k_weight_ptr + tl.arange(0, head_size)).to(tl.float32) + RMS_WEIGHT_OFFSET
    if RMS_WEIGHT_OFFSET != 0.0:
        # Round `weight + offset` back to the weight dtype so the inline path
        # matches the numerics of the cached cos/sin call sites, which pass
        # `1.0 + weight` precomputed in the weight dtype.
        q_rmsnorm_weight = q_rmsnorm_weight.to(q_weight_ptr.dtype.element_ty).to(tl.float32)
        k_rmsnorm_weight = k_rmsnorm_weight.to(k_weight_ptr.dtype.element_ty).to(tl.float32)

    if has_bias:
        q_bias = tl.load(q_bias_ptr + tl.arange(0, head_size))
        k_bias = tl.load(k_bias_ptr + tl.arange(0, head_size))

    # Full-row section masks: the cached row is [cos | sin], and both halves
    # share the same per-frequency section assignment.
    t_mask_full = tl.broadcast_to(t_mask.reshape(1, half_rope_dim), (2, half_rope_dim)).reshape(rope_dim)
    h_mask_full = tl.broadcast_to(h_mask.reshape(1, half_rope_dim), (2, half_rope_dim)).reshape(rope_dim)

    if INLINE_COS_SIN:
        inv_freq_row = tl.load(inv_freq_ptr + cos_offsets).to(tl.float32)

    qkv_ty = in_qkv_ptr.dtype.element_ty

    # Token blocks: positions are loaded once per block as vectors and the
    # inline cos/sin transcendentals run on full-width (block, half) tiles.
    for pb in tl.range(0, tl.cdiv(core_tokens, cos_block)):
        block_tokens = tl.minimum(core_tokens - pb * cos_block, cos_block)
        # Clamp the block's token indices so the unmasked position loads stay
        # in range; out-of-range lanes are never extracted below.
        tokv = tl.minimum(token_begin + pb * cos_block + tl.arange(0, cos_block), num_tokens - 1)
        if INLINE_COS_SIN:
            pos_base = positions_ptr + tokv * positions_stride_1
            t_pos = tl.load(pos_base).to(tl.float32)
            h_pos = tl.load(pos_base + positions_stride_0).to(tl.float32)
            w_pos = tl.load(pos_base + 2 * positions_stride_0).to(tl.float32)
            selected_pos = tl.where(
                h_mask[None, :],
                h_pos[:, None],
                tl.where(w_mask[None, :], w_pos[:, None], t_pos[:, None]),
            )
            freqs = selected_pos * inv_freq_row[None, :]
            # Round to the qkv dtype so the inline cos/sin values match the
            # cached cos/sin path, which gathers values stored in qkv dtype.
            cos_tile = tl.cos(freqs).to(qkv_ty).to(tl.float32)
            sin_tile = tl.sin(freqs).to(qkv_ty).to(tl.float32)
        for i in tl.range(0, block_tokens):
            tok = token_begin + pb * cos_block + i
            in_row = in_qkv_ptr + tok * in_width

            ## cos, sin: (1, half_rope_dim) shared by every head group ##
            if INLINE_COS_SIN:
                cos_half = extract_slice(
                    cos_tile,
                    offsets=(i, 0),
                    sizes=(1, half_rope_dim),
                    strides=(1, 1),
                )
                sin_half = extract_slice(
                    sin_tile,
                    offsets=(i, 0),
                    sizes=(1, half_rope_dim),
                    strides=(1, 1),
                )
            else:
                # Load each plane's full [cos | sin] row and select per frequency.
                plane = num_tokens * rope_dim
                cache_row = cos_sin_ptr + tok * rope_dim + tl.arange(0, rope_dim)
                t_row = tl.load(cache_row)
                h_row = tl.load(cache_row + plane)
                w_row = tl.load(cache_row + 2 * plane)
                combined = tl.where(t_mask_full, t_row, tl.where(h_mask_full, h_row, w_row))
                cos_half = extract_slice(
                    combined.reshape(1, rope_dim),
                    offsets=(0, 0),
                    sizes=(1, half_rope_dim),
                    strides=(1, 1),
                ).to(tl.float32)
                sin_half = extract_slice(
                    combined.reshape(1, rope_dim),
                    offsets=(0, half_rope_dim),
                    sizes=(1, half_rope_dim),
                    strides=(1, 1),
                ).to(tl.float32)

            ## q (+ gate): one head group per inner iteration ##
            for g in range(0, num_q_heads // q_head_tile):
                head_idx = tl.arange(0, q_head_tile)
                if gate_size > 0:
                    in_q_gate = tl.load(in_row + g * (2 * q_head_tile * head_size) + tl.arange(0, 2 * q_head_tile * head_size)).reshape(
                        q_head_tile, 2 * head_size
                    )
                    in_q_tensor = extract_slice(
                        in_q_gate,
                        offsets=(0, 0),
                        sizes=(q_head_tile, head_size),
                        strides=(1, 1),
                    ).to(tl.float32)
                    in_gate_tensor = extract_slice(
                        in_q_gate,
                        offsets=(0, head_size),
                        sizes=(q_head_tile, head_size),
                        strides=(1, 1),
                    ).reshape(q_head_tile * head_size)
                    tl.store(
                        out_gate_ptr + tok * gate_size + g * q_head_tile * head_size + tl.arange(0, q_head_tile * head_size),
                        in_gate_tensor,
                    )
                else:
                    in_q_tensor = tl.load(in_row + g * q_head_tile * head_size + tl.arange(0, q_head_tile * head_size)).to(
                        tl.float32
                    ).reshape(q_head_tile, head_size)

                # q-rmsnorm
                squares = in_q_tensor * in_q_tensor
                variances = tl.sum(squares, axis=1) / head_size
                reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(q_head_tile, 1)
                q_normalized = in_q_tensor * reciprocal_std
                q_normalized = q_normalized * q_rmsnorm_weight.reshape(1, head_size)
                if has_bias:
                    q_normalized = q_normalized + q_bias.reshape(1, head_size)

                # q-mrope: rotate the two half dims directly and store each half
                q1 = extract_slice(
                    q_normalized,
                    offsets=(0, 0),
                    sizes=(q_head_tile, half_rope_dim),
                    strides=(1, 1),
                )
                q2 = extract_slice(
                    q_normalized,
                    offsets=(0, half_rope_dim),
                    sizes=(q_head_tile, half_rope_dim),
                    strides=(1, 1),
                )
                out_q_head = out_q_ptr + tok * q_size + (g * q_head_tile + head_idx)[:, None] * head_size
                if IS_PARTIAL_ROPE:
                    # Assemble the roped head with the un-roped tail and store once.
                    roped_q = tl.zeros((q_head_tile, rope_dim), dtype=tl.float32)
                    roped_q = insert_slice(
                        roped_q,
                        q1 * cos_half - q2 * sin_half,
                        offsets=(0, 0),
                        sizes=(q_head_tile, half_rope_dim),
                        strides=(1, 1),
                    )
                    roped_q = insert_slice(
                        roped_q,
                        q2 * cos_half + q1 * sin_half,
                        offsets=(0, half_rope_dim),
                        sizes=(q_head_tile, half_rope_dim),
                        strides=(1, 1),
                    )
                    out_q_tile = insert_slice(
                        q_normalized,
                        roped_q,
                        offsets=(0, 0),
                        sizes=(q_head_tile, rope_dim),
                        strides=(1, 1),
                    )
                    tl.store(out_q_head + tl.arange(0, head_size)[None, :], out_q_tile)
                else:
                    tl.store(out_q_head + tl.arange(0, half_rope_dim)[None, :], q1 * cos_half - q2 * sin_half)
                    tl.store(
                        out_q_head + half_rope_dim + tl.arange(0, half_rope_dim)[None, :],
                        q2 * cos_half + q1 * sin_half,
                    )

            ## v copy: bf16 passthrough ##
            in_v_tensor = tl.load(in_row + q_size + gate_size + kv_size + tl.arange(0, kv_size))
            tl.store(out_v_ptr + tok * kv_size + tl.arange(0, kv_size), in_v_tensor)

            ## k: one head group per inner iteration ##
            for g in range(0, num_kv_heads // kv_head_tile):
                in_k_tensor = tl.load(
                    in_row + q_size + gate_size + g * kv_head_tile * head_size + tl.arange(0, kv_head_tile * head_size)
                ).to(tl.float32).reshape(kv_head_tile, head_size)

                # k-rmsnorm
                squares = in_k_tensor * in_k_tensor
                variances = tl.sum(squares, axis=1) / head_size
                reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(kv_head_tile, 1)
                k_normalized = in_k_tensor * reciprocal_std
                k_normalized = k_normalized * k_rmsnorm_weight.reshape(1, head_size)
                if has_bias:
                    k_normalized = k_normalized + k_bias.reshape(1, head_size)

                # k-mrope
                k1 = extract_slice(
                    k_normalized,
                    offsets=(0, 0),
                    sizes=(kv_head_tile, half_rope_dim),
                    strides=(1, 1),
                )
                k2 = extract_slice(
                    k_normalized,
                    offsets=(0, half_rope_dim),
                    sizes=(kv_head_tile, half_rope_dim),
                    strides=(1, 1),
                )
                kv_head_idx = tl.arange(0, kv_head_tile)
                out_k_head = out_k_ptr + tok * kv_size + (g * kv_head_tile + kv_head_idx)[:, None] * head_size
                if IS_PARTIAL_ROPE:
                    roped_k = tl.zeros((kv_head_tile, rope_dim), dtype=tl.float32)
                    roped_k = insert_slice(
                        roped_k,
                        k1 * cos_half - k2 * sin_half,
                        offsets=(0, 0),
                        sizes=(kv_head_tile, half_rope_dim),
                        strides=(1, 1),
                    )
                    roped_k = insert_slice(
                        roped_k,
                        k2 * cos_half + k1 * sin_half,
                        offsets=(0, half_rope_dim),
                        sizes=(kv_head_tile, half_rope_dim),
                        strides=(1, 1),
                    )
                    out_k_tile = insert_slice(
                        k_normalized,
                        roped_k,
                        offsets=(0, 0),
                        sizes=(kv_head_tile, rope_dim),
                        strides=(1, 1),
                    )
                    tl.store(out_k_head + tl.arange(0, head_size)[None, :], out_k_tile)
                else:
                    tl.store(out_k_head + tl.arange(0, half_rope_dim)[None, :], k1 * cos_half - k2 * sin_half)
                    tl.store(
                        out_k_head + half_rope_dim + tl.arange(0, half_rope_dim)[None, :],
                        k2 * cos_half + k1 * sin_half,
                    )



def triton_split_qkv_rmsnorm_mrope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor | None = None,
    num_q_heads: int = 0,
    num_kv_heads: int = 0,
    head_size: int = 0,
    eps: float = 1e-6,
    mrope_section: list[int] | None = None,
    is_interleaved: bool = False,
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    has_gate: bool = False,
    positions: torch.Tensor | None = None,
    inv_freq: torch.Tensor | None = None,
    rms_weight_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    core_num = get_vectorcore_num()

    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size

    num_tokens = qkv.shape[0]

    gate_size = q_size if has_gate else 0
    expected_qkv_width = q_size + gate_size + 2 * kv_size
    if qkv.ndim != 2 or qkv.shape[1] != expected_qkv_width:
        raise ValueError(f"qkv must have shape [num_tokens, {expected_qkv_width}]")
    if q_weight.numel() != head_size or k_weight.numel() != head_size:
        raise ValueError("q_weight and k_weight must each contain head_size elements")
    if (q_bias is None) != (k_bias is None):
        raise ValueError("q_bias and k_bias must be both present or both absent")
    if q_bias is not None and (q_bias.numel() != head_size or k_bias.numel() != head_size):
        raise ValueError("q_bias and k_bias must each contain head_size elements")

    if rope_dim is None:
        rope_dim = head_size
    if mrope_section is None or len(mrope_section) != 3:
        raise ValueError("mrope_section must contain the T, H, and W sections")
    if 2 * sum(mrope_section) != rope_dim:
        raise ValueError("2 * sum(mrope_section) must equal rope_dim")
    if rope_dim > head_size or rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even and no larger than head_size")
    IS_PARTIAL_ROPE = rope_dim != head_size

    has_inline_arg = positions is not None or inv_freq is not None
    if cos_sin is not None and has_inline_arg:
        raise ValueError("cos_sin and (positions, inv_freq) are mutually exclusive")
    if cos_sin is None and (positions is None or inv_freq is None):
        raise ValueError("provide either cos_sin or both positions and inv_freq")
    INLINE_COS_SIN = cos_sin is None
    if cos_sin is not None:
        if cos_sin.ndim != 3 or cos_sin.shape != (3, num_tokens, rope_dim):
            raise ValueError("cos_sin must have shape [3, num_tokens, rope_dim]")
        cos_sin = cos_sin.contiguous()

    q_output = torch.empty(num_tokens, q_size, device=qkv.device, dtype=qkv.dtype)
    k_output = torch.empty(num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype)
    v_output = torch.empty(num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype)
    gate_output = torch.empty(num_tokens, gate_size, device=qkv.device, dtype=qkv.dtype)
    if num_tokens == 0:
        return q_output, k_output, v_output, gate_output

    positions_stride_0 = 0
    positions_stride_1 = 0
    if INLINE_COS_SIN:
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError("positions must have shape [3, num_tokens] for inline MRoPE")
        if positions.shape[1] != num_tokens:
            raise ValueError("positions.shape[1] must match qkv.shape[0]")
        if positions.stride(0) <= 0 or positions.stride(1) <= 0:
            raise ValueError("positions strides must be positive")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must use int32 or int64 indices")
        if inv_freq.numel() != rope_dim // 2:
            raise ValueError("inv_freq length must equal rope_dim // 2")
        if inv_freq.dtype != torch.float32 or not inv_freq.is_contiguous():
            raise ValueError("inv_freq must be a contiguous float32 tensor")
        positions_stride_0 = positions.stride(0)
        positions_stride_1 = positions.stride(1)

    # Split tokens across vector cores; each core walks its contiguous token
    # range one token at a time, tiling heads into UB-sized groups.
    grid_blocks = min(core_num, num_tokens)
    tokens_per_core = (num_tokens + grid_blocks - 1) // grid_blocks

    q_head_tile = _pow2_floor(_HEAD_TILE_ELEMS // head_size, 1, num_q_heads)
    while num_q_heads % q_head_tile != 0:
        q_head_tile //= 2
    kv_head_tile = _pow2_floor(_HEAD_TILE_ELEMS // head_size, 1, num_kv_heads)
    while num_kv_heads % kv_head_tile != 0:
        kv_head_tile //= 2
    block_v = _pow2_floor(_UB_BUDGET_BYTES // (4 * kv_size), 1, _MAX_V_BLOCK_T)
    cos_block = _pow2_floor(tokens_per_core, 1, _COS_BLOCK)

    has_bias = q_bias is not None

    split_qkv_rmsnorm_mrope_kernel[(grid_blocks,)](
        qkv,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        cos_sin,
        positions,
        inv_freq,
        q_output,
        k_output,
        v_output,
        gate_output,
        num_tokens,
        positions_stride_0,
        positions_stride_1,
        tokens_per_core,
        num_q_heads,
        num_kv_heads,
        head_size,
        q_size,
        kv_size,
        eps,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        has_bias,
        is_interleaved,
        rope_dim,
        rope_dim // 2,
        IS_PARTIAL_ROPE,
        INLINE_COS_SIN,
        rms_weight_offset,
        gate_size,
        expected_qkv_width,
        q_head_tile,
        kv_head_tile,
        block_v,
        cos_block,
    )

    return q_output, k_output, v_output, gate_output


def triton_split_qkv_rmsnorm_mrope_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor | None = None,
    num_q_heads: int = 0,
    num_kv_heads: int = 0,
    head_size: int = 0,
    eps: float = 1e-6,
    mrope_section: list[int] | None = None,
    is_interleaved: bool = False,
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    has_gate: bool = False,
    positions: torch.Tensor | None = None,
    inv_freq: torch.Tensor | None = None,
    rms_weight_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = qkv.shape[0]
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    gate_size = q_size if has_gate else 0

    q_output = torch.empty(
        num_tokens,
        q_size,
        device=qkv.device,
        dtype=qkv.dtype,
    )

    k_output = torch.empty(
        num_tokens,
        kv_size,
        device=qkv.device,
        dtype=qkv.dtype,
    )

    v_output = torch.empty(
        num_tokens,
        kv_size,
        device=qkv.device,
        dtype=qkv.dtype,
    )

    gate_output = torch.empty(
        num_tokens,
        gate_size,
        device=qkv.device,
        dtype=qkv.dtype,
    )

    return q_output, k_output, v_output, gate_output


direct_register_custom_op(
    op_name="triton_split_qkv_rmsnorm_mrope",
    op_func=triton_split_qkv_rmsnorm_mrope,
    fake_impl=triton_split_qkv_rmsnorm_mrope_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

