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
# Token block size for the position loads and inline cos/sin tiles.
_COS_BLOCK = 8


@triton.jit(do_not_specialize=["num_tokens", "positions_stride_0", "positions_stride_1", "grid_blocks"])
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
    grid_blocks,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    MROPE_SECTION_T: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_INTERLEAVED: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    INLINE_COS_SIN: tl.constexpr,
    RMS_WEIGHT_OFFSET: tl.constexpr,
    GATE_SIZE: tl.constexpr,
    IN_WIDTH: tl.constexpr,
    Q_HEAD_TILE: tl.constexpr,
    KV_HEAD_TILE: tl.constexpr,
    COS_BLOCK: tl.constexpr,
):
    core_idx = tl.program_id(0)
    tokens_per_core = num_tokens // grid_blocks
    extra_tokens = num_tokens % grid_blocks
    token_begin = core_idx * tokens_per_core + tl.minimum(core_idx, extra_tokens)
    core_tokens = tokens_per_core + tl.where(core_idx < extra_tokens, 1, 0)

    # Section masks are loop-invariant: compute them once per core.
    cos_offsets = tl.arange(0, HALF_ROPE_DIM)
    cos_offsets_fp32 = cos_offsets.to(tl.float32)
    if IS_INTERLEAVED:
        axis = cos_offsets - (cos_offsets // 3) * 3
        h_mask = (axis == 1) & (cos_offsets_fp32 < 3.0 * MROPE_SECTION_H)
        w_mask = (axis == 2) & (cos_offsets_fp32 < 3.0 * MROPE_SECTION_W)
        t_mask = ~(h_mask | w_mask)
    else:
        t_mask = cos_offsets_fp32 < MROPE_SECTION_T
        h_mask = (cos_offsets_fp32 >= MROPE_SECTION_T) & (cos_offsets_fp32 < MROPE_SECTION_T + MROPE_SECTION_H)
        w_mask = (cos_offsets_fp32 >= MROPE_SECTION_T + MROPE_SECTION_H) & (
            cos_offsets_fp32 < MROPE_SECTION_T + MROPE_SECTION_H + MROPE_SECTION_W
        )

    q_rmsnorm_weight = tl.load(q_weight_ptr + tl.arange(0, HEAD_SIZE)).to(tl.float32) + RMS_WEIGHT_OFFSET
    k_rmsnorm_weight = tl.load(k_weight_ptr + tl.arange(0, HEAD_SIZE)).to(tl.float32) + RMS_WEIGHT_OFFSET
    if RMS_WEIGHT_OFFSET != 0.0:
        # Round `weight + offset` back to the weight dtype so the inline path
        # matches the numerics of the cached cos/sin call sites, which pass
        # `1.0 + weight` precomputed in the weight dtype.
        q_rmsnorm_weight = q_rmsnorm_weight.to(q_weight_ptr.type.element_ty).to(tl.float32)
        k_rmsnorm_weight = k_rmsnorm_weight.to(k_weight_ptr.type.element_ty).to(tl.float32)

    if HAS_BIAS:
        q_bias = tl.load(q_bias_ptr + tl.arange(0, HEAD_SIZE))
        k_bias = tl.load(k_bias_ptr + tl.arange(0, HEAD_SIZE))

    # Full-row section masks: the cached row is [cos | sin], and both halves
    # share the same per-frequency section assignment.
    t_mask_full = tl.broadcast_to(t_mask.reshape(1, HALF_ROPE_DIM), (2, HALF_ROPE_DIM)).reshape(ROPE_DIM)
    h_mask_full = tl.broadcast_to(h_mask.reshape(1, HALF_ROPE_DIM), (2, HALF_ROPE_DIM)).reshape(ROPE_DIM)
    w_mask_full = ~(t_mask_full | h_mask_full)

    if INLINE_COS_SIN:
        inv_freq_row = tl.load(inv_freq_ptr + cos_offsets).to(tl.float32)

    qkv_ty = in_qkv_ptr.type.element_ty

    # Token blocks: positions are loaded once per block as vectors and the
    # inline cos/sin transcendentals run on full-width (block, half) tiles.
    for pb in tl.range(0, tl.cdiv(core_tokens, COS_BLOCK)):
        block_tokens = tl.minimum(core_tokens - pb * COS_BLOCK, COS_BLOCK)
        # Clamp the block's token indices so the unmasked position loads stay
        # in range; out-of-range lanes are never extracted below.
        tokv = tl.minimum(token_begin + pb * COS_BLOCK + tl.arange(0, COS_BLOCK), num_tokens - 1)
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
            tok = token_begin + pb * COS_BLOCK + i
            in_row = in_qkv_ptr + tok * IN_WIDTH

            ## cos, sin: (1, HALF_ROPE_DIM) shared by every head group ##
            if INLINE_COS_SIN:
                cos_half = extract_slice(
                    cos_tile,
                    offsets=(i, 0),
                    sizes=(1, HALF_ROPE_DIM),
                    strides=(1, 1),
                )
                sin_half = extract_slice(
                    sin_tile,
                    offsets=(i, 0),
                    sizes=(1, HALF_ROPE_DIM),
                    strides=(1, 1),
                )
            else:
                # Load each plane's full [cos | sin] row and select per frequency.
                plane = num_tokens * ROPE_DIM
                cache_row = cos_sin_ptr + tok * ROPE_DIM + tl.arange(0, ROPE_DIM)
                t_row = tl.load(cache_row, mask=t_mask_full, other=0)
                h_row = tl.load(cache_row + plane, mask=h_mask_full, other=0)
                w_row = tl.load(cache_row + 2 * plane, mask=w_mask_full, other=0)
                combined = t_row + h_row + w_row
                cos_half = extract_slice(
                    combined.reshape(1, ROPE_DIM),
                    offsets=(0, 0),
                    sizes=(1, HALF_ROPE_DIM),
                    strides=(1, 1),
                ).to(tl.float32)
                sin_half = extract_slice(
                    combined.reshape(1, ROPE_DIM),
                    offsets=(0, HALF_ROPE_DIM),
                    sizes=(1, HALF_ROPE_DIM),
                    strides=(1, 1),
                ).to(tl.float32)

            ## q (+ gate): one head group per inner iteration ##
            for g in range(0, NUM_Q_HEADS // Q_HEAD_TILE):
                head_idx = tl.arange(0, Q_HEAD_TILE)
                if GATE_SIZE > 0:
                    # Each head stores a query followed by its output gate.
                    Q_AND_GATE_COMPONENTS: tl.constexpr = 2
                    Q_GATE_WIDTH: tl.constexpr = Q_AND_GATE_COMPONENTS * HEAD_SIZE
                    in_q_gate = tl.load(
                        in_row + g * (Q_HEAD_TILE * Q_GATE_WIDTH) + tl.arange(0, Q_HEAD_TILE * Q_GATE_WIDTH)
                    ).reshape(Q_HEAD_TILE, Q_GATE_WIDTH)
                    in_q_tensor = extract_slice(
                        in_q_gate,
                        offsets=(0, 0),
                        sizes=(Q_HEAD_TILE, HEAD_SIZE),
                        strides=(1, 1),
                    ).to(tl.float32)
                    in_gate_tensor = extract_slice(
                        in_q_gate,
                        offsets=(0, HEAD_SIZE),
                        sizes=(Q_HEAD_TILE, HEAD_SIZE),
                        strides=(1, 1),
                    ).reshape(Q_HEAD_TILE * HEAD_SIZE)
                    tl.store(
                        out_gate_ptr
                        + tok * GATE_SIZE
                        + g * Q_HEAD_TILE * HEAD_SIZE
                        + tl.arange(0, Q_HEAD_TILE * HEAD_SIZE),
                        in_gate_tensor,
                    )
                else:
                    in_q_tensor = (
                        tl.load(in_row + g * Q_HEAD_TILE * HEAD_SIZE + tl.arange(0, Q_HEAD_TILE * HEAD_SIZE))
                        .to(tl.float32)
                        .reshape(Q_HEAD_TILE, HEAD_SIZE)
                    )

                # q-rmsnorm
                squares = in_q_tensor * in_q_tensor
                variances = tl.sum(squares, axis=1) / HEAD_SIZE
                reciprocal_std = (1 / tl.sqrt(variances + EPS)).reshape(Q_HEAD_TILE, 1)
                q_normalized = in_q_tensor * reciprocal_std
                q_normalized = q_normalized * q_rmsnorm_weight.reshape(1, HEAD_SIZE)
                if HAS_BIAS:
                    q_normalized = q_normalized + q_bias.reshape(1, HEAD_SIZE)

                # q-mrope: rotate the two half dims directly and store each half
                q1 = extract_slice(
                    q_normalized,
                    offsets=(0, 0),
                    sizes=(Q_HEAD_TILE, HALF_ROPE_DIM),
                    strides=(1, 1),
                )
                q2 = extract_slice(
                    q_normalized,
                    offsets=(0, HALF_ROPE_DIM),
                    sizes=(Q_HEAD_TILE, HALF_ROPE_DIM),
                    strides=(1, 1),
                )
                out_q_head = out_q_ptr + tok * Q_SIZE + (g * Q_HEAD_TILE + head_idx)[:, None] * HEAD_SIZE
                if IS_PARTIAL_ROPE:
                    # Assemble the roped head with the un-roped tail and store once.
                    roped_q = tl.zeros((Q_HEAD_TILE, ROPE_DIM), dtype=tl.float32)
                    roped_q = insert_slice(
                        roped_q,
                        q1 * cos_half - q2 * sin_half,
                        offsets=(0, 0),
                        sizes=(Q_HEAD_TILE, HALF_ROPE_DIM),
                        strides=(1, 1),
                    )
                    roped_q = insert_slice(
                        roped_q,
                        q2 * cos_half + q1 * sin_half,
                        offsets=(0, HALF_ROPE_DIM),
                        sizes=(Q_HEAD_TILE, HALF_ROPE_DIM),
                        strides=(1, 1),
                    )
                    out_q_tile = insert_slice(
                        q_normalized,
                        roped_q,
                        offsets=(0, 0),
                        sizes=(Q_HEAD_TILE, ROPE_DIM),
                        strides=(1, 1),
                    )
                    tl.store(out_q_head + tl.arange(0, HEAD_SIZE)[None, :], out_q_tile)
                else:
                    tl.store(out_q_head + tl.arange(0, HALF_ROPE_DIM)[None, :], q1 * cos_half - q2 * sin_half)
                    tl.store(
                        out_q_head + HALF_ROPE_DIM + tl.arange(0, HALF_ROPE_DIM)[None, :],
                        q2 * cos_half + q1 * sin_half,
                    )

            ## v copy: passthrough in the input dtype ##
            in_v_tensor = tl.load(in_row + Q_SIZE + GATE_SIZE + KV_SIZE + tl.arange(0, KV_SIZE))
            tl.store(out_v_ptr + tok * KV_SIZE + tl.arange(0, KV_SIZE), in_v_tensor)

            ## k: one head group per inner iteration ##
            for g in range(0, NUM_KV_HEADS // KV_HEAD_TILE):
                in_k_tensor = (
                    tl.load(
                        in_row
                        + Q_SIZE
                        + GATE_SIZE
                        + g * KV_HEAD_TILE * HEAD_SIZE
                        + tl.arange(0, KV_HEAD_TILE * HEAD_SIZE)
                    )
                    .to(tl.float32)
                    .reshape(KV_HEAD_TILE, HEAD_SIZE)
                )

                # k-rmsnorm
                squares = in_k_tensor * in_k_tensor
                variances = tl.sum(squares, axis=1) / HEAD_SIZE
                reciprocal_std = (1 / tl.sqrt(variances + EPS)).reshape(KV_HEAD_TILE, 1)
                k_normalized = in_k_tensor * reciprocal_std
                k_normalized = k_normalized * k_rmsnorm_weight.reshape(1, HEAD_SIZE)
                if HAS_BIAS:
                    k_normalized = k_normalized + k_bias.reshape(1, HEAD_SIZE)

                # k-mrope
                k1 = extract_slice(
                    k_normalized,
                    offsets=(0, 0),
                    sizes=(KV_HEAD_TILE, HALF_ROPE_DIM),
                    strides=(1, 1),
                )
                k2 = extract_slice(
                    k_normalized,
                    offsets=(0, HALF_ROPE_DIM),
                    sizes=(KV_HEAD_TILE, HALF_ROPE_DIM),
                    strides=(1, 1),
                )
                kv_head_idx = tl.arange(0, KV_HEAD_TILE)
                out_k_head = out_k_ptr + tok * KV_SIZE + (g * KV_HEAD_TILE + kv_head_idx)[:, None] * HEAD_SIZE
                if IS_PARTIAL_ROPE:
                    roped_k = tl.zeros((KV_HEAD_TILE, ROPE_DIM), dtype=tl.float32)
                    roped_k = insert_slice(
                        roped_k,
                        k1 * cos_half - k2 * sin_half,
                        offsets=(0, 0),
                        sizes=(KV_HEAD_TILE, HALF_ROPE_DIM),
                        strides=(1, 1),
                    )
                    roped_k = insert_slice(
                        roped_k,
                        k2 * cos_half + k1 * sin_half,
                        offsets=(0, HALF_ROPE_DIM),
                        sizes=(KV_HEAD_TILE, HALF_ROPE_DIM),
                        strides=(1, 1),
                    )
                    out_k_tile = insert_slice(
                        k_normalized,
                        roped_k,
                        offsets=(0, 0),
                        sizes=(KV_HEAD_TILE, ROPE_DIM),
                        strides=(1, 1),
                    )
                    tl.store(out_k_head + tl.arange(0, HEAD_SIZE)[None, :], out_k_tile)
                else:
                    tl.store(out_k_head + tl.arange(0, HALF_ROPE_DIM)[None, :], k1 * cos_half - k2 * sin_half)
                    tl.store(
                        out_k_head + HALF_ROPE_DIM + tl.arange(0, HALF_ROPE_DIM)[None, :],
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
    if num_q_heads <= 0 or num_kv_heads <= 0 or head_size <= 0:
        raise ValueError("num_q_heads, num_kv_heads, and head_size must be positive")
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    # The kernel uses unpadded tl.arange for head_size and the V row.
    if head_size & (head_size - 1) or kv_size & (kv_size - 1):
        raise ValueError("head_size and num_kv_heads must be powers of two")
    gate_size = q_size if has_gate else 0
    expected_qkv_width = q_size + gate_size + 2 * kv_size
    if qkv.ndim != 2 or qkv.shape[1] != expected_qkv_width:
        raise ValueError(f"qkv must have shape [num_tokens, {expected_qkv_width}]")
    num_tokens = qkv.shape[0]
    if (q_bias is None) != (k_bias is None):
        raise ValueError("q_bias and k_bias must be both present or both absent")
    for name, tensor in (("q_weight", q_weight), ("k_weight", k_weight), ("q_bias", q_bias), ("k_bias", k_bias)):
        if tensor is not None and (
            tensor.numel() != head_size or not tensor.is_contiguous() or tensor.device != qkv.device
        ):
            raise ValueError(f"{name} must be contiguous, contain head_size elements, and be on qkv.device")

    if rope_dim is None:
        rope_dim = head_size
    if not 2 <= rope_dim <= head_size or rope_dim & (rope_dim - 1):
        raise ValueError("rope_dim must be a power of two between 2 and head_size")
    if mrope_section is None or len(mrope_section) != 3 or min(mrope_section) < 0:
        raise ValueError("mrope_section must contain three non-negative T, H, and W section lengths")
    if 2 * sum(mrope_section) != rope_dim:
        raise ValueError("2 * sum(mrope_section) must equal rope_dim")
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
        if cos_sin.device != qkv.device:
            raise ValueError("cos_sin must be on the same device as qkv")

    positions_stride_0 = 0
    positions_stride_1 = 0
    if INLINE_COS_SIN:
        assert positions is not None
        assert inv_freq is not None
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError("positions must have shape [3, num_tokens] for inline MRoPE")
        if positions.device != qkv.device or inv_freq.device != qkv.device:
            raise ValueError("positions and inv_freq must be on the same device as qkv")
        if positions.shape[1] != num_tokens:
            raise ValueError("positions.shape[1] must match qkv.shape[0]")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must use int32 or int64 indices")
        if inv_freq.numel() != rope_dim // 2:
            raise ValueError("inv_freq length must equal rope_dim // 2")
        if inv_freq.dtype != torch.float32 or not inv_freq.is_contiguous():
            raise ValueError("inv_freq must be a contiguous float32 tensor")
        positions_stride_0 = positions.stride(0)
        positions_stride_1 = positions.stride(1)

    q_output = qkv.new_empty((num_tokens, q_size))
    k_output = qkv.new_empty((num_tokens, kv_size))
    v_output = qkv.new_empty((num_tokens, kv_size))
    gate_output = qkv.new_empty((num_tokens, gate_size))
    if num_tokens == 0:
        return q_output, k_output, v_output, gate_output

    qkv = qkv.contiguous()
    if cos_sin is not None:
        cos_sin = cos_sin.contiguous()

    # Split tokens across vector cores; each core walks its contiguous token
    # range one token at a time, tiling heads into UB-sized groups.
    grid_blocks = min(get_vectorcore_num(), num_tokens)
    tokens_per_core = (num_tokens + grid_blocks - 1) // grid_blocks

    # next_power_of_2(n + 1) // 2 rounds n down, including exact powers of two.
    q_head_tile = triton.next_power_of_2(max(1, min(_HEAD_TILE_ELEMS // head_size, num_q_heads)) + 1) // 2
    while num_q_heads % q_head_tile != 0:
        q_head_tile //= 2
    kv_head_tile = triton.next_power_of_2(max(1, min(_HEAD_TILE_ELEMS // head_size, num_kv_heads)) + 1) // 2
    while num_kv_heads % kv_head_tile != 0:
        kv_head_tile //= 2
    cos_block = triton.next_power_of_2(min(tokens_per_core, _COS_BLOCK) + 1) // 2

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
        grid_blocks,
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
