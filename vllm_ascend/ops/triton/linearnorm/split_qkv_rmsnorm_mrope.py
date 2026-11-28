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


@triton.jit(
    do_not_specialize=["num_tokens", "front_core_num", "num_tokens_each_front_core", "num_tokens_each_tail_core"]
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
    front_core_num,
    num_tokens_each_front_core,
    num_tokens_each_tail_core,
    positions_stride_0,
    positions_stride_1,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    eps: tl.constexpr,
    mrope_section_t,
    mrope_section_h,
    mrope_section_w,
    has_bias: tl.constexpr,
    is_interleaved: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    gate_size: tl.constexpr,
    INLINE_COS_SIN: tl.constexpr,
):
    block_idx = tl.program_id(0)

    loop_num = num_tokens_each_front_core
    if block_idx >= front_core_num:
        loop_num = num_tokens_each_tail_core

    block_offset = num_tokens_each_front_core * block_idx
    if block_idx >= front_core_num:
        block_offset = (
            num_tokens_each_front_core * front_core_num + (block_idx - front_core_num) * num_tokens_each_tail_core
        )

    q_rmsnorm_weight = tl.load(q_weight_ptr + tl.arange(0, head_size))
    k_rmsnorm_weight = tl.load(k_weight_ptr + tl.arange(0, head_size))

    if has_bias:
        q_bias = tl.load(q_bias_ptr + tl.arange(0, head_size))
        k_bias = tl.load(k_bias_ptr + tl.arange(0, head_size))

    if INLINE_COS_SIN:
        inv_freq = tl.load(inv_freq_ptr + tl.arange(0, half_rope_dim)).to(tl.float32)

    for index in range(loop_num):
        ## load ##
        # q
        in_q_offset = in_qkv_ptr + (block_offset + index) * (q_size + gate_size + 2 * kv_size)
        if gate_size > 0:
            # Each head packs a query and its output gate.
            Q_AND_GATE_COMPONENTS: tl.constexpr = 2
            in_q_gate_tensor = (
                tl.load(in_q_offset + tl.arange(0, q_size + gate_size))
                .to(tl.float32)
                .reshape(num_q_heads, head_size * Q_AND_GATE_COMPONENTS)
            )
            in_q_tensor = extract_slice(
                in_q_gate_tensor,
                offsets=(0, 0),
                sizes=(num_q_heads, head_size),
                strides=(1, 1),
            )
            in_gate_tensor = extract_slice(
                in_q_gate_tensor,
                offsets=(0, head_size),
                sizes=(num_q_heads, head_size),
                strides=(1, 1),
            ).reshape(q_size)
        else:
            in_q_tensor = tl.load(in_q_offset + tl.arange(0, q_size)).to(tl.float32).reshape(num_q_heads, head_size)

        # k
        in_k_offset = in_q_offset + q_size + gate_size
        in_k_tensor = tl.load(in_k_offset + tl.arange(0, kv_size)).to(tl.float32).reshape(num_kv_heads, head_size)
        # v
        in_v_offset = in_k_offset + kv_size
        in_v_tensor = tl.load(in_v_offset + tl.arange(0, kv_size))

        # cos, sin
        cos_offsets = tl.arange(0, half_rope_dim)
        if is_interleaved:
            h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
            w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
            t_mask = ~(h_mask | w_mask)
        else:
            t_mask = cos_offsets < mrope_section_t
            h_mask = (mrope_section_t - 1 < cos_offsets) & (cos_offsets < mrope_section_t + mrope_section_h)
            w_mask = (mrope_section_t + mrope_section_h - 1 < cos_offsets) & (
                cos_offsets < mrope_section_t + mrope_section_h + mrope_section_w
            )

        if INLINE_COS_SIN:
            pos_offset = positions_ptr + (block_offset + index) * positions_stride_1
            t_pos = tl.load(pos_offset).to(tl.float32)
            h_pos = tl.load(pos_offset + positions_stride_0).to(tl.float32)
            w_pos = tl.load(pos_offset + 2 * positions_stride_0).to(tl.float32)
            selected_pos = tl.where(h_mask, h_pos, tl.where(w_mask, w_pos, t_pos))
            phase = selected_pos * inv_freq
            cos_half = tl.cos(phase).to(in_qkv_ptr.type.element_ty).to(tl.float32)
            sin_half = tl.sin(phase).to(in_qkv_ptr.type.element_ty).to(tl.float32)
            cos_tensor = tl.broadcast_to(cos_half.reshape(1, half_rope_dim), (2, half_rope_dim)).reshape(1, rope_dim)
            sin_tensor = tl.broadcast_to(sin_half.reshape(1, half_rope_dim), (2, half_rope_dim)).reshape(1, rope_dim)
        else:
            t_cos_offset = cos_sin_ptr + (block_offset + index) * rope_dim
            h_cos_offset = t_cos_offset + num_tokens * rope_dim
            w_cos_offset = h_cos_offset + num_tokens * rope_dim

            t_sin_offset = cos_sin_ptr + (block_offset + index) * rope_dim + half_rope_dim
            h_sin_offset = t_sin_offset + num_tokens * rope_dim
            w_sin_offset = h_sin_offset + num_tokens * rope_dim

            t_cos_tensor = tl.load(t_cos_offset + cos_offsets, mask=t_mask, other=0)
            h_cos_tensor = tl.load(h_cos_offset + cos_offsets, mask=h_mask, other=0)
            w_cos_tensor = tl.load(w_cos_offset + cos_offsets, mask=w_mask, other=0)
            t_sin_tensor = tl.load(t_sin_offset + cos_offsets, mask=t_mask, other=0)
            h_sin_tensor = tl.load(h_sin_offset + cos_offsets, mask=h_mask, other=0)
            w_sin_tensor = tl.load(w_sin_offset + cos_offsets, mask=w_mask, other=0)

            cos_tensor = (t_cos_tensor + h_cos_tensor + w_cos_tensor).to(tl.float32).reshape(1, half_rope_dim)
            cos_tensor = tl.broadcast_to(cos_tensor, (2, half_rope_dim)).reshape(1, rope_dim)

            sin_tensor = (t_sin_tensor + h_sin_tensor + w_sin_tensor).to(tl.float32).reshape(1, half_rope_dim)
            sin_tensor = tl.broadcast_to(sin_tensor, (2, half_rope_dim)).reshape(1, rope_dim)

        ## compute ##
        # q-rmsnorm
        squares = in_q_tensor * in_q_tensor
        variances = tl.sum(squares, axis=1) / head_size
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_q_heads, 1)
        q_normalized = in_q_tensor * reciprocal_std
        q_normalized = q_normalized * q_rmsnorm_weight
        if has_bias:
            q_normalized = q_normalized + q_bias

        # k-rmsnorm
        squares = in_k_tensor * in_k_tensor
        variances = tl.sum(squares, axis=1) / head_size
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_kv_heads, 1)
        k_normalized = in_k_tensor * reciprocal_std
        k_normalized = k_normalized * k_rmsnorm_weight
        if has_bias:
            k_normalized = k_normalized + k_bias

        # q-mrope
        x1 = extract_slice(
            q_normalized,
            offsets=(0, 0),
            sizes=(num_q_heads, half_rope_dim),
            strides=(1, 1),
        )
        x2 = extract_slice(
            q_normalized,
            offsets=(0, half_rope_dim),
            sizes=(num_q_heads, half_rope_dim),
            strides=(1, 1),
        )
        cat_x = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
        cat_x = insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(num_q_heads, half_rope_dim),
            strides=(1, 1),
        )
        cat_x = insert_slice(
            cat_x,
            x1,
            offsets=(0, half_rope_dim),
            sizes=(num_q_heads, half_rope_dim),
            strides=(1, 1),
        )
        if IS_PARTIAL_ROPE:
            orig_qk = extract_slice(
                q_normalized,
                offsets=(0, 0),
                sizes=(num_q_heads, rope_dim),
                strides=(1, 1),
            )
        else:
            orig_qk = q_normalized
        roped_q = cat_x * sin_tensor + orig_qk * cos_tensor

        # k-mrope
        y1 = extract_slice(
            k_normalized,
            offsets=(0, 0),
            sizes=(num_kv_heads, half_rope_dim),
            strides=(1, 1),
        )
        y2 = extract_slice(
            k_normalized,
            offsets=(0, half_rope_dim),
            sizes=(num_kv_heads, half_rope_dim),
            strides=(1, 1),
        )
        cat_y = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
        cat_y = insert_slice(
            cat_y,
            -y2,
            offsets=(0, 0),
            sizes=(num_kv_heads, half_rope_dim),
            strides=(1, 1),
        )
        cat_y = insert_slice(
            cat_y,
            y1,
            offsets=(0, half_rope_dim),
            sizes=(num_kv_heads, half_rope_dim),
            strides=(1, 1),
        )
        if IS_PARTIAL_ROPE:
            orig_qk = extract_slice(
                k_normalized,
                offsets=(0, 0),
                sizes=(num_kv_heads, rope_dim),
                strides=(1, 1),
            )
        else:
            orig_qk = k_normalized
        roped_k = cat_y * sin_tensor + orig_qk * cos_tensor

        if IS_PARTIAL_ROPE:
            q_normalized = insert_slice(
                q_normalized,
                roped_q,
                offsets=(0, 0),
                sizes=(num_q_heads, rope_dim),
                strides=(1, 1),
            )
            k_normalized = insert_slice(
                k_normalized,
                roped_k,
                offsets=(0, 0),
                sizes=(num_kv_heads, rope_dim),
                strides=(1, 1),
            )
        else:
            q_normalized = roped_q
            k_normalized = roped_k

        ## store ##
        # out_q
        out_q_offset = out_q_ptr + (block_offset + index) * q_size
        out_q_indices = tl.arange(0, q_size)
        tl.store(out_q_offset + out_q_indices, q_normalized.reshape(q_size))

        # out_k
        out_k_offset = out_k_ptr + (block_offset + index) * kv_size
        out_k_indices = tl.arange(0, kv_size)
        tl.store(out_k_offset + out_k_indices, k_normalized.reshape(kv_size))

        # out_v
        out_v_offset = out_v_ptr + (block_offset + index) * kv_size
        tl.store(out_v_offset + tl.arange(0, kv_size), in_v_tensor)

        # out_gate
        if gate_size > 0:
            out_gate_offset = out_gate_ptr + (block_offset + index) * gate_size
            tl.store(out_gate_offset + tl.arange(0, gate_size), in_gate_tensor)


def triton_split_qkv_rmsnorm_mrope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor | None,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    has_gate: bool = False,
    positions: torch.Tensor | None = None,
    inv_freq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    core_num = get_vectorcore_num()

    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    if num_q_heads <= 0 or num_kv_heads <= 0 or head_size <= 0:
        raise ValueError("head counts and head_size must be positive")
    if q_size & (q_size - 1) or kv_size & (kv_size - 1) or head_size & (head_size - 1):
        raise ValueError("head_size and Q/KV widths must be powers of two")
    gate_size = q_size if has_gate else 0

    if rope_dim is None:
        rope_dim = head_size
    if not 2 <= rope_dim <= head_size or rope_dim & (rope_dim - 1):
        raise ValueError("rope_dim must be a power of two between 2 and head_size")
    if mrope_section is None or len(mrope_section) != 3 or min(mrope_section) < 0 or 2 * sum(mrope_section) != rope_dim:
        raise ValueError("invalid mrope_section")
    if qkv.ndim != 2 or qkv.shape[1] != q_size + gate_size + 2 * kv_size:
        raise ValueError("qkv has an invalid shape")
    num_tokens = qkv.shape[0]
    if (q_bias is None) != (k_bias is None):
        raise ValueError("q_bias and k_bias must be both present or both absent")
    for name, tensor in (("q_weight", q_weight), ("k_weight", k_weight), ("q_bias", q_bias), ("k_bias", k_bias)):
        if tensor is not None and (
            tensor.numel() != head_size or not tensor.is_contiguous() or tensor.device != qkv.device
        ):
            raise ValueError(f"{name} must be contiguous, contain head_size elements, and be on qkv.device")
    if cos_sin is not None and (positions is not None or inv_freq is not None):
        raise ValueError("cos_sin and (positions, inv_freq) are mutually exclusive")
    if cos_sin is None and (positions is None or inv_freq is None):
        raise ValueError("provide either cos_sin or both positions and inv_freq")
    INLINE_COS_SIN = cos_sin is None
    positions_stride_0 = positions_stride_1 = 0
    if INLINE_COS_SIN:
        assert positions is not None and inv_freq is not None
        if positions.ndim != 2 or positions.shape != (3, num_tokens):
            raise ValueError("positions must have shape [3, num_tokens]")
        if positions.device != qkv.device or inv_freq.device != qkv.device:
            raise ValueError("positions and inv_freq must be on the same device as qkv")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must use int32 or int64 indices")
        if inv_freq.numel() != rope_dim // 2 or inv_freq.dtype != torch.float32 or not inv_freq.is_contiguous():
            raise ValueError("inv_freq must be a contiguous float32 tensor of length rope_dim // 2")
        positions_stride_0, positions_stride_1 = positions.stride()
    else:
        assert cos_sin is not None
        if cos_sin.shape != (3, num_tokens, rope_dim) or cos_sin.device != qkv.device:
            raise ValueError("cos_sin must have shape [3, num_tokens, rope_dim] and be on qkv.device")
    IS_PARTIAL_ROPE = rope_dim != head_size

    front_core_num = core_num
    if num_tokens % core_num != 0:
        front_core_num = num_tokens % core_num

    num_tokens_each_front_core = (num_tokens + core_num - 1) // core_num

    tail_core_num = 0
    if num_tokens > core_num:
        tail_core_num = core_num - front_core_num

    num_tokens_each_tail_core = num_tokens // core_num

    q_output = torch.empty(num_tokens, q_size, device=qkv.device, dtype=qkv.dtype)
    k_output = torch.empty(num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype)
    v_output = torch.empty(num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype)
    gate_output = torch.empty(num_tokens, gate_size, device=qkv.device, dtype=qkv.dtype)
    if num_tokens == 0:
        return q_output, k_output, v_output, gate_output
    qkv = qkv.contiguous()
    if cos_sin is not None:
        cos_sin = cos_sin.contiguous()

    total_core = front_core_num + tail_core_num
    block_dim = core_num
    if total_core < core_num:
        block_dim = total_core

    has_bias = q_bias is not None

    split_qkv_rmsnorm_mrope_kernel[(block_dim,)](
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
        front_core_num,
        num_tokens_each_front_core,
        num_tokens_each_tail_core,
        positions_stride_0,
        positions_stride_1,
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
        gate_size,
        INLINE_COS_SIN,
    )

    return q_output, k_output, v_output, gate_output


def triton_split_qkv_rmsnorm_mrope_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor | None,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    has_gate: bool = False,
    positions: torch.Tensor | None = None,
    inv_freq: torch.Tensor | None = None,
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
