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
import triton  # type: ignore
import triton.language as tl  # type: ignore
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["num_tokens",
                               "front_core_num",
                               "num_tokens_each_front_core",
                               "num_tokens_each_tail_core"])
def split_qkv_rmsnorm_mrope_kernel(
    in_qkv_gm_ptr: torch.Tensor,
    q_weight_ptr: torch.Tensor,
    q_bias_ptr: torch.Tensor,
    k_weight_ptr: torch.Tensor,
    k_bias_ptr: torch.Tensor,
    cos_sin_ptr: torch.Tensor,
    out_q_ptr: torch.Tensor,
    out_k_ptr: torch.Tensor,
    out_v_ptr: torch.Tensor,
    batch_size,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_qk_heads: tl.constexpr,
    head_size: tl.constexpr,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    qk_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    has_bias: tl.constexpr,
    thw_mask_ptr: torch.Tensor,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    batch_size_per_vec,
    batch_size_per_iter_per_vec,
    cos_sin_batch_per_iter_per_vec,
    cos_sin_size_batch_per_iter_per_vec,
    iter_num_per_vec,
    qk_head_nums_per_iter_per_vec,
    v_batch_size_per_iter_per_vec,
    v_iter_num_per_vec,
    t_batch_offset_end,
    h_batch_offset_end,
):
    row_pid = tl.program_id(0)
    weight_idx = tl.arange(0, head_size)
    q_rmsnorm_weight = tl.load(q_weight_ptr + weight_idx)
    k_rmsnorm_weight = tl.load(k_weight_ptr + weight_idx)

    if has_bias:
        q_bias = tl.load(q_bias_ptr + weight_idx)
        k_bias = tl.load(k_bias_ptr + weight_idx)

    cos_sin_nblk_idx = tl.arange(0, half_rope_dim)

    thw_mmask = tl.arange(0, 3)
    thw_idx = thw_mmask[:, None] * half_rope_dim + cos_sin_nblk_idx[None, :]
    thw_mask = tl.load(thw_mask_ptr + thw_idx)
    t_nmask = tl.extract_slice(thw_mask, offsets=(0, 0), sizes=(1, half_rope_dim), strides=(1, 1))
    h_nmask = tl.extract_slice(thw_mask, offsets=(1, 0), sizes=(1, half_rope_dim), strides=(1, 1))
    w_nmask = tl.extract_slice(thw_mask, offsets=(2, 0), sizes=(1, half_rope_dim), strides=(1, 1))
    input_batch_offset = row_pid * batch_size_per_vec
    cos_sin_batch_offset = (input_batch_offset + input_batch_offset) * half_rope_dim

    input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)

    mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(0, qk_size)   
    nmask = nblk_idx < total_hidden_size

    # cos_sin_mask = tl.zeros((1, half_rope_dim), dtype=tl.bfloat16)

    output_q_nblk_idx = tl.arange(0, q_size)
    output_q_nmask = output_q_nblk_idx < q_size
    output_kv_nblk_idx = tl.arange(0, kv_size)
    output_kv_nmask = output_kv_nblk_idx < kv_size

    batch_indices = tl.arange(0, cos_sin_batch_per_iter_per_vec) * half_rope_dim
    t_mblk_idx = batch_indices + cos_sin_batch_offset
    h_mblk_idx = t_mblk_idx + t_batch_offset_end
    w_mblk_idx = t_mblk_idx + h_batch_offset_end
    
    for index in tl.range(iter_num_per_vec):
        # ## load ##
        cur_mblk_idx = mblk_idx + index * batch_size_per_iter_per_vec
        mmask = cur_mblk_idx < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        idx = cur_mblk_idx[:, None] * total_hidden_size + nblk_idx[None, :]
        values_tmp1 = tl.load(in_qkv_gm_ptr + idx, mask=mask).reshape(qk_head_nums_per_iter_per_vec, head_size)

        # # cos, sin
        mblk_idx_offset = index * cos_sin_size_batch_per_iter_per_vec
        t_idx = (t_mblk_idx + mblk_idx_offset)[:, None] + cos_sin_nblk_idx[None, :]
        t_tensor = tl.load(cos_sin_ptr + t_idx)
        
        h_idx = (h_mblk_idx + mblk_idx_offset)[:, None] + cos_sin_nblk_idx[None, :]
        h_tensor = tl.load(cos_sin_ptr + h_idx)

        w_idx = (w_mblk_idx + mblk_idx_offset)[:, None] + cos_sin_nblk_idx[None, :]
        w_tensor = tl.load(cos_sin_ptr + w_idx)
        t_tensor *= t_nmask
        h_tensor *= h_nmask
        w_tensor *= w_nmask
        thw_tensor = h_tensor + w_tensor + t_tensor
        thw_tensor = (thw_tensor).reshape(batch_size_per_iter_per_vec, rope_dim)
        
        cos_tensor = tl.extract_slice(
                        thw_tensor,
                        offsets=(0, 0),
                        sizes=(batch_size_per_iter_per_vec, half_rope_dim),
                        strides=(1, 1),
                    ).to(tl.float32)
        sin_tensor = tl.extract_slice(
                        thw_tensor,
                        offsets=(0, half_rope_dim),
                        sizes=(batch_size_per_iter_per_vec, half_rope_dim),
                        strides=(1, 1),
                    ).to(tl.float32)
        cos_tensor = (tl.broadcast_to(cos_tensor.reshape(batch_size_per_iter_per_vec, 1, half_rope_dim), (batch_size_per_iter_per_vec, 2, half_rope_dim))
                      .reshape(batch_size_per_iter_per_vec, 1, rope_dim))
        sin_tensor = (tl.broadcast_to(sin_tensor.reshape(batch_size_per_iter_per_vec, 1, half_rope_dim), (batch_size_per_iter_per_vec, 2, half_rope_dim))
                      .reshape(batch_size_per_iter_per_vec, 1, rope_dim))

        # ## compute ##
        # # q-rmsnorm
        normalized_values = values_tmp1.to(tl.float32)
        normalized_values = normalized_values * normalized_values
        normalized_values = tl.sum(normalized_values, axis=1) / head_size
        normalized_values = values_tmp1 / tl.sqrt(normalized_values + eps).reshape(qk_head_nums_per_iter_per_vec, 1)

        # q
        normalized_values_tmp = tl.extract_slice(
                normalized_values.reshape(batch_size_per_iter_per_vec, num_qk_heads, head_size),
                offsets=(0, 0, 0),
                sizes=(batch_size_per_iter_per_vec, num_q_heads, head_size),
                strides=(1, 1, 1),
            )
        normalized_values_tmp = normalized_values_tmp * q_rmsnorm_weight
        if has_bias:
            normalized_values_tmp = normalized_values_tmp + q_bias
        
        values_tmp = tl.zeros((batch_size_per_iter_per_vec, num_q_heads, rope_dim),
                            dtype=tl.float32)

        x = tl.extract_slice(
                 normalized_values_tmp,
                 offsets=(0, 0, 0),
                 sizes=(batch_size_per_iter_per_vec, num_q_heads, half_rope_dim),
                 strides=(1, 1, 1),
             )
        values_tmp = tl.insert_slice(
             values_tmp,
             x,
             offsets=(0, 0, half_rope_dim),
             sizes=(batch_size_per_iter_per_vec, num_q_heads, half_rope_dim),
             strides=(1, 1, 1),
        )   
        x = tl.extract_slice(
                 normalized_values_tmp,
                 offsets=(0, 0, half_rope_dim),
                 sizes=(batch_size_per_iter_per_vec, num_q_heads, half_rope_dim),
                 strides=(1, 1, 1),
             )
        values_tmp = tl.insert_slice(
             values_tmp,
             -x,
             offsets=(0, 0, 0),
             sizes=(batch_size_per_iter_per_vec, num_q_heads, half_rope_dim),
             strides=(1, 1, 1),
        )
        
        values_tmp = values_tmp * sin_tensor
        q_output_idx = output_q_nblk_idx[None, :] + cur_mblk_idx[:, None] * q_size
        mask = (mmask[:, None]) & (output_q_nmask[None, :])
        if IS_PARTIAL_ROPE:
             x = tl.extract_slice(
                 normalized_values_tmp,
                 offsets=(0, 0, 0),
                 sizes=(batch_size_per_iter_per_vec, num_q_heads, rope_dim),
                 strides=(1, 1, 1),
             )
             values_tmp += x * cos_tensor

             normalized_values_tmp = tl.insert_slice(
                 normalized_values_tmp,
                 values_tmp,
                 offsets=(0, 0, 0),
                 sizes=(batch_size_per_iter_per_vec, num_q_heads, rope_dim),
                 strides=(1, 1, 1),
             )
             tl.store(
                 out_q_ptr + q_output_idx,
                 normalized_values_tmp.reshape(batch_size_per_iter_per_vec, q_size),
                 mask=mask,
             )
        else:
             values_tmp += normalized_values_tmp * cos_tensor
             tl.store(
                 out_q_ptr + q_output_idx,
                 values_tmp.reshape(batch_size_per_iter_per_vec, q_size),
                 mask=mask,
             )


        # k
        normalized_values_tmp = tl.extract_slice(
                 normalized_values.reshape(batch_size_per_iter_per_vec, num_qk_heads, head_size),
                 offsets=(0, num_q_heads, 0),
                 sizes=(batch_size_per_iter_per_vec, num_kv_heads, head_size),
                 strides=(1, 1, 1),
             )
        normalized_values_tmp = normalized_values_tmp * k_rmsnorm_weight
        if has_bias:
            normalized_values_tmp = normalized_values_tmp + k_bias

        values_tmp = tl.zeros((batch_size_per_iter_per_vec, num_kv_heads, rope_dim),
                             dtype=tl.float32)
        x = tl.extract_slice(
                normalized_values_tmp,
                 offsets=(0, 0, 0),
                 sizes=(batch_size_per_iter_per_vec, num_kv_heads, half_rope_dim),
                 strides=(1, 1, 1),
        )
        values_tmp = tl.insert_slice(
             values_tmp,
             x,
             offsets=(0, 0, half_rope_dim),
             sizes=(batch_size_per_iter_per_vec, num_kv_heads, half_rope_dim),
             strides=(1, 1, 1),
        )
        x = tl.extract_slice(
                 normalized_values_tmp,
                 offsets=(0, 0, half_rope_dim),
                 sizes=(batch_size_per_iter_per_vec, num_kv_heads , half_rope_dim),
                 strides=(1, 1, 1),
             )
        values_tmp = tl.insert_slice(
             values_tmp,
             -x,
             offsets=(0, 0, 0),
             sizes=(batch_size_per_iter_per_vec, num_kv_heads, half_rope_dim),
             strides=(1, 1, 1),
        )

        values_tmp = values_tmp * sin_tensor
        
        kv_output_idx = output_kv_nblk_idx[None, :] + cur_mblk_idx[:, None] * kv_size
        mask = (mmask[:, None]) & (output_kv_nmask[None, :])
        if IS_PARTIAL_ROPE:
            x1 = tl.extract_slice(
                 normalized_values_tmp,
                 offsets=(0, 0, 0),
                 sizes=(batch_size_per_iter_per_vec, num_kv_heads, rope_dim),
                 strides=(1, 1, 1),
             )
            values_tmp += x1 * cos_tensor

            normalized_values_tmp = tl.insert_slice(
                 normalized_values_tmp,
                 values_tmp,
                 offsets=(0, 0, 0),
                 sizes=(batch_size_per_iter_per_vec, num_kv_heads, rope_dim),
                 strides=(1, 1, 1),
             )
            tl.store(
                 out_k_ptr + kv_output_idx,
                 normalized_values_tmp.to(tl.bfloat16).reshape(batch_size_per_iter_per_vec, kv_size),
                 mask=mask,
             )
        else:
            values_tmp += normalized_values_tmp * cos_tensor
            tl.store(
                 out_k_ptr + kv_output_idx,
                 values_tmp.to(tl.bfloat16).reshape(batch_size_per_iter_per_vec, kv_size),
                 mask=mask,
             )


    mblk_idx = tl.arange(0, v_batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(q_size + kv_size, total_hidden_size)
    nmask = nblk_idx < total_hidden_size
    out_nblk_idx = tl.arange(0, kv_size)
    out_nmask = out_nblk_idx < kv_size

    for _ in tl.range(v_iter_num_per_vec):
        mmask = mblk_idx < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        idx = mblk_idx[:, None] * total_hidden_size + nblk_idx[None, :]  
        values = tl.load(in_qkv_gm_ptr + idx, mask=mask)
        out_mask = (mmask[:, None]) & (out_nmask[None, :])
        out_idx = mblk_idx[:, None] * kv_size + out_nblk_idx[None, :]
        tl.store(out_v_ptr + out_idx, values, mask=out_mask)
        mblk_idx += v_batch_size_per_iter_per_vec


def triton_split_qkv_rmsnorm_mrope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    thw_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    core_num = get_vectorcore_num()

    num_qk_heads = int(num_q_heads + num_kv_heads)
    has_bias = q_bias is not None

    batch_size, total_hidden_size = qkv.shape
    batch_size_per_vec = triton.cdiv(batch_size, core_num)
    
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    qk_size = q_size + kv_size
    if rope_dim is None:
        rope_dim = head_size
    half_rope_dim = rope_dim // 2
    IS_PARTIAL_ROPE = rope_dim != head_size
    
# 设每个iter有X行（bfloat16）：
# values_tmp:       max(x * (num_q_heads + num_kv_heads)*head_size, x * num_q_heads * rope_dim *2)  = x * num_q_heads * head_size * 2
# thw_tensor： x * rope_dim
# cos_tensor、sin_tensor: x * half_rope_dim * 2 * 2 * 2 = x * rope_dim * 4
# normalized_values: x * (num_q_heads + num_kv_heads)*head_size * 2
# normalized_values_tmp: x * num_q_heads * head_size * 2
# x: x * num_q_heads * rope_dim * 2 +  x * num_q_heads * rope_dim

    batch_size_per_iter_per_vec = 85*1024/qkv.element_size()//(max((num_q_heads+num_kv_heads)*head_size, num_q_heads*rope_dim*3) + rope_dim*7 + (
            num_q_heads*head_size*4 + num_kv_heads*head_size*2) + num_q_heads*rope_dim*2)

    batch_size_per_iter_per_vec = min(batch_size_per_iter_per_vec, batch_size_per_vec)
    qk_head_nums_per_iter_per_vec = batch_size_per_iter_per_vec * num_qk_heads
    iter_num_per_vec = triton.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)

    # cos_sin_batch_per_vec = batch_size_per_vec * 2 * half_rope_dim
    cos_sin_batch_per_iter_per_vec = batch_size_per_iter_per_vec * 2
    cos_sin_size_batch_per_iter_per_vec = cos_sin_batch_per_iter_per_vec * half_rope_dim
    t_batch_offset_end = batch_size * 2 * half_rope_dim
    h_batch_offset_end = batch_size * 4 * half_rope_dim
    # w_batch_offset_end = batch_size * 6 * half_rope_dim
    # h_cos_sin_batch_vec_offset_end = (cos_sin_batch_per_vec + batch_size * 2) * half_rope_dim
    # w_cos_sin_batch_vec_offset_end = (cos_sin_batch_per_vec + batch_size * 4) * half_rope_dim
    h_nmask_end = mrope_section[0] + mrope_section[1]
    # w_nmask_end = h_nmask_end + mrope_section[2]
    
    q_output = torch.empty(batch_size,
                           q_size,
                           device=qkv.device,
                           dtype=qkv.dtype)
    k_output = torch.empty(batch_size,
                           kv_size,
                           device=qkv.device,
                           dtype=qkv.dtype)
    v_output = torch.empty(batch_size,
                           kv_size,
                           device=qkv.device,
                           dtype=qkv.dtype)
    
    v_batch_size_per_iter_per_vec = 85 * 1024 / torch.bfloat16.itemsize // (kv_size + 1)
    v_batch_size_per_iter_per_vec = min(v_batch_size_per_iter_per_vec, batch_size_per_vec)
    v_iter_num_per_vec = triton.cdiv(batch_size_per_vec, v_batch_size_per_iter_per_vec)

    grid = (min(core_num,batch_size), 1)

    split_qkv_rmsnorm_mrope_kernel[grid](
        qkv,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        cos_sin.reshape(batch_size * 6, half_rope_dim).contiguous(),
        q_output,
        k_output,
        v_output,
        int(batch_size),
        num_q_heads,
        num_kv_heads,
        num_qk_heads,
        head_size,
        q_size,
        kv_size,
        qk_size,
        total_hidden_size,
        eps,
        has_bias,
        thw_mask,
        rope_dim,
        int(half_rope_dim),
        IS_PARTIAL_ROPE,
        int(batch_size_per_vec),
        int(batch_size_per_iter_per_vec),
        int(cos_sin_batch_per_iter_per_vec),
        int(cos_sin_size_batch_per_iter_per_vec),
        int(iter_num_per_vec),
        int(qk_head_nums_per_iter_per_vec),
        int(v_batch_size_per_iter_per_vec), 
        int(v_iter_num_per_vec),
        int(t_batch_offset_end),
        int(h_batch_offset_end),
    )

    return q_output, k_output, v_output


def triton_split_qkv_rmsnorm_mrope_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    thw_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = qkv.shape[0]
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size

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

    return q_output, k_output, v_output


direct_register_custom_op(
    op_name="triton_split_qkv_rmsnorm_mrope",
    op_func=triton_split_qkv_rmsnorm_mrope,
    fake_impl=triton_split_qkv_rmsnorm_mrope_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1"
)

