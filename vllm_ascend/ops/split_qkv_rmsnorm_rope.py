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
import torch_npu
from vllm.utils.torch_utils import direct_register_custom_op


def split_qkv_rmsnorm_rope_impl(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = input.shape[0]
    num_q_heads = q_hidden_size // head_dim
    num_kv_heads = kv_hidden_size // head_dim

    # 1. Split input into Q, K, V portions
    q_input = input[:, :q_hidden_size]
    k_input = input[:, q_hidden_size : q_hidden_size + kv_hidden_size]
    v_output = input[:, q_hidden_size + kv_hidden_size :].clone()

    # 2. Reshape to [batch_size * num_heads, head_dim] for RMSNorm
    q_input = q_input.reshape(-1, head_dim)
    k_input = k_input.reshape(-1, head_dim)

    # 3. Apply RMSNorm using fused NPU operator
    # npu_rms_norm returns (normalized_output, rstd_for_backward)
    q_normalized, _ = torch_npu.npu_rms_norm(q_input, q_weight, eps)
    k_normalized, _ = torch_npu.npu_rms_norm(k_input, k_weight, eps)

    # 4. Apply bias if present
    if q_bias is not None:
        q_normalized = q_normalized + q_bias
    if k_bias is not None:
        k_normalized = k_normalized + k_bias

    # 5. Reshape to [batch_size, num_heads, head_dim] for RoPE
    q_normalized = q_normalized.view(batch_size, num_q_heads, head_dim)
    k_normalized = k_normalized.view(batch_size, num_kv_heads, head_dim)

    # 6. Prepare sin/cos for npu_apply_rotary_pos_emb
    # The NPU op expects all inputs as 4D tensors
    # Q/K: [1, batch_size, num_heads, head_dim]
    # cos/sin: will be prepared from cos_sin_cache using positions
    q_normalized = q_normalized.unsqueeze(0)  # [1, batch_size, num_q_heads, head_dim]
    k_normalized = k_normalized.unsqueeze(0)  # [1, batch_size, num_kv_heads, head_dim]

    # positions: [batch_size]
    # cos_sin_cache is expected to be indexed by positions to yield
    # a vector per position where first half is cos and second half is sin
    pos_vals = cos_sin_cache.index_select(0, positions.to(dtype=torch.long))
    half = head_dim // 2
    cos_half = pos_vals[:, :half]
    sin_half = pos_vals[:, half : half + half]

    # Expand halves to full head_dim by repeating each half
    # and reshape to [1, batch_size, 1, head_dim]
    cos_full = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(0).unsqueeze(2)
    sin_full = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(0).unsqueeze(2)

    # 7. Apply RoPE using fused NPU operator
    # npu_apply_rotary_pos_emb modifies tensors in-place but returns them
    q_output, k_output = torch_npu.npu_apply_rotary_pos_emb(
        q_normalized,
        k_normalized,
        cos_full,
        sin_full,
    )

    # 8. Reshape back to [batch_size, hidden_size]
    q_output = q_output.squeeze(0).reshape(batch_size, q_hidden_size)
    k_output = k_output.squeeze(0).reshape(batch_size, kv_hidden_size)

    return q_output, k_output, v_output


def split_qkv_rmsnorm_rope_impl_fake(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = input.shape[0]
    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    return q_output, k_output, v_output


direct_register_custom_op(
    op_name="qkv_rmsnorm_rope",
    op_func=split_qkv_rmsnorm_rope_impl,
    fake_impl=split_qkv_rmsnorm_rope_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
