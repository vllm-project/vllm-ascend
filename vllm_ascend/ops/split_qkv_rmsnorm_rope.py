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
"""
Non-Triton fallback implementation of the fused split_qkv_rmsnorm_rope op.

When Triton (triton-ascend) is not available (e.g., in A+X environments where
triton and triton-ascend versions conflict), this module provides a native
torch_npu-based implementation so that the `torch.ops.vllm.qkv_rmsnorm_rope`
custom op is still registered and the QKNormRopeFusionPass can function
correctly.

See: https://github.com/vllm-project/vllm-ascend/issues/6737
"""

import torch
import torch_npu  # noqa: F401
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
    """Fused split-QKV + RMSNorm + RoPE using native torch_npu operators.

    This is the non-Triton fallback path. It uses ``npu_rms_norm`` for
    RMSNorm and the registered ``npu_rotary_embedding`` custom op for RoPE,
    keeping the implementation consistent with the rest of the codebase.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE."

    batch_size = input.shape[0]

    # 1. Split input into Q, K, V portions
    q_input = input[:, :q_hidden_size]
    k_input = input[:, q_hidden_size : q_hidden_size + kv_hidden_size]
    v_output = input[:, q_hidden_size + kv_hidden_size :].clone()

    # 2. Reshape to [batch_size * num_heads, head_dim] for RMSNorm
    q_input = q_input.reshape(-1, head_dim)
    k_input = k_input.reshape(-1, head_dim)

    # 3. Apply RMSNorm using fused NPU operator
    q_normalized, _ = torch_npu.npu_rms_norm(q_input, q_weight, eps)
    k_normalized, _ = torch_npu.npu_rms_norm(k_input, k_weight, eps)

    # 4. Apply bias if present
    if q_bias is not None:
        q_normalized = q_normalized + q_bias
    if k_bias is not None:
        k_normalized = k_normalized + k_bias

    # 5. Flatten back to [batch_size, hidden_size] for RoPE
    q_flat = q_normalized.reshape(batch_size, q_hidden_size)
    k_flat = k_normalized.reshape(batch_size, kv_hidden_size)

    # 6. Apply RoPE via the already-registered npu_rotary_embedding custom op
    #    This keeps the implementation consistent and avoids duplicating
    #    RoPE logic. The op handles cos/sin cache indexing internally.
    q_rope, k_rope = torch.ops.vllm.npu_rotary_embedding(
        positions, q_flat, k_flat, cos_sin_cache, head_dim, head_dim, True
    )

    return q_rope, k_rope, v_output


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
    """Fake implementation for shape inference during Dynamo/AOT tracing."""
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
