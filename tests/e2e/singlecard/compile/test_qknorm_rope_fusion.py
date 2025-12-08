#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
from typing import List

import pytest
import torch
import torch.nn as nn
import torch_npu
import vllm.config
from vllm.compilation.fx_utils import OpOverload
from vllm.config import ModelConfig, VllmConfig
from tests.e2e.singlecard.compile.backend import TestBackend
from vllm_ascend.compilation.passes.qknorm_rope_fusion_pass import \
    QKNormRopeFusionPass


class TestQKNormRopeModelNoBias(nn.Module):
    """
    A minimal test model that simulates the pattern:
        QKV split → Q RMSNorm → K RMSNorm → Reshape → RoPE (no bias)
    """

    def __init__(self, head_dim: int, num_heads: int, num_kv_heads: int, 
                 eps: float = 1e-6, device: str = "npu"):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.eps = eps
        
        # Calculate sizes
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        # Parameters
        self.q_weight = nn.Parameter(torch.randn(head_dim, device=device))
        self.k_weight = nn.Parameter(torch.randn(head_dim, device=device))
        
        # RoPE parameters
        self.cos_weight = nn.Parameter(torch.randn(1, 1, 1, head_dim, device=device))
        self.sin_weight = nn.Parameter(torch.randn(1, 1, 1, head_dim, device=device))
        
        self.seq_len = None  # To be set during forward pass

    def forward(self, qkv):
        """
        Forward pass simulating the unfused pattern (no bias)
        """
        seq_len = qkv.shape[0]
        cos = self.cos_weight.expand(1, seq_len, 1, self.head_dim)
        sin = self.sin_weight.expand(1, seq_len, 1, self.head_dim)
        
        # Split QKV
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_norm_out, _ = torch_npu.npu_rms_norm(q_by_head, self.q_weight, self.eps)
        q_flat = q_norm_out.view(q.shape)
        q_reshape = q_flat.contiguous().view(1, q_flat.shape[0], -1, self.head_dim)
        
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_norm_out, _ = torch_npu.npu_rms_norm(k_by_head, self.k_weight, self.eps)
        k_flat = k_norm_out.view(k.shape)
        k_reshape = k_flat.contiguous().view(1, k_flat.shape[0], -1, self.head_dim)
        
        # Apply RoPE
        q_rope, k_rope = torch_npu.npu_apply_rotary_pos_emb(
            q_reshape, k_reshape, cos, sin
        )
        
        return q_rope, k_rope, v

    def ops_in_model_before(self) -> List[OpOverload]:
        """Return the list of expected operators BEFORE fusion."""
        return [
            torch.ops.npu.npu_apply_rotary_pos_emb.default
        ]

    def ops_in_model_after(self) -> List[OpOverload]:
        """Return the list of expected operators AFTER successful fusion."""
        return [torch.ops.vllm.qkv_rmsnorm_rope.default]


class TestQKNormRopeModelWithBias(nn.Module):
    """
    A minimal test model that simulates the pattern:
        QKV split → Q RMSNorm → Q Bias → K RMSNorm → K Bias → Reshape → RoPE (with bias)
    """

    def __init__(self, head_dim: int, num_heads: int, num_kv_heads: int, 
                 eps: float = 1e-6, device: str = "npu"):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.eps = eps
        
        # Calculate sizes
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        # Parameters
        self.q_weight = nn.Parameter(torch.randn(head_dim, device=device))
        self.k_weight = nn.Parameter(torch.randn(head_dim, device=device))
        self.q_bias = nn.Parameter(torch.randn(head_dim, device=device))
        self.k_bias = nn.Parameter(torch.randn(head_dim, device=device))
        
        # RoPE parameters
        self.cos_weight = nn.Parameter(torch.randn(1, 1, 1, head_dim, device=device))
        self.sin_weight = nn.Parameter(torch.randn(1, 1, 1, head_dim, device=device))
        
    def forward(self, qkv):
        """
        Forward pass simulating the unfused pattern (with bias)
        """
        seq_len = qkv.shape[0]
        cos = self.cos_weight.expand(1, seq_len, 1, self.head_dim)
        sin = self.sin_weight.expand(1, seq_len, 1, self.head_dim)
        
        # Split QKV
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_norm_out, _ = torch_npu.npu_rms_norm(q_by_head, self.q_weight, self.eps)
        q_normed = q_norm_out + self.q_bias
        q_flat = q_normed.view(q.shape)
        q_reshape = q_flat.contiguous().view(1, q_flat.shape[0], -1, self.head_dim)
        
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_norm_out, _ = torch_npu.npu_rms_norm(k_by_head, self.k_weight, self.eps)
        k_normed = k_norm_out + self.k_bias
        k_flat = k_normed.view(k.shape)
        k_reshape = k_flat.contiguous().view(1, k_flat.shape[0], -1, self.head_dim)
        
        # Apply RoPE
        q_rope, k_rope = torch_npu.npu_apply_rotary_pos_emb(
            q_reshape, k_reshape, cos, sin
        )
        
        return q_rope, k_rope, v

    def ops_in_model_before(self) -> List[OpOverload]:
        """Return the list of expected operators BEFORE fusion."""
        return [
            torch.ops.npu.npu_apply_rotary_pos_emb.default,
            torch.ops.aten.add.Tensor
        ]

    def ops_in_model_after(self) -> List[OpOverload]:
        """Return the list of expected operators AFTER successful fusion."""
        return [torch.ops.vllm.qkv_rmsnorm_rope.default]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [1, 128])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("with_bias", [False])
def test_qknorm_rope_fusion(
    dtype: torch.dtype,
    seq_len: int,
    eps: float,
    with_bias: bool,
):
    """
    End-to-end test for QKV split + RMSNorm + RoPE fusion.
    Tests both with and without bias versions.
    """
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)
    
    # Model parameters
    head_dim = 128
    num_heads = 32
    num_kv_heads = 8
    
    vllm_config = VllmConfig(model_config=ModelConfig(dtype=dtype))

    with vllm.config.set_current_vllm_config(vllm_config):
        # Create backend with the fusion pass
        backend = TestBackend(custom_passes=[QKNormRopeFusionPass(vllm_config)])
        
        # Create appropriate test model based on bias flag
        if with_bias:
            model = TestQKNormRopeModelWithBias(
                head_dim=head_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                eps=eps,
                device="npu"
            )
        else:
            model = TestQKNormRopeModelNoBias(
                head_dim=head_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                eps=eps,
                device="npu"
            )
        
        model = model.to("npu")
        
        # Create test input
        qkv_size = num_heads * head_dim + 2 * num_kv_heads * head_dim
        x = torch.rand(
            seq_len,
            qkv_size,
            device="npu",
            dtype=dtype,
            requires_grad=False
        )
        
        # Run unfused model
        result_unfused = model(x)
        print(f"Unfused result shapes: {[t.shape for t in result_unfused]}")
        
        # Compile with fusion
        with torch.no_grad():
            model_fused = torch.compile(model, backend=backend)
            result_fused = model_fused(x)
        print(f"Fused result shapes: {[t.shape for t in result_fused]}")
        
        print("=== Checking operator fusion ===")
        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())