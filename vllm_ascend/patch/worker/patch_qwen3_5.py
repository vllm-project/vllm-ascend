#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""
Performance patch for Qwen3.5 GatedDeltaNet on Ascend NPU.

Qwen3.5 uses 4 separate input projections:
  in_proj_qkv  (hidden → key_dim*2 + value_dim)  — MergedColumnParallelLinear
  in_proj_z    (hidden → value_dim)               — ColumnParallelLinear
  in_proj_b    (hidden → num_v_heads)             — ColumnParallelLinear
  in_proj_a    (hidden → num_v_heads)             — ColumnParallelLinear

This results in 4 GEMM calls per forward, each reading hidden_states from HBM.
The b/a projections are especially wasteful: num_v_heads is tiny (e.g., 64)
compared to hidden_size (e.g., 4096), giving extremely low compute utilization.

This patch fuses them into 2 projections:
  fused_qkvz   (hidden → key_dim*2 + value_dim*2)  — 1 GEMM
  fused_ba     (hidden → num_v_heads*2)             — 1 GEMM

Benefits:
  - Halves the number of hidden_states HBM reads
  - Eliminates 2 tiny, inefficient GEMM kernels
  - Reduces kernel launch overhead
"""

import torch
import torch.nn.functional as F
from einops import rearrange

try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
except ImportError:
    Qwen3_5GatedDeltaNet = None

# Save original forward for fallback (quantized / LoRA models)
if Qwen3_5GatedDeltaNet is not None:
    _original_qwen3_5_forward = Qwen3_5GatedDeltaNet.forward


class AscendQwen3_5GatedDeltaNet:
    """Optimized Qwen3.5 GatedDeltaNet forward for Ascend NPU."""

    def _fuse_projections(self):
        """
        Fuse 4 separate projection weights into 2 fused weights.
        Called lazily on first forward. Returns True if fusion succeeded.

        Fusion layout:
          _fused_qkvz_weight = cat([in_proj_qkv.weight, in_proj_z.weight])
          _fused_ba_weight   = cat([in_proj_b.weight,   in_proj_a.weight])

        After GEMM, output is split by recorded dimensions:
          qkvz_out → mixed_qkv[:qkv_dim] + z[qkv_dim:]
          ba_out   → b[:b_dim]            + a[b_dim:]
        """
        if hasattr(self, '_projections_fused'):
            return self._projections_fused

        # Verify all projections have accessible weight tensors
        modules = [self.in_proj_qkv, self.in_proj_z,
                   self.in_proj_b, self.in_proj_a]
        if not all(hasattr(m, 'weight') and m.weight is not None
                   for m in modules):
            self._projections_fused = False
            return False

        qkv_w = self.in_proj_qkv.weight
        z_w = self.in_proj_z.weight
        b_w = self.in_proj_b.weight
        a_w = self.in_proj_a.weight

        # Only fuse plain float weights (not quantized int4/int8)
        if not all(w.is_floating_point() for w in [qkv_w, z_w, b_w, a_w]):
            self._projections_fused = False
            return False

        # Record split dimensions (after TP sharding)
        self._qkv_out_dim = qkv_w.shape[0]
        self._z_out_dim = z_w.shape[0]
        self._b_out_dim = b_w.shape[0]
        self._a_out_dim = a_w.shape[0]

        # Fuse qkv + z weights
        self._fused_qkvz_weight = torch.cat(
            [qkv_w.data, z_w.data], dim=0,
        ).contiguous()

        # Fuse b + a weights
        self._fused_ba_weight = torch.cat(
            [b_w.data, a_w.data], dim=0,
        ).contiguous()

        self._projections_fused = True
        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Optimized forward: 2 GEMMs instead of 4.

        Part 1: Fused projection
          GEMM 1 (qkvz): hidden_states @ fused_qkvz_weight^T → split → mixed_qkv, z
          GEMM 2 (ba):   hidden_states @ fused_ba_weight^T   → split → b, a
        Part 2: Core attention (_forward_core inherited from patch_qwen3_next.py)
        Part 3: RMSNormGated + output projection
        """
        # Lazy fusion on first call; fallback if fusion not possible
        if not hasattr(self, '_projections_fused'):
            if not self._fuse_projections():
                return _original_qwen3_5_forward(self, hidden_states, output)
        if not self._projections_fused:
            return _original_qwen3_5_forward(self, hidden_states, output)

        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Fused Input Projections (2 GEMMs instead of 4)
        # ============================================================

        # GEMM 1: qkvz — fuses in_proj_qkv + in_proj_z
        qkvz_out = F.linear(hidden_states, self._fused_qkvz_weight)
        mixed_qkv = qkvz_out[:, :self._qkv_out_dim].contiguous()
        z = qkvz_out[:, self._qkv_out_dim:].contiguous()
        z = z.reshape(z.size(0), -1, self.head_v_dim)

        # GEMM 2: ba — fuses in_proj_b + in_proj_a
        ba_out = F.linear(hidden_states, self._fused_ba_weight)
        b = ba_out[:, :self._b_out_dim].contiguous()
        a = ba_out[:, self._b_out_dim:].contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)


if Qwen3_5GatedDeltaNet is not None:
    Qwen3_5GatedDeltaNet._fuse_projections = (
        AscendQwen3_5GatedDeltaNet._fuse_projections
    )
    Qwen3_5GatedDeltaNet.forward = AscendQwen3_5GatedDeltaNet.forward
