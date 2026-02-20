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

Qwen3.5 uses 4 separate input projections (4 GEMM calls per forward).
This patch fuses them into 2 by modifying the weights of existing
ColumnParallelLinear layers:

  in_proj_qkv + in_proj_z → in_proj_qkv (weight extended)  — 1 GEMM
  in_proj_b   + in_proj_a → in_proj_b   (weight extended)  — 1 GEMM

Key design decisions:
  - Weights are fused via register_forward_pre_hook (runs once before
    first forward, before graph capture) — no lazy init in forward.
  - Fused weights stay in existing ColumnParallelLinear layers, so the
    custom op path (default_unquantized_gemm, MatmulAllReduceAddRMSNorm
    fusion) is fully preserved.
  - Forward uses a simple boolean flag (set in __init__) instead of
    hasattr, making it graph mode compatible.
"""

import torch
from einops import rearrange

try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
except ImportError:
    Qwen3_5GatedDeltaNet = None

if Qwen3_5GatedDeltaNet is not None:
    _original_init = Qwen3_5GatedDeltaNet.__init__


def _fusion_pre_hook(module, args):
    """
    Pre-forward hook: fuse projection weights into existing layers.
    Runs once before the first forward call (during eager warmup,
    before graph capture), then removes itself.

    Fusion:
      in_proj_qkv.weight = cat([qkv_w, z_w])  → GEMM output includes z
      in_proj_b.weight   = cat([b_w, a_w])     → GEMM output includes a

    The fused weights stay inside ColumnParallelLinear, so the custom
    op path (default_unquantized_gemm) is preserved.
    """
    if module._projections_fused:
        module._fusion_hook_handle.remove()
        return

    try:
        qkv_w = module.in_proj_qkv.weight
        z_w = module.in_proj_z.weight
        b_w = module.in_proj_b.weight
        a_w = module.in_proj_a.weight
    except AttributeError:
        module._fusion_hook_handle.remove()
        return

    if not all(w.is_floating_point() for w in [qkv_w, z_w, b_w, a_w]):
        module._fusion_hook_handle.remove()
        return

    # Record split dimensions before fusion
    module._qkv_out_dim = qkv_w.shape[0]
    module._b_out_dim = b_w.shape[0]

    # Fuse qkv + z → in_proj_qkv
    fused_qkvz = torch.cat(
        [qkv_w.data, z_w.data], dim=0,
    ).contiguous()
    module.in_proj_qkv.weight = torch.nn.Parameter(
        fused_qkvz, requires_grad=False)

    # Fuse b + a → in_proj_b
    fused_ba = torch.cat(
        [b_w.data, a_w.data], dim=0,
    ).contiguous()
    module.in_proj_b.weight = torch.nn.Parameter(
        fused_ba, requires_grad=False)

    module._projections_fused = True
    module._fusion_hook_handle.remove()


def _patched_init(self, *args, **kwargs):
    """Override __init__ to set fusion flags and register pre-hook."""
    _original_init(self, *args, **kwargs)
    self._projections_fused = False
    self._qkv_out_dim = 0
    self._b_out_dim = 0
    self._fusion_hook_handle = self.register_forward_pre_hook(
        _fusion_pre_hook)


def _fused_forward(
    self,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
):
    """
    Optimized forward: 2 GEMMs instead of 4.

    When fused (non-quantized):
      GEMM 1: in_proj_qkv(x) → split → mixed_qkv, z
      GEMM 2: in_proj_b(x)   → split → b, a
    When not fused (quantized fallback):
      GEMM 1-4: original 4 separate calls
    """
    num_tokens = hidden_states.size(0)

    if self._projections_fused:
        # ==========================================================
        # Fused: 2 GEMMs via ColumnParallelLinear (custom op preserved)
        # ==========================================================

        # GEMM 1: qkvz — in_proj_qkv now contains [qkv, z] weights
        qkvz, _ = self.in_proj_qkv(hidden_states)
        mixed_qkv = qkvz[:, :self._qkv_out_dim].contiguous()
        z = qkvz[:, self._qkv_out_dim:].contiguous()
        z = z.reshape(z.size(0), -1, self.head_v_dim)

        # GEMM 2: ba — in_proj_b now contains [b, a] weights
        ba, _ = self.in_proj_b(hidden_states)
        b = ba[:, :self._b_out_dim].contiguous()
        a = ba[:, self._b_out_dim:].contiguous()
    else:
        # ==========================================================
        # Fallback: 4 GEMMs (quantized models)
        # ==========================================================
        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b, _ = self.in_proj_b(hidden_states)
        a, _ = self.in_proj_a(hidden_states)

    # ==============================================================
    # Core Attention (_forward_core inherited from patch_qwen3_next)
    # ==============================================================
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

    # ==============================================================
    # Output Projection
    # ==============================================================
    z_shape_og = z.shape
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
    output[:num_tokens], _ = self.out_proj(core_attn_out)


if Qwen3_5GatedDeltaNet is not None:
    Qwen3_5GatedDeltaNet.__init__ = _patched_init
    Qwen3_5GatedDeltaNet.forward = _fused_forward
