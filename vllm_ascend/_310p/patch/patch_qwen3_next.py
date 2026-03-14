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

import torch
from einops import rearrange
from torch import nn
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet

# NOTE: 310P does not support Triton, so we use PyTorch fallback
# for all operations. We do NOT import any triton-related modules here.


class Ascend310Qwen3Next_GatedDeltaNet(nn.Module, MambaBase):
    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (Python fallback core)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)
        
        # For 310P, always use PyTorch fallback implementation
        # since Triton is not supported
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query, key, value = map(lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value))
        mixed_qkv = torch.cat((query, key, value), dim=-1)

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # 310P path: directly invoke Python core to avoid dynamic custom-op
        # registration dependency.
        self._forward_core(mixed_qkv, b, a, core_attn_out)

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation entry for 310P fallback.
        """
        # Import implementation
        from vllm_ascend._310p.ops.gdn_attention import gdn_attention_core_impl
        
        # Get prefix from self
        prefix = self.prefix
        
        # Call implementation
        gdn_attention_core_impl(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            prefix=prefix,
        )


# Patch Qwen3NextGatedDeltaNet class
Qwen3NextGatedDeltaNet.forward = Ascend310Qwen3Next_GatedDeltaNet.forward
Qwen3NextGatedDeltaNet._forward_core = Ascend310Qwen3Next_GatedDeltaNet._forward_core
