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
#from collections.abc import Iterable

from typing import Optional

import torch
from torch import nn
from einops import rearrange
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet
from vllm_ascend.ops.triton.fla.fused_qkvzba_split_reshape import fused_qkvzba_split_reshape_cat
from vllm.triton_utils import tl, triton
from . import is_cuda_graph

class AscendQwen3Next_GatedDeltaNet(nn.Module, MambaBase):
    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)
        # triton grid should be less than 66536
        divide_grid=projected_states_qkvz.shape[0]*triton.cdiv(self.num_k_heads, self.tp_size)
        if self.num_v_heads // self.num_k_heads in [1, 2, 4] and is_cuda_graph and divide_grid < 65536:
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat(
                projected_states_qkvz,
                projected_states_ba,
                triton.cdiv(self.num_k_heads, self.tp_size),
                triton.cdiv(self.num_v_heads, self.tp_size),
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                projected_states_qkvz, projected_states_ba)
            query, key, value = map(lambda x: rearrange(x, 'l p d -> l (p d)'),
                                    (query, key, value))
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
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)


Qwen3NextGatedDeltaNet.forward = AscendQwen3Next_GatedDeltaNet.forward