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
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

from vllm_ascend.ops.triton.qk_rmsnorm import qk_rmsnorm_triton


def forward(
    self,
    positions: torch.Tensor,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
):
    qkv, _ = self.qkv_proj(hidden_states)

    q, k, v, gate = qk_rmsnorm_triton(
        input=qkv,
        q_weight=self.q_norm.weight,
        k_weight=self.k_norm.weight,
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        eps=self.q_norm.variance_epsilon,
        has_gate=self.attn_output_gate,
    )

    q, k = self.rotary_emb(positions, q, k)

    attn_output = self.attn(q, k, v)

    if self.attn_output_gate:
        gate = torch.sigmoid(gate)
        attn_output = attn_output * gate

    output[:], _ = self.o_proj(attn_output)


Qwen3NextAttention.forward = forward