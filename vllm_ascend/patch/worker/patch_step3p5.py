# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# mypy: ignore-errors

import torch
from vllm.logger import logger

try:
    from vllm.model_executor.models.step3p5 import Step3p5Attention
except ImportError:
    logger.debug("Step3p5Attention not available, skipping QK RMSNorm+RoPE fusion patch")
    Step3p5Attention = None


if Step3p5Attention is not None:

    def _patched_forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        if self.use_rope:
            # NPU: fuse QK GemmaRMSNorm + RoPE into a single kernel.
            q_w = 1.0 + self.q_norm.weight.float()
            k_w = 1.0 + self.k_norm.weight.float()
            q, k, v = torch.ops.vllm.qkv_rmsnorm_rope(
                input=qkv,
                q_weight=q_w,
                k_weight=k_w,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                eps=self.q_norm.variance_epsilon,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                positions=positions,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            q_by_head = self.q_norm(q_by_head.contiguous())
            q = q_by_head.view(q.shape)
            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            k_by_head = self.k_norm(k_by_head.contiguous())
            k = k_by_head.view(k.shape)

        attn_output = self.attn(q, k, v)
        if self.use_head_wise_attn_gate:
            extra_dims, _ = self.g_proj(hidden_states)
            output = (
                attn_output.view(
                    *attn_output.shape[:-1],
                    self.num_heads,
                    self.head_dim,
                )
                * extra_dims.unsqueeze(-1).sigmoid()
            )
            attn_output = output.view(*attn_output.shape)
        output, _ = self.o_proj(attn_output)
        return output

    Step3p5Attention.forward = _patched_forward
