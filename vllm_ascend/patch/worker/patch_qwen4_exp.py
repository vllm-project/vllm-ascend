#
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
# distributed under the License is distributed on the "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# mypy: ignore-errors

"""NPU enablement scaffold for Qwen4Exp (Qwen3.8-Flash-Next).

Upstream vLLM registers ``qwen4_exp`` in its model registry (#53896). This
patch wires Ascend-specific behavior without duplicating model files.

GDN layers reuse ``QwenGatedDeltaNetAttention``, already patched in
``patch_qwen3_5.py``. QSA/PLE/HyperConnection NPU backends are follow-up work.
"""

from __future__ import annotations

import torch
from torch import nn

QWEN4_EXP_AVAILABLE = False

try:
    import vllm.models.qwen4_exp.nvidia.model as qwen4_exp_model_module
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
except ImportError:
    pass
else:
    QWEN4_EXP_AVAILABLE = True

    def enable_qwen4_exp_low_latency_gemm(module: nn.Module, dtype: torch.dtype) -> None:
        """Skip CUDA-only skinny GEMM hooks on Ascend NPU."""
        del module, dtype

    qwen4_exp_model_module.enable_qwen4_exp_low_latency_gemm = enable_qwen4_exp_low_latency_gemm

    def _ascend_qwen4_exp_qsa_project_qkv_gate(
        self: Qwen4ExpQSAAttention,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply the Ascend text-only QKV/RoPE path used by Qwen3Next attention."""
        if self.attn_output_gate:
            q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.reshape(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(-1, self.num_heads * self.head_dim)
        k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(
            -1, self.num_kv_heads * self.head_dim
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    Qwen4ExpQSAAttention._project_qkv_gate = _ascend_qwen4_exp_qsa_project_qkv_gate
