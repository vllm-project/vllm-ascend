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
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""NPU-friendly method replacements for BailingMoELinearAttention.

This module provides NPU-friendly drop-in replacements for the following
methods of ``BailingMoELinearAttention``:

- ``_prefill_and_mix_infer``: replaces the GPU Triton-based prefill path.
- ``_decode_infer``: replaces the GPU Triton-based decode path.
- ``_forward``: replaces the full forward pass to fix the group-norm branch.

These functions are designed to be monkey-patched onto the class via
``vllm_ascend/patch/worker/patch_bailing_linear_attn.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops.layernorm_guard import layernorm_fn
from vllm.model_executor.layers.mamba.linear_attn import (
    clear_linear_attention_cache_for_new_sequences,
    linear_attention_decode,
    linear_attention_prefill_and_mix,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

from vllm_ascend.ops.triton.mamba.lightning_attn import (
    LightningAttentionKernelNPU,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.bailing_moe_linear import BailingMoELinearAttention


def _prefill_and_mix_infer_npu(
    self: BailingMoELinearAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    state_indices_tensor: torch.Tensor,
    attn_metadata: LinearAttentionMetadata,
) -> torch.Tensor:
    return linear_attention_prefill_and_mix(
        q=q,
        k=k,
        v=v,
        kv_cache=kv_cache,
        state_indices_tensor=state_indices_tensor,
        attn_metadata=attn_metadata,
        slope_rate=self.tp_slope,
        block_size=self.BLOCK,
        decode_fn=self._decode_infer,
        prefix_fn=LightningAttentionKernelNPU.jit_linear_forward_prefix_npu,
        layer_idx=self.layer_id,
    )


def _decode_infer_npu(self, q, k, v, kv_cache, state_indices_tensor, attn_metadata):
    """Handle decode (single token per sequence)."""
    hidden = linear_attention_decode(
        q,
        k,
        v,
        kv_cache,
        self.tp_slope,
        state_indices_tensor,
        q_start=0,
        q_end=attn_metadata.num_decode_tokens,
        slot_start=0,
        slot_end=attn_metadata.num_decodes,
        block_size=32,
    )
    return hidden


def _forward_npu(
    self: BailingMoELinearAttention,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    forward_context = get_forward_context()
    attn_metadata: AttentionMetadata = forward_context.attn_metadata
    if attn_metadata is not None:
        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, LinearAttentionMetadata)
        num_actual_tokens = attn_metadata.num_prefill_tokens + attn_metadata.num_decode_tokens
    else:
        num_actual_tokens = hidden_states.shape[0]

    # QKV projection
    qkv, _ = self.query_key_value(hidden_states[:num_actual_tokens])

    qkv = qkv.to(torch.float32)
    if self.linear_silu:
        qkv = F.silu(qkv)

    # Split q, k, v
    q, k, v = torch.split(
        qkv,
        [self.q_size_per_rank, self.kv_size_per_rank, self.kv_size_per_rank],
        dim=-1,
    )

    # Apply QK norm if needed
    if self.use_qk_norm:
        q = q.reshape(-1, self.tp_heads, self.head_dim)
        k = k.reshape(-1, self.tp_kv_heads, self.head_dim)
        q = layernorm_fn(
            q,
            self.query_layernorm.weight.data,
            bias=None,
            eps=self.rms_norm_eps,
            is_rms_norm=True,
        )
        k = layernorm_fn(
            k,
            self.key_layernorm.weight.data,
            bias=None,
            eps=self.rms_norm_eps,
            is_rms_norm=True,
        )
        q = q.reshape(-1, self.q_size_per_rank)
        k = k.reshape(-1, self.kv_size_per_rank)

    # Apply rotary embeddings
    if self.linear_rope:
        q, k = self.rotary_emb(positions[:num_actual_tokens], q, k)

    # Reshape to [batch, heads, head_dim]
    q = q.view((qkv.shape[0], self.tp_heads, self.head_dim))
    k = k.view((qkv.shape[0], self.tp_kv_heads, self.head_dim))
    v = v.view((qkv.shape[0], self.tp_kv_heads, self.head_dim))

    # Apply scaling if using minimax backend
    if self.linear_scale:
        q = q * self.scaling

    # Get KV cache and state indices
    if attn_metadata is not None:
        kv_cache = self.kv_cache[0]
        state_indices_tensor = attn_metadata.state_indices_tensor
        clear_linear_attention_cache_for_new_sequences(kv_cache, state_indices_tensor, attn_metadata)

    # Compute attention
    decode_only = getattr(attn_metadata, "num_prefills", 0) == 0
    if attn_metadata is None:
        hidden = torch.empty((q.shape[0], q.shape[1] * q.shape[2]), device=q.device, dtype=q.dtype)
    else:
        if not decode_only:
            hidden = self._prefill_and_mix_infer(q, k, v, kv_cache, state_indices_tensor, attn_metadata)
        else:
            hidden = self._decode_infer(q, k, v, kv_cache, state_indices_tensor, attn_metadata)

    # Apply group norm and gate (matching SGLang behavior).
    gate, _ = self.g_proj(hidden_states[:num_actual_tokens])

    hidden = self.g_norm(hidden)
    hidden = F.sigmoid(gate) * hidden

    hidden = hidden.to(hidden_states.dtype)

    # Output projection
    dense_out, _ = self.dense(hidden)
    output[:num_actual_tokens] = dense_out
