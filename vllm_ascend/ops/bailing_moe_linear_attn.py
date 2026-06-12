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

"""NPU-friendly OOT replacement for BailingMoELinearAttention.

This module provides ``AscendBailingMoELinearAttention``, an out-of-tree (OOT)
replacement for the upstream ``BailingMoELinearAttention`` class.  It is
registered via the ``PluggableLayer`` mechanism so that the upstream class is
transparently replaced at instantiation time when running on Ascend NPU.
"""

import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops.layernorm_guard import layernorm_fn
from vllm.model_executor.layers.mamba.linear_attn import linear_attention_decode
from vllm.model_executor.models.bailing_moe_linear import BailingMoELinearAttention
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

from vllm_ascend.ops.triton.mamba.lightning_attn import (
    AscendLightningAttentionKernel,
    clear_linear_attention_cache_for_prefill_npu,
    pack_qkv_for_prefill,
)


def ascend_linear_attention_prefill_and_mix(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    state_indices_tensor: torch.Tensor,
    attn_metadata: LinearAttentionMetadata,
    slope_rate: torch.Tensor,
    block_size: int,
    decode_fn,
    prefix_fn,
    layer_idx: int | None = None,
) -> torch.Tensor:
    hidden = []
    query_start_loc = getattr(attn_metadata, "query_start_loc_cpu", None)
    if query_start_loc is None:
        query_start_loc = attn_metadata.query_start_loc
    state_indices = getattr(attn_metadata, "state_indices_cpu", None)
    if state_indices is None:
        state_indices = state_indices_tensor
    offset = attn_metadata.num_decode_tokens
    for prefill_idx in range(getattr(attn_metadata, "num_prefills", 0)):
        if offset + prefill_idx + 1 >= len(query_start_loc):
            break
        if offset + prefill_idx >= len(state_indices):
            break
        start = int(query_start_loc[offset + prefill_idx])
        end = int(query_start_loc[offset + prefill_idx + 1])
        slot_id = state_indices[offset + prefill_idx]
        if getattr(slot_id, "device", None) is not None and slot_id.device.type == "cpu":
            slot_id = int(slot_id)

        qs, ks, vs = pack_qkv_for_prefill(q, k, v, start, end)
        slice_layer_cache = kv_cache[slot_id, ...]
        out_slice = prefix_fn(
            qs,
            ks,
            vs,
            slice_layer_cache,
            slope_rate,
            block_size,
            layer_idx=layer_idx,
        )
        hidden.append(out_slice)

    if attn_metadata.num_decode_tokens > 0:
        hidden_decode = decode_fn(q, k, v, kv_cache, state_indices_tensor, attn_metadata)
        hidden.insert(0, hidden_decode)

    if not hidden:
        return torch.empty((0, q.shape[1] * q.shape[2]), device=q.device, dtype=q.dtype)

    if len(hidden) == 1:
        return hidden[0]
    return torch.concat(hidden, dim=0).contiguous()


class AscendBailingMoELinearAttention(BailingMoELinearAttention):
    """NPU-friendly drop-in replacement for BailingMoELinearAttention.

    Registered as an OOT PluggableLayer so that the upstream class is
    transparently replaced when running on Ascend NPU.  Only the three
    platform-specific methods are overridden; everything else (``__init__``,
    ``forward``, weight loading, state shape, etc.) is inherited from the
    upstream implementation.
    """

    def _prefill_and_mix_infer(self, q, k, v, kv_cache, state_indices_tensor, attn_metadata):
        return ascend_linear_attention_prefill_and_mix(
            q=q,
            k=k,
            v=v,
            kv_cache=kv_cache,
            state_indices_tensor=state_indices_tensor,
            attn_metadata=attn_metadata,
            slope_rate=self.tp_slope,
            block_size=self.BLOCK,
            decode_fn=self._decode_infer,
            prefix_fn=AscendLightningAttentionKernel.jit_linear_forward_prefix,
            layer_idx=self.layer_id,
        )

    def _decode_infer(self, q, k, v, kv_cache, state_indices_tensor, attn_metadata):
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

    def _forward(self, hidden_states, output, positions):
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
            if getattr(attn_metadata, "num_prefills", 0) > 0:
                clear_linear_attention_cache_for_prefill_npu(kv_cache, state_indices_tensor, attn_metadata)

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
