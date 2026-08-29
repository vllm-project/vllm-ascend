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

import torch
from vllm.logger import logger

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope_vnorm import qkv_rmsnorm_rope_vnorm_fits_ub
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

try:
    from vllm.model_executor.models.gemma4 import Gemma4Attention

    _original_attention_init = Gemma4Attention.__init__
except ImportError:
    Gemma4Attention = None
    _original_attention_init = None

# The fused kernel keeps its intermediates in bfloat16, which is the dtype
# Gemma4 checkpoints are served with.
FUSED_PREATTENTION_DTYPE = torch.bfloat16


def _unfused_preattention_reason(self) -> str | None:
    """Why this layer keeps the unfused pre-attention chain, None if it fuses.

    Both Gemma4 attention types are eligible. Sliding layers rotate a full
    256-dim head. Full attention layers use proportional RoPE, which
    Gemma4RotaryEmbedding implements by zero-padding inv_freq and passing
    `rotary_dim=head_size` to the base class, so their cos/sin cache is an
    ordinary full-width cache whose non-rotated pairs hold cos=1 and sin=0.
    Both therefore reduce to the same neox rotation over the whole head, and
    the kernel needs no knowledge of the layer type - only shapes that fit one
    vector core.
    """
    if get_ascend_device_type() == AscendDeviceType.A5:
        return "the fused kernel has no A5 variant"
    # KV-shared layers reuse the K/V of an earlier layer and only apply RoPE to
    # Q, so K/V must not be recomputed here.
    if self.is_kv_shared_layer:
        return "the layer shares the KV cache of an earlier layer"
    # The kernel normalizes V without a learnable scale, applies a single eps to
    # Q, K and V, and applies no norm bias (which quantized checkpoints may
    # carry, see AscendRMSNorm).
    if self.v_norm.has_weight:
        return "v_norm carries a learnable scale"
    if any(getattr(norm, "bias", None) is not None for norm in (self.q_norm, self.k_norm, self.v_norm)):
        return "a q/k/v norm carries a bias"
    eps = self.q_norm.variance_epsilon
    if eps != self.k_norm.variance_epsilon or eps != self.v_norm.variance_epsilon:
        return "q, k and v norm use different epsilons"
    if self.q_norm.weight.dtype != FUSED_PREATTENTION_DTYPE:
        return f"norm weights are {self.q_norm.weight.dtype}, not {FUSED_PREATTENTION_DTYPE}"
    cos_sin_cache = getattr(self.rotary_emb, "cos_sin_cache", None)
    if cos_sin_cache is None or not getattr(self.rotary_emb, "is_neox_style", False):
        return f"{type(self.rotary_emb).__name__} exposes no neox-style cos/sin cache"
    if cos_sin_cache.dtype != FUSED_PREATTENTION_DTYPE:
        return f"the cos/sin cache is {cos_sin_cache.dtype}, not {FUSED_PREATTENTION_DTYPE}"
    if not qkv_rmsnorm_rope_vnorm_fits_ub(
        q_hidden_size=self.q_size,
        kv_hidden_size=self.kv_size,
        head_dim=self.head_dim,
        rope_dim=cos_sin_cache.shape[-1],
    ):
        return "one token's tiles exceed the vector core unified buffer, which a larger tensor parallel size shrinks"
    return None


def _configure_fused_preattention(self) -> None:
    """Resolve the fused pre-attention decision of this layer, reporting fallbacks.

    The decision only depends on the layer's configuration. Model construction
    never runs inside a compiled region, so resolving it here keeps the branch
    out of the traced graph and keeps the outcome observable in every
    compilation mode - unlike a check inside `forward`, which torch.compile
    evaluates once at trace time and an ACL graph then replays without Python.

    Only the fallback is logged. Fusing is the expected case, so silence means
    the fused path is in use.
    """
    reason = _unfused_preattention_reason(self)
    self.use_fused_preattention = reason is None
    if reason is not None:
        logger.info_once(
            "Gemma4 %s attention: pre-attention stays unfused because %s (q_size=%d, kv_size=%d, head_dim=%d).",
            "sliding" if self.is_sliding else "full",
            reason,
            self.q_size,
            self.kv_size,
            self.head_dim,
            scope="global",
        )


def _patched_attention_init(self, *args, **kwargs) -> None:
    _original_attention_init(self, *args, **kwargs)
    _configure_fused_preattention(self)


def _patched_attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)

    # The kernel indexes the fused qkv tensor as [num_tokens, hidden].
    if self.use_fused_preattention:
        q, k, v = DeviceOperator.split_qkv_rmsnorm_rope_vnorm(
            input=qkv,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_hidden_size=self.q_size,
            kv_hidden_size=self.kv_size,
            head_dim=self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_bias=None,
            k_bias=None,
            cos_sin_cache=self.rotary_emb.cos_sin_cache,
            positions=positions,
        )
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        if not self.is_kv_shared_layer:
            k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
            k = self.k_norm(k)
            k = k.flatten(-2, -1)
            q, k = self.rotary_emb(positions, q, k)

            v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_norm(v)
            v = v.flatten(-2, -1)
        else:
            q = self.rotary_emb(positions, q, k)[0]

    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)

    return output


if Gemma4Attention is not None:
    Gemma4Attention.__init__ = _patched_attention_init
    Gemma4Attention.forward = _patched_attention_forward
