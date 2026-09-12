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
from vllm.model_executor.models.gemma4 import Gemma4Attention

from vllm_ascend.device.device_config import is_950
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope_vnorm import qkv_rmsnorm_rope_vnorm_fits_ub

_original_attention_init = Gemma4Attention.__init__
_original_attention_forward = Gemma4Attention.forward

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
    if is_950():
        return "the fused kernel has no A5 variant"
    if self.is_kv_shared_layer:
        return "the layer shares the KV cache of an earlier layer"
    # The kernel applies no norm bias, which quantized checkpoints may carry
    # (see AscendRMSNorm). `bias` rather than `bias_loaded` because this runs
    # before the weights are loaded.
    if any(getattr(norm, "bias", None) is not None for norm in (self.q_norm, self.k_norm, self.v_norm)):
        return "a q/k/v norm carries a bias"
    if self.q_norm.weight.dtype != FUSED_PREATTENTION_DTYPE:
        return f"norm weights are {self.q_norm.weight.dtype}, not {FUSED_PREATTENTION_DTYPE}"
    cos_sin_cache = self.rotary_emb.cos_sin_cache
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
    if not self.use_fused_preattention:
        return _original_attention_forward(self, positions, hidden_states, **kwargs)

    qkv, _ = self.qkv_proj(hidden_states)
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
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)

    return output


Gemma4Attention.__init__ = _patched_attention_init
Gemma4Attention.forward = _patched_attention_forward
