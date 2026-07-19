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

from __future__ import annotations

from inspect import signature

import torch

from vllm_ascend import envs

try:
    from vllm.model_executor.models.gemma4 import Gemma4Attention
    from vllm.model_executor.models.utils import extract_layer_index
except ImportError:
    Gemma4Attention = None
    extract_layer_index = None


def _get_prefix(init_fn, args: tuple[object, ...], kwargs: dict[str, object]) -> str:
    try:
        bound = signature(init_fn).bind_partial(None, *args, **kwargs)
        prefix = bound.arguments.get("prefix", "")
        return "" if prefix is None else str(prefix)
    except Exception:
        pass
    return ""


def _get_layer_index(prefix: str) -> int:
    if extract_layer_index is None:
        return 0
    try:
        return int(extract_layer_index(prefix))
    except Exception:
        return 0


if Gemma4Attention is not None and not hasattr(Gemma4Attention, "_ascend_dgemma_original_forward"):
    Gemma4Attention._ascend_dgemma_original_init = Gemma4Attention.__init__
    Gemma4Attention._ascend_dgemma_original_forward = Gemma4Attention.forward

    def _init(self: Gemma4Attention, *args, **kwargs) -> None:
        original_init = self.__class__._ascend_dgemma_original_init
        prefix = _get_prefix(original_init, args, kwargs)
        original_init(self, *args, **kwargs)
        # Rotate the FFTS softsync flag pair across layers. Reusing the same
        # pair too soon can alias AIC/AIV cross-core sync state in graph replay.
        layer_idx = _get_layer_index(prefix)
        self._dgemma_qkv_sync_base = 5 + 2 * (layer_idx % 5)

    def _forward(
        self: Gemma4Attention,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        original_forward = self.__class__._ascend_dgemma_original_forward
        if (
            not envs.VLLM_ASCEND_DGEMMA_FUSE_QKVPROJ_ASCENDC
            or self.is_kv_shared_layer
            or getattr(self.qkv_proj, "bias", None) is not None
        ):
            return original_forward(self, positions, hidden_states, **kwargs)

        cos_sin_cache = self.rotary_emb.cos_sin_cache.to(hidden_states.device)
        cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
        seq_len = hidden_states.shape[0]
        qkv_out = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim

        # Persistent GM scratch keeps the qkv intermediate address stable
        # across ACL graph capture/replay.
        scratch = getattr(self, "_dgemma_qkv_scratch", None)
        if scratch is None or scratch.shape[0] < seq_len:
            scratch = torch.empty((max(seq_len, 256), qkv_out), dtype=hidden_states.dtype, device=hidden_states.device)
            self._dgemma_qkv_scratch = scratch
        qkv_scratch = scratch[:seq_len]

        sync_base = getattr(self, "_dgemma_qkv_sync_base", 5) | 0x100
        _, _, _, mix_scratch = torch.ops._C_ascend.npu_dgemma_fused_qkv_proj_norm_rope(
            hidden_states.contiguous(),
            self.qkv_proj.weight,
            self.q_norm.weight,
            self.k_norm.weight,
            cos.float().contiguous(),
            sin.float().contiguous(),
            qkv_scratch,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            hidden_states.shape[-1],
            sync_base,
            self.q_norm.variance_epsilon,
        )

        guard = mix_scratch[:1, :1].reshape(1, 1, 1).to(hidden_states.dtype) * 0
        q_raw, k_raw, v_raw = mix_scratch.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        qh, kh, vh = torch.ops._C_ascend.npu_dgemma_fused_norm_rope(
            (q_raw.unflatten(-1, (self.num_heads, self.head_dim)) + guard).contiguous(),
            (k_raw.unflatten(-1, (self.num_kv_heads, self.head_dim)) + guard).contiguous(),
            (v_raw.unflatten(-1, (self.num_kv_heads, self.head_dim)) + guard).contiguous(),
            self.q_norm.weight,
            self.k_norm.weight,
            cos.float().contiguous(),
            sin.float().contiguous(),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.q_norm.variance_epsilon,
        )
        q = qh.flatten(-2, -1)
        k = kh.flatten(-2, -1)
        v = vh.flatten(-2, -1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    Gemma4Attention.__init__ = _init
    Gemma4Attention.forward = _forward
