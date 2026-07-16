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
# from collections.abc import Iterable
# mypy: ignore-errors


import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.utils import is_310p, vllm_version_is

if vllm_version_is("0.21.0"):
    from vllm.model_executor.layers.mamba.gdn_linear_attn import (  # type: ignore[import-not-found]
        GatedDeltaNetAttention as _GDNBaseCls,
    )
else:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as _GDNBaseCls

_GDN_PATCH_TARGET = _GDNBaseCls


class AscendQwen3NextAttention(Qwen3NextAttention):
    def forward(self, positions: torch.Tensor, output: torch.Tensor, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)
        if "qwen3_5" in self.config.model_type:
            cos_sin = self.rotary_emb.cos_sin_cache[positions]
            if cos_sin.device != qkv.device:
                cos_sin = cos_sin.to(qkv.device)
            if cos_sin.dtype != qkv.dtype:
                cos_sin = cos_sin.to(qkv.dtype)

            q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
                qkv=qkv,
                q_weight=1.0 + self.q_norm.weight,
                k_weight=1.0 + self.k_norm.weight,
                cos_sin=cos_sin,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                eps=self.config.rms_norm_eps,
                mrope_section=self.rotary_emb.mrope_section,
                is_interleaved=self.rotary_emb.mrope_interleaved,
                rope_dim=self.rotary_emb.rotary_dim,
                has_gate=self.attn_output_gate,
            )
        else:
            if self.attn_output_gate:
                q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
                orig_shape = q_gate.shape[:-1]
                q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
                q, gate = torch.chunk(q_gate, 2, dim=-1)
                q = q.reshape(*orig_shape, -1)
                gate = gate.reshape(*orig_shape, -1)
            else:
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)

            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)


class AscendQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_idx == 0 and _EXTRA_CTX.flash_comm_v1_enabled:
            tp_size = get_tensor_model_parallel_world_size()
            n_out = (hidden_states.shape[0] + tp_size - 1) // tp_size
            hidden_dim = hidden_states.shape[-1]
            self_attention_output = torch.empty(
                (n_out, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
        else:
            self_attention_output = torch.empty_like(hidden_states)

        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype) + 1)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, {len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype) + 1)

        return hidden_states, residual


Qwen3_5DecoderLayer.forward = AscendQwen3_5DecoderLayer.forward
Qwen3NextAttention.forward = AscendQwen3NextAttention.forward
_GDN_PATCH_TARGET._split_ba_for_tp = AscendGatedDeltaNetAttention._split_ba_for_tp
_GDN_PATCH_TARGET.get_state_shape = AscendGatedDeltaNetAttention.get_state_shape

if is_310p():
    from vllm_ascend._310p.ops.fla.gdn_310 import AscendGatedDeltaNetAttention310

    _GDN_PATCH_TARGET._forward_core = AscendGatedDeltaNetAttention310._forward_core
    _GDN_PATCH_TARGET.get_state_dtype = AscendGatedDeltaNetAttention310.get_state_dtype
else:
    _GDN_PATCH_TARGET.forward = AscendGatedDeltaNetAttention.forward
    _GDN_PATCH_TARGET._forward_core = AscendGatedDeltaNetAttention._forward_core
    _GDN_PATCH_TARGET._warmup_prefill_kernels = AscendGatedDeltaNetAttention._warmup_prefill_kernels


# ----------------------------------------------------------------------------
# EVS (Efficient Video Sampling) multimodal pruning for Qwen3.5.
# Qwen3.5 inherits the full EVS pipeline (pruning in _process_video_input +
# recompute_mrope_positions) from Qwen3VLForConditionalGeneration but ships
# with it explicitly disabled. Re-enable it on Ascend NPU by:
#   1. supports_multimodal_pruning = True
#   2. drop the NotImplementedError override of recompute_mrope_positions so
#      the parent's impl is inherited
#   3. wrap __init__ to honor multimodal_config flags (--video-pruning-rate
#      q>0) and set the EVS post-processing attributes (_tokenizer /
#      use_deepstack / visual_dim / ...) that Qwen3.5's __init__ omits (it
#      calls nn.Module.__init__ instead of super().__init__).
# The runner-side wiring (NPUModelRunner.load_model setting
# is_multimodal_pruning_enabled) lives in worker/model_runner_v1.py.
# ----------------------------------------------------------------------------
from vllm.model_executor.models.qwen3_5 import (  # noqa: E402
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from vllm.tokenizers.registry import cached_tokenizer_from_config  # noqa: E402


def _set_qwen3_5_evs_attrs(self, vllm_config):
    mm_config = vllm_config.model_config.multimodal_config
    config = vllm_config.model_config.hf_config
    self.is_multimodal_pruning_enabled = mm_config.is_multimodal_pruning_enabled() if mm_config is not None else False
    self.video_pruning_rate = mm_config.video_pruning_rate if mm_config is not None else 0.0
    self.model_config = vllm_config.model_config
    self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
    # Qwen3.5 ships an empty deepstack_visual_indexes; use bool(...) so the
    # EVS path does not enter the deepstack branch with num_level == 0.
    vision_config = getattr(config, "vision_config", None)
    _ds = getattr(vision_config, "deepstack_visual_indexes", None) or []
    self.use_deepstack = bool(_ds)
    self.deepstack_num_level = len(_ds)
    self.visual_dim = vision_config.out_hidden_size if vision_config is not None else 0
    self.multiscale_dim = self.visual_dim * self.deepstack_num_level


_orig_qwen3_5_init = Qwen3_5ForConditionalGeneration.__init__


def _qwen3_5_evs_init(self, *, vllm_config, prefix="model"):
    _orig_qwen3_5_init(self, vllm_config=vllm_config, prefix=prefix)
    _set_qwen3_5_evs_attrs(self, vllm_config)


Qwen3_5ForConditionalGeneration.__init__ = _qwen3_5_evs_init
Qwen3_5ForConditionalGeneration.supports_multimodal_pruning = True
if "recompute_mrope_positions" in Qwen3_5ForConditionalGeneration.__dict__:
    del Qwen3_5ForConditionalGeneration.recompute_mrope_positions

# MoE variant: it overrides __init__ but inherits recompute_mrope_positions.
_orig_qwen3_5_moe_init = Qwen3_5MoeForConditionalGeneration.__init__


def _qwen3_5_moe_evs_init(self, *, vllm_config, prefix="model"):
    _orig_qwen3_5_moe_init(self, vllm_config=vllm_config, prefix=prefix)
    _set_qwen3_5_evs_attrs(self, vllm_config)


Qwen3_5MoeForConditionalGeneration.__init__ = _qwen3_5_moe_evs_init
Qwen3_5MoeForConditionalGeneration.supports_multimodal_pruning = True
if "recompute_mrope_positions" in Qwen3_5MoeForConditionalGeneration.__dict__:
    del Qwen3_5MoeForConditionalGeneration.recompute_mrope_positions
