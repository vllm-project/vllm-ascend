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


from functools import wraps
from itertools import islice

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as _GDNBaseCls
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5Model

try:
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor
    from vllm.sequence import IntermediateTensors
except ImportError:
    Qwen3_5MultiTokenPredictor = None
    IntermediateTensors = None
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.utils import sequence_parallel_chunk

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention, initialize_packed_conv_weight
from vllm_ascend.ops.gdn_checkpoint import initialize_gdn_checkpoints

_GDN_PATCH_TARGET = _GDNBaseCls
_GDN_ORIGINAL_INIT = _GDN_PATCH_TARGET.__init__


def _uses_multimodal_rope(attention: Qwen3NextAttention) -> bool:
    """Return whether a Qwen3.5 attention layer exposes multimodal RoPE."""
    return "qwen3_5" in attention.config.model_type and hasattr(attention.rotary_emb, "mrope_section")


def _get_shared_mrope_cos_sin(rotary_emb, positions: torch.Tensor, cache: dict) -> torch.Tensor:
    """Reuse only within one model forward and the identical RoPE module.

    The caller owns the local dictionary. Graph capture records the lookup as
    a device operation, so replay reads the current positions buffer.
    """
    key = id(rotary_emb)
    if key not in cache:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        cache[key] = rotary_emb.cos_sin_cache[positions]
    return cache[key]


class AscendQwen3NextAttention(Qwen3NextAttention):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        output: torch.Tensor = None,
        mrope_cos_sin: torch.Tensor | None = None,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        if _uses_multimodal_rope(self):
            # MRV2 MTP uses 1D text positions; the fused kernel reads three planes.
            if positions.ndim == 1:
                positions = positions.unsqueeze(0).expand(3, -1)
            cos_sin = mrope_cos_sin
            if cos_sin is None:
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
            if (
                attn_output.device.type == "npu"
                and attn_output.ndim == 2
                and gate.shape == attn_output.shape
                and gate.dtype == attn_output.dtype
                and attn_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
            ):
                attn_output = torch.ops.vllm.ascend_attention_sigmoid_mul(attn_output, gate)
            else:
                attn_output = attn_output * torch.sigmoid(gate)

        out, _ = self.o_proj(attn_output)
        if output is not None:
            output[:] = out
        return out


class AscendQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        mrope_cos_sin: torch.Tensor | None = None,
        **kwargs: object,
    ):
        full_num_tokens = positions.shape[-1]

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.use_attn_reduce_scatter_for_moe:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                mrope_cos_sin=mrope_cos_sin,
            )
        else:
            raise ValueError("Invalid layer_type")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype) + 1)

        if self.use_attn_reduce_scatter_for_moe:
            tp_world_size = get_tensor_model_parallel_world_size()
            # small trick using minus, eg. -17 % 8 = 7
            sp_pad = (-hidden_states.shape[0]) % tp_world_size
            # pad if not divisible by world size
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.use_attn_reduce_scatter_for_moe:
            hidden_states = self.mlp(
                hidden_states,
                already_sequence_parallel=True,
            )
        else:
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


def qwen3_5_model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    full_num_tokens = positions.shape[-1]
    if self.use_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)
        assert residual is None

    mrope_cache = {}
    aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
    for layer_idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        mrope_cos_sin = None
        if layer.layer_type == "full_attention" and _uses_multimodal_rope(layer.self_attn):
            mrope_cos_sin = _get_shared_mrope_cos_sin(layer.self_attn.rotary_emb, positions, mrope_cache)
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
            mrope_cos_sin=mrope_cos_sin,
        )
        self._maybe_add_hidden_state(aux_hidden_states, layer_idx + 1, hidden_states, residual)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
    hidden_states, _ = self.norm(hidden_states, residual)
    if self.use_sequence_parallel:
        if aux_hidden_states:
            hidden_size = hidden_states.shape[-1]
            hidden_states = torch.cat([hidden_states, *aux_hidden_states], dim=-1)
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]
            hidden_states, *aux_hidden_states = hidden_states.split(hidden_size, dim=-1)
        else:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


# Remove these patches when upstream exposes a per-forward RoPE input.
Qwen3_5Model.forward = qwen3_5_model_forward
Qwen3_5DecoderLayer.forward = AscendQwen3_5DecoderLayer.forward

if Qwen3_5MultiTokenPredictor is not None:

    def qwen3_5_mtp_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # Backport upstream Qwen3.5 MTP behavior: the local drafter runs on the
        # last PP stage and should always combine token embeddings with the
        # target hidden states instead of consuming PP intermediate tensors.
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        residual = None

        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[current_step_idx]
        if mtp_layer.use_attn_reduce_scatter_for_moe:
            # SP chunk before decoder to keep residual shape consistent with
            # reduced hidden_states after Qwen3NextDecoderLayer's reduce_scatter.
            from vllm.model_executor.models.utils import sequence_parallel_chunk

            assert hidden_states.shape[0] == positions.shape[-1]
            hidden_states = sequence_parallel_chunk(hidden_states)
            assert residual is None
        hidden_states, residual = mtp_layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if mtp_layer.use_attn_reduce_scatter_for_moe:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[: positions.shape[-1]]
        return hidden_states

    Qwen3_5MultiTokenPredictor.forward = qwen3_5_mtp_forward


Qwen3NextAttention.forward = AscendQwen3NextAttention.forward
_GDN_PATCH_TARGET._split_ba_for_tp = AscendGatedDeltaNetAttention._split_ba_for_tp
_GDN_PATCH_TARGET.get_state_shape = AscendGatedDeltaNetAttention.get_state_shape
_GDN_PATCH_TARGET.get_attn_backend = AscendGatedDeltaNetAttention.get_attn_backend

if get_current_hardware_profile().supports(HardwareCapability.GDN_COMPATIBILITY):
    from vllm_ascend._310p.ops.fla.gdn_310 import AscendGatedDeltaNetAttention310

    _GDN_PATCH_TARGET._forward_core = AscendGatedDeltaNetAttention310._forward_core
    _GDN_PATCH_TARGET.get_state_dtype = AscendGatedDeltaNetAttention310.get_state_dtype
else:
    _GDN_PATCH_TARGET._pack_conv_weights = AscendGatedDeltaNetAttention._pack_conv_weights

    @wraps(_GDN_ORIGINAL_INIT)
    def _gdn_init_with_packed_weight(self, *args, **kwargs):
        _GDN_ORIGINAL_INIT(self, *args, **kwargs)
        initialize_packed_conv_weight(self)
        initialize_gdn_checkpoints(self)

    _GDN_PATCH_TARGET.__init__ = _gdn_init_with_packed_weight
    _GDN_PATCH_TARGET.forward = AscendGatedDeltaNetAttention.forward
    _GDN_PATCH_TARGET._forward_core = AscendGatedDeltaNetAttention._forward_core
    _GDN_PATCH_TARGET._warmup_prefill_kernels = AscendGatedDeltaNetAttention._warmup_prefill_kernels
