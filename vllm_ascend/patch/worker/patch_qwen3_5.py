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

import torch
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as _GDNBaseCls
from vllm.model_executor.models import qwen3_5 as _qwen3_5
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer as _UpstreamQwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather, sp_reduce_scatter
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention, initialize_packed_conv_weight
from vllm_ascend.utils import enable_dense_sp

logger = init_logger(__name__)

_GDN_PATCH_TARGET = _GDNBaseCls
_GDN_ORIGINAL_INIT = _GDN_PATCH_TARGET.__init__


def _uses_multimodal_rope(attention: Qwen3NextAttention) -> bool:
    """Return whether a Qwen3.5 attention layer exposes multimodal RoPE."""
    return "qwen3_5" in attention.config.model_type and hasattr(attention.rotary_emb, "mrope_section")


class AscendQwen3NextAttention(Qwen3NextAttention):
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, output: torch.Tensor = None):
        qkv, _ = self.qkv_proj(hidden_states)
        if _uses_multimodal_rope(self):
            # MRV2 MTP uses 1D text positions; the fused kernel reads three planes.
            if positions.ndim == 1:
                positions = positions.unsqueeze(0).expand(3, -1)
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

        out, _ = self.o_proj(attn_output)
        if output is not None:
            output[:] = out
        return out


def _disable_projection_reduction(projection: torch.nn.Module) -> None:
    custom_op = getattr(projection, "custom_op", None)
    if custom_op is not None:
        raise ValueError(
            "Qwen3.5 dense sequence parallelism does not support a custom "
            f"projection communication operator on {projection.prefix}."
        )
    projection.reduce_results = False


def _configure_dense_sequence_parallel(layer: _UpstreamQwen3_5DecoderLayer) -> None:
    # Reuse Qwen3NextModel's entry shard/final gather contract. Dense MLPs
    # need an explicit gather/reduce-scatter pair because upstream only
    # implements the middle-of-layer contract for MoE blocks.
    # The upstream model-level contract is currently keyed by this MoE-named
    # field, so keep that compatibility detail isolated here.
    layer.use_attn_reduce_scatter_for_moe = True
    if layer.layer_type == "linear_attention":
        _disable_projection_reduction(layer.linear_attn.out_proj)
    else:
        _disable_projection_reduction(layer.self_attn.o_proj)
    _disable_projection_reduction(layer.mlp.down_proj)


class AscendQwen3_5DecoderLayer(_UpstreamQwen3_5DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            layer_type=layer_type,
            prefix=prefix,
        )
        self._use_dense_sequence_parallel = enable_dense_sp(vllm_config)
        if self._use_dense_sequence_parallel:
            _configure_dense_sequence_parallel(self)
            logger.info_once(
                "Enabled Qwen3.5 dense sequence-parallel decoder; "
                "attention o_proj and MLP down_proj use reduce-scatter."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        **kwargs: object,
    ):
        if not self._use_dense_sequence_parallel:
            return super().forward(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                **kwargs,
            )

        full_num_tokens = positions.shape[-1]
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        if self.layer_scale:
            if hidden_states.ndim == 2:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype) + 1)
        hidden_states = sp_reduce_scatter(hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        hidden_states = self.mlp(hidden_states)
        hidden_states = sp_reduce_scatter(hidden_states)

        if self.layer_scale:
            if hidden_states.ndim == 2:
                hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                assert hidden_states.ndim == self.ffn_layer_scale.ndim, (
                    f"shape must be the same {hidden_states.ndim}, {self.ffn_layer_scale.ndim}"
                )
                hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype) + 1)

        return hidden_states, residual


# Qwen3_5Model resolves this symbol when its layer factory runs. Replacing the
# factory symbol keeps the upstream class itself untouched and also preserves
# isinstance checks because the Ascend implementation is a proper subclass.
_qwen3_5.Qwen3_5DecoderLayer = AscendQwen3_5DecoderLayer

try:
    from vllm.model_executor.models import qwen3_5_mtp as _qwen3_5_mtp
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor
    from vllm.sequence import IntermediateTensors

    # qwen3_5_mtp may have imported the decoder before this patch module.
    _qwen3_5_mtp.Qwen3_5DecoderLayer = AscendQwen3_5DecoderLayer
except ImportError:
    Qwen3_5MultiTokenPredictor = None
    IntermediateTensors = None


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

    _GDN_PATCH_TARGET.__init__ = _gdn_init_with_packed_weight
    _GDN_PATCH_TARGET.forward = AscendGatedDeltaNetAttention.forward
    _GDN_PATCH_TARGET._forward_core = AscendGatedDeltaNetAttention._forward_core
    _GDN_PATCH_TARGET._warmup_prefill_kernels = AscendGatedDeltaNetAttention._warmup_prefill_kernels
