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
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as _GDNBaseCls
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention, Qwen3NextSparseMoeBlock

try:
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor
    from vllm.sequence import IntermediateTensors
except ImportError:
    Qwen3_5MultiTokenPredictor = None
    IntermediateTensors = None

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather, sp_reduce_scatter
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention, initialize_packed_conv_weight
from vllm_ascend.ops.shared_quant_proj import project_with_shared_quant
from vllm_ascend.utils import enable_dense_sp

_GDN_PATCH_TARGET = _GDNBaseCls
_GDN_ORIGINAL_INIT = _GDN_PATCH_TARGET.__init__
_GDN_ORIGINAL_FORWARD = _GDN_PATCH_TARGET.forward


def _uses_multimodal_rope(attention: Qwen3NextAttention) -> bool:
    """Return whether a Qwen3.5 attention layer exposes multimodal RoPE."""
    return "qwen3_5" in attention.config.model_type and hasattr(attention.rotary_emb, "mrope_section")


class AscendQwen3NextAttention(Qwen3NextAttention):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        output: torch.Tensor = None,
        sp_num_tokens: int | None = None,
    ):
        """``sp_num_tokens`` is the full sequence's token count. Passing it
        means ``hidden_states`` holds only this rank's sequence-parallel
        shard, and ``qkv_proj`` all-gathers the rest - after its quantize, so
        the collective carries one byte per element instead of two. Everything
        from the q/k/v split on sees the full rows either way.
        """
        (qkv,) = project_with_shared_quant(self, hidden_states, ("qkv_proj",), sp_num_tokens)
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


def _sequence_parallel_dense_mlp_forward(mlp: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run a TP-sharded dense MLP on the sequence-parallel token shard.

    ``gate_up_proj``/``down_proj`` keep their TP weight shards, so every rank
    must see the same (padded) token rows: all-gather the shard, run the
    column-parallel / row-parallel GEMMs (``down_proj.reduce_results=False``
    leaves partial sums) and reduce-scatter back to the rank-local shard.
    """
    hidden_states = sp_all_gather(hidden_states)
    gate_up, _ = mlp.gate_up_proj(hidden_states)
    hidden_states = mlp.act_fn(gate_up)
    hidden_states, _ = mlp.down_proj(hidden_states)
    return sp_reduce_scatter(hidden_states)


class AscendQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    """Sequence-parallel decoder layer for the Qwen3.5 family.

    Replicates upstream ``Qwen3NextDecoderLayer.forward`` and extends it for
    Ascend sequence parallelism: under SP the attention output projections
    (``o_proj`` / GDN ``out_proj``) leave TP partial sums that this layer
    completes with a reduce-scatter, and a dense ``Qwen3NextMLP`` gathers the
    full sequence around its TP-sharded weights. MoE blocks keep the upstream
    ``already_sequence_parallel=True`` contract. The layout is driven by
    ``use_attn_reduce_scatter_for_moe``, which the patched ``__init__`` below
    also raises for the dense Qwen3.5 family when the FlashComm switch
    enables dense SP.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        use_sp = getattr(self, "use_attn_reduce_scatter_for_moe", False)
        full_num_tokens = positions.shape[-1]

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Attention (full and GDN) needs the full token sequence, and its input
        # projection all-gathers the shard itself - behind the projection's
        # quantize, so the collective carries one byte per element instead of
        # two. The sequence-parallel output projection above it leaves partial
        # sums that the reduce-scatter below completes on the shard.
        sp_num_tokens = full_num_tokens if use_sp else None

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states, sp_num_tokens=sp_num_tokens)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                sp_num_tokens=sp_num_tokens,
            )
        else:
            raise ValueError("Invalid layer_type")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype) + 1)

        if use_sp:
            hidden_states = sp_reduce_scatter(hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if use_sp:
            if isinstance(self.mlp, Qwen3NextSparseMoeBlock):
                hidden_states = self.mlp(hidden_states, already_sequence_parallel=True)
            else:
                # TP-sharded dense weights must see the same token rows on
                # every rank; restore the padded full sequence for the MLP,
                # then reduce its TP partial output back to the local shard.
                hidden_states = _sequence_parallel_dense_mlp_forward(self.mlp, hidden_states)
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


_QWEN3_5_LAYER_ORIGINAL_INIT = Qwen3_5DecoderLayer.__init__


def _qwen3_5_decoder_layer_init_with_dense_sp(
    self,
    vllm_config: VllmConfig,
    layer_type: str,
    prefix: str = "",
) -> None:
    _QWEN3_5_LAYER_ORIGINAL_INIT(self, vllm_config, layer_type, prefix=prefix)
    if not enable_dense_sp(vllm_config) or self.use_attn_reduce_scatter_for_moe:
        return
    # Dense FlashComm SP reuses the upstream SP-MoE layer contract: the flag
    # drives Qwen3NextModel's entry chunk / exit all-gather, the MTP head's
    # chunk / gather, and this patch's gather / scatter decoder layout.
    self.use_attn_reduce_scatter_for_moe = True
    # The output projections were built with reduce_results=True (the upstream
    # flag is False for dense checkpoints); under SP the decoder layer's
    # reduce-scatter completes the sum instead.
    if layer_type == "linear_attention":
        self.linear_attn.out_proj.reduce_results = False
    else:
        self.self_attn.o_proj.reduce_results = False
    if not isinstance(self.mlp, Qwen3NextSparseMoeBlock):
        self.mlp.down_proj.reduce_results = False


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
# Sequence-parallel decoder layout (upstream SP-MoE contract plus the dense
# Qwen3.5-family FlashComm SP extension) and the dense-SP layer wiring.
Qwen3_5DecoderLayer.__init__ = _qwen3_5_decoder_layer_init_with_dense_sp
Qwen3_5DecoderLayer.forward = AscendQwen3_5DecoderLayer.forward
_GDN_PATCH_TARGET._split_ba_for_tp = AscendGatedDeltaNetAttention._split_ba_for_tp
_GDN_PATCH_TARGET.get_state_shape = AscendGatedDeltaNetAttention.get_state_shape
_GDN_PATCH_TARGET.get_attn_backend = AscendGatedDeltaNetAttention.get_attn_backend

if get_current_hardware_profile().supports(HardwareCapability.GDN_COMPATIBILITY):
    from vllm_ascend._310p.ops.fla.gdn_310 import AscendGatedDeltaNetAttention310

    _GDN_PATCH_TARGET._forward_core = AscendGatedDeltaNetAttention310._forward_core
    _GDN_PATCH_TARGET.get_state_dtype = AscendGatedDeltaNetAttention310.get_state_dtype

    def _gdn_forward_gathering_sp_shard(self, hidden_states, *args, sp_num_tokens=None, **kwargs):
        """Accept the decoder layer's shard for the compatibility forward."""
        if sp_num_tokens is not None:
            hidden_states = sp_all_gather(hidden_states)[:sp_num_tokens]
        return _GDN_ORIGINAL_FORWARD(self, hidden_states, *args, **kwargs)

    _GDN_PATCH_TARGET.forward = _gdn_forward_gathering_sp_shard
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
