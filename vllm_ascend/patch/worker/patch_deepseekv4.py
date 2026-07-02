#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is part of the vllm-ascend project.
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
"""AFD (Attention-FFN Disaggregation) patches for DeepSeek V4.

This module splits ``DeepseekV2DecoderLayer.forward`` into an attention-side
``compute_attn_output`` and an FFN-side ``compute_ffn_output`` so that the two
halves can run on separate workers and exchange intermediates through the AFD
connector (``NPUP2PAFDConnector``).
"""

import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any, Optional

import torch
import torch.nn.functional as F
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config  # noqa: F401  # kept for downstream patches
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.sequence import IntermediateTensors

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.models.deepseek_v4 import (
    AscendDeepseekV4ForCausalLM,
    DeepseekV2DecoderLayer,
    DeepseekV4Model,
    DeepseekV4MoE,
    get_spec_layer_idx_from_weight_name,
)
from vllm_ascend.ops.triton.mul_add import muls_add_triton
from vllm_ascend.utils import enable_dsa_cp


# ---------------------------------------------------------------------------
# DeepseekV4MoE.afd_forward
# ---------------------------------------------------------------------------
def afd_forward(
    self: DeepseekV4MoE,
    hidden_states: torch.Tensor,
    router_logits: Optional[torch.Tensor] = None,
    group_list: Optional[torch.Tensor] = None,
    dynamic_scales: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    row_idx: Optional[torch.Tensor] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    cam_p2p_ep_name: str = "",
) -> torch.Tensor:
    """FFN-side MoE forward driven by the AFD connector.

    Mirrors ``DeepseekV4MoE.forward`` but delegates the expert computation to
    ``afd_connector.compute_moe`` so that routing tensors produced on the
    attention side can be consumed here. The shared-output merge, scaling and
    sequence-parallel/all-reduce handling match the regular forward path.
    """
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    if self.is_sequence_parallel:
        from vllm.model_executor.models.utils import sequence_parallel_chunk
        hidden_states = sequence_parallel_chunk(hidden_states)

    forward_ctx = get_forward_context()
    afd_connector = forward_ctx.afd_metadata.afd_connector
    fused_moe_out = afd_connector.compute_moe(
        experts=self.experts,
        hidden_states=hidden_states,
        router_logits=router_logits,
        group_list=group_list,
        dynamic_scales=dynamic_scales,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        row_idx=row_idx,
        x_active_mask=x_active_mask,
        cam_p2p_ep_name=cam_p2p_ep_name,
        connector_name=getattr(self, "connector_name", None),
    )

    fused_moe_out_is_tuple = isinstance(fused_moe_out, tuple)
    if fused_moe_out_is_tuple:
        shared_output, final_hidden_states = fused_moe_out
        if self.shared_experts is None:
            assert shared_output is None

        # Fix FP16 overflow (see DeepseekV2DecoderLayer for details).
        if hidden_states.dtype != torch.float16:
            if not self.is_rocm_aiter_moe_enabled:
                if self.shared_experts is not None:
                    assert shared_output is not None
                    final_hidden_states = muls_add_triton(
                        final_hidden_states, shared_output, self.routed_scaling_factor
                    )
                else:
                    final_hidden_states *= self.routed_scaling_factor
        elif self.shared_experts is not None:
            assert shared_output is not None
            final_hidden_states = muls_add_triton(
                shared_output, final_hidden_states, 1.0 / self.routed_scaling_factor
            )
    else:
        final_hidden_states = fused_moe_out

    if self.is_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]
    elif self.tp_size > 1 and fused_moe_out_is_tuple:
        # Legacy tuple outputs are reduced here. Tensor outputs from the
        # upstream MoERunner have already gone through its final reduction.
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states
        )

    return final_hidden_states.view(num_tokens, hidden_dim)


# ---------------------------------------------------------------------------
# DeepseekV2DecoderLayer.compute_attn_output / compute_ffn_output
# ---------------------------------------------------------------------------
def compute_attn_output(
    self: DeepseekV2DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    llama_4_scaling: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention half of ``DeepseekV2DecoderLayer.forward``.

    Runs ``hc_pre -> input_layernorm -> self_attn -> hc_post`` and returns the
    attention output that will be sent to the FFN worker.
    """
    residual = hidden_states.clone()
    hidden_states, post, comb = self.hc_pre(
        hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
    )
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
        llama_4_scaling=llama_4_scaling,
    )
    hidden_states = self.hc_post(hidden_states, residual, post, comb)
    return hidden_states


def compute_ffn_output(
    self: DeepseekV2DecoderLayer,
    layer_idx: int,
    hidden_states: torch.Tensor,
    router_logits: Optional[torch.Tensor] = None,
    group_list: Optional[torch.Tensor] = None,
    dynamic_scales: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    row_idx: Optional[torch.Tensor] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    cam_p2p_ep_name: str = "",
) -> torch.Tensor:
    """FFN half of ``DeepseekV2DecoderLayer.forward``.

    Runs ``hc_pre -> post_attention_layernorm -> mlp.afd_forward -> hc_post``
    using the routing tensors received from the attention worker.
    """
    residual = hidden_states.clone()
    hidden_states, post, comb = self.hc_pre(
        hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp.afd_forward(
        hidden_states,
        router_logits=router_logits,
        group_list=group_list,
        dynamic_scales=dynamic_scales,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        row_idx=row_idx,
        x_active_mask=x_active_mask,
        cam_p2p_ep_name=cam_p2p_ep_name,
    )
    hidden_states = self.hc_post(hidden_states, residual, post, comb)
    return hidden_states


# ---------------------------------------------------------------------------
# DeepseekV4Model.forward_m2n / forward (AFD dispatch)
# ---------------------------------------------------------------------------
def forward_m2n(
    self: DeepseekV4Model,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    positions: torch.Tensor,
    afd_metadata: Any,
    llama_4_scaling: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Attention-side layer loop that ships intermediates to the FFN worker.

    For each decoder layer:
      1. (layer > 0) receive the previous layer's FFN output,
      2. compute the attention output,
      3. optionally compute the router gate + ``select_experts`` when
         ``compute_gate_on_attention`` is enabled,
      4. send the attention output (and routing tensors) to the FFN worker.
    After the loop, the final FFN output is received.
    """
    afd_connector = afd_metadata.afd_connector
    # NOTE: avoid calling get_current_vllm_config() here because torch dynamo
    # compilation runs outside set_current_vllm_config() context. The connector
    # already caches afd_config at init time.
    afd_config = getattr(afd_connector, "afd_config", None)

    for layer in islice(self.layers, self.start_layer, self.end_layer):
        if layer.layer_idx > 0:
            hidden_states = afd_connector.recv_ffn_output(
                hidden_states=hidden_states, metadata=None
            )

        hidden_states = layer.compute_attn_output(
            positions, hidden_states, residual, llama_4_scaling
        )

        router_logits = None
        topk_weights = None
        topk_ids = None
        if afd_config is not None and afd_config.compute_gate_on_attention:
            router_logits = F.linear(hidden_states.float(), layer.mlp.gate.weight)
            topk_weights, topk_ids = afd_connector.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=layer.mlp.experts.top_k,
                use_grouped_topk=True,
                renormalize=getattr(self.config, "norm_topk_prob", True),
                topk_group=getattr(self.config, "topk_group", 1),
                num_expert_group=getattr(self.config, "n_group", 1),
                e_score_correction_bias=layer.mlp.gate.e_score_correction_bias,
            )
            topk_weights = topk_weights.to(torch.float)

        afd_connector.send_attn_output(
            hidden_states=hidden_states,
            metadata=None,
            router_logits=router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

    hidden_states = afd_connector.recv_ffn_output(
        hidden_states=hidden_states, metadata=afd_metadata
    )
    return hidden_states, residual


def afd_model_forward(
    self: DeepseekV4Model,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: Optional[IntermediateTensors],
    inputs_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor | IntermediateTensors:
    """Replacement ``DeepseekV4Model.forward`` with an AFD dispatch branch.

    When ``afd_metadata`` is present in the forward context, the decoder layer
    loop is replaced by ``forward_m2n``; otherwise the original layer-by-layer
    path is used. The MTP hidden-state stash and ``hc_head``/``norm`` epilogue
    are preserved unchanged.
    """
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = None

    # Compute llama 4 scaling once per forward pass if enabled
    llama_4_scaling_config = None
    llama_4_scaling: torch.Tensor | None
    if llama_4_scaling_config is not None:
        from vllm.model_executor.models.deepseek_v2 import _get_llama_4_scaling

        llama_4_scaling = _get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config[
                "original_max_position_embeddings"
            ],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    if get_pp_group().is_first_rank:
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

    forward_ctx = get_forward_context()
    afd_metadata = forward_ctx.afd_metadata if forward_ctx is not None else None

    if afd_metadata is not None:
        hidden_states, residual = self.forward_m2n(
            hidden_states, residual, positions, afd_metadata, llama_4_scaling
        )
    else:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions, hidden_states, residual, llama_4_scaling
            )

    # Stash pre-hc_head residual for the MTP draft (captured copy_).
    if forward_ctx is not None and forward_ctx.flash_comm_v1_enabled:
        h_states_flat = tensor_model_parallel_all_gather(
            hidden_states.flatten(1), dim=0
        )
        pad_size = forward_ctx.pad_size
        if pad_size > 0:
            h_states_flat = h_states_flat[:-pad_size]
        num_tokens = h_states_flat.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(h_states_flat)
    else:
        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states})

    hidden_states = self.hc_head(
        hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
    )
    hidden_states = self.norm(hidden_states)
    return hidden_states


# ---------------------------------------------------------------------------
# AscendDeepseekV4ForCausalLM AFD helpers
# ---------------------------------------------------------------------------
def is_moe_weight(self: AscendDeepseekV4ForCausalLM, name: str) -> bool:
    """Return True for expert / shared-expert / router-gate weights."""
    if (
        "shared_experts" in name
        or "experts" in name
        or "gate" in name
        or "up" in name
        or "down" in name
    ):
        return True
    return False


def is_common_weight(self: AscendDeepseekV4ForCausalLM, name: str) -> bool:
    """Return True for weights required by both AFD roles.

    Layernorms and hc_* structural parameters are needed on both sides because
    ``hc_pre``/``hc_post`` are invoked in both ``compute_attn_output`` and
    ``compute_ffn_output``.
    """
    if (
        "lm_head" in name
        or "model.norm.weight" in name
        or "embed_tokens" in name
        or "input_layernorm" in name
        or "post_attention_layernorm" in name
        or "hc_" in name
    ):
        return True
    return False


def model_compute_ffn_output(
    self: AscendDeepseekV4ForCausalLM,
    hidden_states: torch.Tensor,
    layer_idx: int,
    router_logits: Optional[torch.Tensor] = None,
    group_list: Optional[torch.Tensor] = None,
    dynamic_scales: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    row_idx: Optional[torch.Tensor] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    cam_p2p_ep_name: str = "",
) -> torch.Tensor:
    """Model-level FFN entry point used by ``NPUFFNModelRunner``."""
    if self.afd_config is not None and self.afd_config.compute_gate_on_attention:
        hidden_states = self.model.layers[layer_idx].compute_ffn_output(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            router_logits=router_logits,
            group_list=group_list,
            dynamic_scales=dynamic_scales,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            row_idx=row_idx,
            x_active_mask=x_active_mask,
            cam_p2p_ep_name=cam_p2p_ep_name,
        )
    else:
        hidden_states = self.model.layers[layer_idx].compute_ffn_output(
            layer_idx=layer_idx, hidden_states=hidden_states
        )
    return hidden_states


def afd_load_weights(
    self: AscendDeepseekV4ForCausalLM, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    """AFD-aware ``load_weights``.

    * attention role: skip MoE expert weights; load the router gate when
      ``compute_gate_on_attention`` is enabled.
    * ffn role: skip non-MoE / non-common weights; skip the router gate when
      ``compute_gate_on_attention`` is enabled (attention side owns it).
    """
    rocm_aiter_moe_shared_expert_enabled = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
    rocm_aiter_moe_shared_expert_enabled = getattr(get_ascend_config(), "mix_placement", False)
    stacked_params_mapping = [
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        self.model,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.n_routed_experts
        + (self.config.n_shared_experts if rocm_aiter_moe_shared_expert_enabled else 0),
        num_redundant_experts=self.num_redundant_experts,
    )

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    heads_per_rank = self.config.num_attention_heads // tp_size
    head_start = tp_rank * heads_per_rank

    for name, loaded_weight in weights:
        spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
        if spec_layer is not None:
            continue  # skip spec decode layers for main model

        if not name.startswith("model"):
            name = f"model.{name}"

        if ".w1." in name:
            name = name.replace(".w1.", ".gate_proj.")
        if ".w2." in name:
            name = name.replace(".w2.", ".down_proj.")
        if ".w3." in name:
            name = name.replace(".w3.", ".up_proj.")

        if "model.head." in name and "model.lm_head." not in name:
            name = name.replace("model.head.", "lm_head.")
        if "model.lm_head." in name:
            name = name.replace("model.lm_head.", "lm_head.")
        if "embed." in name and "embed_token." not in name:
            name = name.replace("embed.", "embed_tokens.")
        if "attn" in name and "self_attn" not in name:
            name = name.replace(".attn.", ".self_attn.")
        if ".ffn." in name:
            name = name.replace(".ffn.", ".mlp.")
        if ".ffn_norm." in name:
            name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
        if ".attn_norm." in name:
            name = name.replace(".attn_norm.", ".input_layernorm.")
        if name.endswith(".scale"):
            name = name.replace(".scale", ".weight_scale")

        if "rotary_emb.inv_freq" in name:
            continue
        if ".gate.bias" in name:
            name = name.replace(".gate.bias", ".gate.e_score_correction_bias")

        # ---------------------------------------------------------------
        # AFD role filtering
        # ---------------------------------------------------------------
        # Attention role: load the router gate when compute_gate_on_attention
        # is enabled (the gate lives at ``mlp.gate.*`` in this model).
        if (
            self.afd_role == "attention"
            and self.afd_config is not None
            and self.afd_config.compute_gate_on_attention
            and "mlp.gate." in name
        ):
            if not is_pp_missing_parameter(name, self) and name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            continue

        # Attention role: skip MoE expert weights.
        if self.afd_role == "attention" and self.is_moe_weight(name):
            continue

        # FFN role: skip the router gate when compute_gate_on_attention is
        # enabled (the attention side owns it).
        if (
            self.afd_role == "ffn"
            and self.afd_config is not None
            and self.afd_config.compute_gate_on_attention
            and "mlp.gate." in name
        ):
            continue
        # ---------------------------------------------------------------

        if "sink" in name:
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            if enable_dsa_cp():
                param.data.copy_(loaded_weight)
            else:
                narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
            loaded_params.add(name)
            continue

        is_fusion_moe_shared_experts_layer = (
            rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)
        )

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            if is_fusion_moe_shared_experts_layer:
                continue
            name_mapped = name.replace(weight_name, param_name)

            if (param_name == "fused_qkv_a_proj") and name_mapped not in params_dict:
                continue
            else:
                name = name_mapped
            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            is_expert_weight = False

            num_chunks = 1
            if is_fusion_moe_shared_experts_layer:
                num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                split_dim = 1 if "down_proj.weight" in name else 0
                total = loaded_weight.shape[split_dim]
                assert total % num_chunks == 0, (
                    f"Shared expert weight dim {total} not divisible by num_chunks {num_chunks}"
                )
                chunk_size = total // num_chunks

            for j in range(num_chunks):
                chunk_name = name
                weight_to_load = loaded_weight

                if is_fusion_moe_shared_experts_layer:
                    if split_dim == 0:
                        weight_to_load = loaded_weight[j * chunk_size : (j + 1) * chunk_size, :]
                    else:
                        weight_to_load = loaded_weight[:, j * chunk_size : (j + 1) * chunk_size]
                    chunk_name = name.replace(
                        "mlp.shared_experts",
                        f"mlp.experts.{self.config.n_routed_experts + j}",
                    )

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in chunk_name:
                        continue

                    is_expert_weight = True
                    name_mapped = chunk_name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                    success = weight_loader(
                        param,
                        weight_to_load,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        if not is_fusion_moe_shared_experts_layer:
                            name = name_mapped
                        else:
                            loaded_params.add(name_mapped)
                        break
                else:
                    # FFN role: skip non-MoE, non-common weights (e.g. attn).
                    if (
                        self.afd_role == "ffn"
                        and not self.is_moe_weight(name)
                        and not self.is_common_weight(name)
                    ):
                        continue
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)

    return loaded_params


# ---------------------------------------------------------------------------
# Apply monkey-patches
# ---------------------------------------------------------------------------
# DeepseekV4MoE gains an afd_forward entry point for the FFN worker.
DeepseekV4MoE.afd_forward = afd_forward  # type: ignore[assignment]

# Decoder layer split into attention / FFN halves.
DeepseekV2DecoderLayer.compute_attn_output = compute_attn_output  # type: ignore[assignment]
DeepseekV2DecoderLayer.compute_ffn_output = compute_ffn_output  # type: ignore[assignment]

# Model-level AFD dispatch.
DeepseekV4Model.forward_m2n = forward_m2n  # type: ignore[assignment]
DeepseekV4Model.forward = afd_model_forward  # type: ignore[assignment]

# CausalLM AFD helpers + weight loading.
_orig_deepseekv4_init = AscendDeepseekV4ForCausalLM.__init__


def _afd_deepseekv4_init(self: AscendDeepseekV4ForCausalLM, *, vllm_config, prefix: str = ""):
    _orig_deepseekv4_init(self, vllm_config=vllm_config, prefix=prefix)
    self.afd_config = getattr(vllm_config, "afd_config", None)
    self.afd_role = (
        self.afd_config.afd_role if self.afd_config is not None else None
    )


AscendDeepseekV4ForCausalLM.__init__ = _afd_deepseekv4_init  # type: ignore[assignment]
AscendDeepseekV4ForCausalLM.is_moe_weight = is_moe_weight  # type: ignore[assignment]
AscendDeepseekV4ForCausalLM.is_common_weight = is_common_weight  # type: ignore[assignment]
AscendDeepseekV4ForCausalLM.compute_ffn_output = model_compute_ffn_output  # type: ignore[assignment]
AscendDeepseekV4ForCausalLM.load_weights = afd_load_weights  # type: ignore[assignment]
