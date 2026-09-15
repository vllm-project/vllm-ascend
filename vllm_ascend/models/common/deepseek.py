# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend feed-forward components shared by DeepSeek V4 and V4.1."""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.config import ParallelConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import PPMissingLayer, is_pp_missing_parameter

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather, sp_shard
from vllm_ascend.ops.triton.mul_add import muls_add_triton
from vllm_ascend.utils import enable_dsa_cp


class DeepseekMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekMoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_draft_layer: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        layer_idx = int(prefix.split(sep=".")[-2])
        self.layer_idx = layer_idx
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.5)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. Only silu is supported for now.")

        self.gate = ReplicatedLinear(
            config.hidden_size, config.n_routed_experts, bias=False, quant_config=None, prefix=f"{prefix}.gate"
        )
        self.gate.precast_fp32_weight = True

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = self.physical_expert_start + self.n_local_physical_experts

        self.is_fusion_moe_shared_experts_enabled = getattr(get_ascend_config(), "mix_placement", False)
        if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.hash = layer_idx < config.num_hash_layers and not is_draft_layer
        self.gate.bias_vl = None
        if getattr(config, "vision_n_layers", 0) > 0:
            self.gate.bias_vl = nn.Parameter(
                torch.empty(
                    config.n_routed_experts,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        if self.hash:
            # Use zeros instead of empty to avoid garbage values causing
            # invalid memory access in dummy mode (--load-format="dummy")
            self.gate.tid2eid = nn.Parameter(
                torch.zeros(
                    config.vocab_size,
                    config.num_experts_per_tok,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            self.gate.e_score_correction_bias = None
        else:
            self.gate.tid2eid = None
            self.gate.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            # Keep scaling outside the router path so the order matches
            # DeepSeek V4: normalize top-k weights, then scale routed output.
            # AITER applies routed_scaling_factor internally.
            routed_scaling_factor=self.routed_scaling_factor,
            swiglu_limit=self.swiglu_limit,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            bias_vl=self.gate.bias_vl,
            image_sentinel_lo=getattr(config, "image_sentinel_base_id", 129257),
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=config.n_shared_experts if self.is_fusion_moe_shared_experts_enabled else 0,
            hash_indices_table=self.gate.tid2eid,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states_fp32: torch.Tensor | None = None,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if hidden_states_fp32 is not None:
            hidden_states_fp32 = hidden_states_fp32.view(-1, hidden_dim)
        # Chunk the hidden states so they aren't replicated across TP ranks.
        # This avoids duplicate computation in self.experts.
        # TODO: We can replace the all_reduce at the end of attn with a
        # reduce_scatter instead of chunking here.
        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sp_shard(hidden_states)
            if hidden_states_fp32 is not None:
                hidden_states_fp32 = sp_shard(hidden_states_fp32)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoEFactory class
            router_input = hidden_states if hidden_states_fp32 is None else hidden_states_fp32
            fused_moe_out = self.experts(
                hidden_states=hidden_states,
                router_logits=router_input,
                input_ids=input_ids,
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_input = hidden_states.float() if hidden_states_fp32 is None else hidden_states_fp32
            router_logits = F.linear(router_input, self.gate.weight)
            fused_moe_out = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )

        fused_moe_out_is_tuple = isinstance(fused_moe_out, tuple)
        if fused_moe_out_is_tuple:
            shared_output, final_hidden_states = fused_moe_out
            if self.shared_experts is None:
                assert shared_output is None

            if hidden_states.dtype != torch.float16:
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

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = sp_all_gather(final_hidden_states)
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1 and fused_moe_out_is_tuple:
            # Legacy tuple outputs are reduced here. Tensor outputs from the
            # upstream MoERunner have already gone through its final reduction.
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


def get_spec_layer_idx_from_weight_name(config: PretrainedConfig, weight_name: str) -> int | None:
    if weight_name.startswith("mtp."):
        return 0
    return None


class DeepseekMixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list[DeepseekMoE]
    """
    List of MoE MLP layers in the model.
    """

    def extract_moe_parameters(self, example_moe: DeepseekMoE | None):
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
        else:
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class DeepseekForCausalLMMixin(DeepseekMixtureOfExperts):
    def set_moe_parameters(self):
        self.expert_weights = []

        self.num_expert_groups = getattr(self.config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            if isinstance(layer.mlp, DeepseekMoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts if getattr(get_ascend_config(), "mix_placement", False) else 0),
            num_redundant_experts=0,
        )

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model._set_aux_hidden_state_layers(layers)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        fuse_shared_experts = getattr(get_ascend_config(), "mix_placement", False)
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + (self.config.n_shared_experts if fuse_shared_experts else 0),
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        for name, loaded_weight in weights:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            # TODO:
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
            if ".gate.bias_vl" in name:
                # The parameter keeps the checkpoint name on Ascend. It is
                # passed to the hash router as its vision-only correction
                # bias, while text rows continue to use tid2eid.
                pass
            elif ".gate.bias" in name:
                name = name.replace(".gate.bias", ".gate.e_score_correction_bias")

            # Hash-router layers route text tokens through ``tid2eid`` and keep
            # ``e_score_correction_bias`` unset, but the checkpoint still ships
            # a router bias for them. Skip it instead of raising a KeyError.
            if name.endswith(".gate.e_score_correction_bias") and name not in params_dict:
                continue

            if "sink" in name:
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                if enable_dsa_cp():
                    param.data.copy_(loaded_weight)
                else:
                    # Handle attention sinks (distributed across ranks)
                    narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
                    param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue

            is_fusion_moe_shared_experts_layer = fuse_shared_experts and ("mlp.shared_experts" in name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                # if go with fusion option, then update name
                if (param_name == "fused_qkv_a_proj") and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                # Skip loading extra bias for GPTQ models.
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

                # Special handling: when AITER fusion_shared_experts is enabled,
                # checkpoints may provide a single widened shared_experts tensor
                # without explicit expert indices
                # (e.g. ...mlp.shared_experts.gate_proj.weight).
                # For models with multiple shared experts, split that tensor
                # evenly into per-shared-expert slices and load them into
                # appended expert slots mlp.experts.{n_routed_experts + j}.*
                # accordingly.
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    # Determine split axis based on op type
                    # gate/up: ColumnParallel → split along dim 0
                    # down: RowParallel → split along dim 1
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
                        # Synthesize an expert-style name so expert mapping
                        # can route it
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    # Use expert_params_mapping to locate the destination
                    # param and delegate to its expert-aware weight_loader
                    # with expert_id.
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # Anyway, this is an expert weight and should not be
                        # attempted to load as other weights later
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here
                        # Instead, create a new variable
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        if is_pp_missing_parameter(name_mapped, self):
                            continue

                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
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
                        if is_expert_weight:
                            # We've checked that this is an expert weight
                            # However it's not mapped locally to this rank
                            # So we simply skip it
                            continue

                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        # Remapping the name of FP8 kv-scale.
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


def init_attention_projections(self, config, quant_config, prefix, reduce_results):
    tp_size = get_tensor_model_parallel_world_size()
    self.dim = config.hidden_size
    self.n_heads = config.num_attention_heads
    self.n_local_heads = config.num_attention_heads // tp_size
    self.q_lora_rank = config.q_lora_rank
    self.o_lora_rank = config.o_lora_rank
    self.head_dim = config.head_dim
    self.rope_head_dim = config.qk_rope_head_dim
    self.nope_head_dim = config.head_dim - config.qk_rope_head_dim
    self.n_groups = config.o_groups
    self.n_local_groups = self.n_groups // tp_size
    self.window_size = config.sliding_window
    self.eps = config.rms_norm_eps
    self.norm_eps = config.rms_norm_eps
    self.scale = self.head_dim**-0.5
    self.enable_dsa_cp = enable_dsa_cp()

    attn_sink_heads = self.n_heads if self.enable_dsa_cp else self.n_local_heads
    self.attn_sink = nn.Parameter(torch.empty(attn_sink_heads, dtype=torch.float32))
    self.wq_a = ReplicatedLinear(
        self.dim,
        self.q_lora_rank,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.wq_a",
        return_bias=False,
    )
    self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
    self.q_norm_without_weight = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)
    wq_b_cls = ReplicatedLinear if self.enable_dsa_cp else ColumnParallelLinear
    self.wq_b = wq_b_cls(
        self.q_lora_rank,
        self.n_heads * self.head_dim,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.wq_b",
        return_bias=False,
    )

    self.wkv = ReplicatedLinear(
        self.dim,
        self.head_dim,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.wkv",
        return_bias=False,
    )
    self.kv_norm = RMSNorm(self.head_dim, self.norm_eps)
    self.wo_a = ColumnParallelLinear(
        self.n_heads * self.head_dim // self.n_groups,
        self.n_groups * config.o_lora_rank,
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.wo_a",
        return_bias=False,
    )
    # Every DSA o_proj path consumes wo_a.weight directly via
    # npu_transpose_batchmatmul / npu_transpose_quant_batchmatmul,
    # so the weight must remain ND.
    self.wo_a.skip_weight_nz_conversion = True
    self.wo_b = RowParallelLinear(
        self.n_groups * config.o_lora_rank,
        self.dim,
        bias=False,
        reduce_results=reduce_results,
        quant_config=quant_config,
        prefix=f"{prefix}.wo_b",
        return_bias=False,
    )
