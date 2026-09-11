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
# Step3.5/3.7 on Ascend: Attention + sequence-parallel (SP) MoE.
#
# Part 1 (Attention): replace Step3p5Attention.forward with the fused
# qkv-rmsnorm-rope path.
#
# Part 2 (SP MoE, deepseek_v2 style): data flow for an SP MoE layer:
#   norm -> all_gather (full seq) -> attention (o_proj partial sum, no
#   all-reduce) -> pad to TP multiple -> reduce_scatter (per-rank token
#   shard) -> expert layer. A dense layer receiving a sliced input restores
#   the full sequence first; Step3p5Model.forward gathers the final output
#   back before compute_logits (MTP draft reuses Step3p5DecoderLayer directly
#   and is gathered by the proposer, so no model-level gather there).
#
# The replicated __init__/forward bodies below must stay in sync with
# vllm.model_executor.models.step3p5 (vllm main). The patched methods run
# inside torch.compile fullgraph capture, so they contain NO print / logger /
# f-string-on-SymInt statements (that would graph-break or violate dynamic
# shape constraints).
#

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.models.step3p5 import (
    FP32ReplicatedLinear,
    FusedMoEBlock,
    Step3p5Attention,
    Step3p5DecoderLayer,
    Step3p5MLP,
    Step3p5Model,
)
from vllm.model_executor.models.utils import (
    extract_layer_index,
    sequence_parallel_chunk,
)

# Apply the DP=1 SP property in worker processes too (platform-side module;
# loaded in engine-core processes via vllm_ascend.patch.platform).
import vllm_ascend.patch.platform.patch_parallel_config  # noqa: F401
from vllm_ascend.device.device_op import DeviceOperator


# ---------------------------------------------------------------------------
# Part 1: Step3p5Attention.forward (fused qkv-rmsnorm-rope).
# ---------------------------------------------------------------------------
def _patched_attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    if self.use_rope:
        q, k, v = DeviceOperator.split_qkv_rmsnorm_rope(
            input=qkv,
            q_weight=self.q_norm.weight + 1.0,
            k_weight=self.k_norm.weight + 1.0,
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
        # Add qk-norm inline similar to Qwen3 MOE attention
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head.contiguous())
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head.contiguous())
        k = k_by_head.view(k.shape)

    attn_output = self.attn(q, k, v)
    if self.use_head_wise_attn_gate:
        extra_dims, _ = self.g_proj(hidden_states)
        output = (
            attn_output.view(*attn_output.shape[:-1], self.num_heads, self.head_dim)
            * extra_dims.unsqueeze(-1).sigmoid()
        )
        attn_output = output.view(*attn_output.shape)
    output, _ = self.o_proj(attn_output)
    return output


# ---------------------------------------------------------------------------
# Part 2: SP-for-MoE.
# ---------------------------------------------------------------------------
# FusedMoEBlock.__init__: replicate upstream + thread `is_sequence_parallel`
# into FusedMoEFactory. This flag selects the SP path inside the MoE runner /
# shared experts (skip final TP all-reduce, gather/reduce-scatter SP input).
def _patched_fused_moe_block_init(
    self,
    vllm_config: VllmConfig,
    prefix: str = "",
    is_sequence_parallel: bool = False,
):
    # Module-level replacement: zero-arg super() is unavailable outside a
    # class body, so call the base explicitly.
    nn.Module.__init__(self)

    self.tp_size = get_tensor_model_parallel_world_size()
    self.layer_idx = extract_layer_index(prefix)
    self.is_sequence_parallel = is_sequence_parallel

    self.ep_size = get_ep_group().device_group.size()
    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config
    parallel_config = vllm_config.parallel_config

    self.hidden_size = config.hidden_size
    self.enable_eplb = parallel_config.enable_eplb
    self.n_routed_experts = config.moe_num_experts
    self.n_logical_experts = self.n_routed_experts
    self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
    self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
    self.n_local_physical_experts = self.n_physical_experts // self.ep_size

    if self.tp_size > config.moe_num_experts:
        raise ValueError(
            f"Tensor parallel size {self.tp_size} is greater than the number of experts {config.moe_num_experts}."
        )

    self.gate = FP32ReplicatedLinear(
        config.hidden_size,
        config.moe_num_experts,
        bias=False,
        quant_config=None,
        params_dtype=torch.float32,  # Use FP32 for higher precision.
        prefix=f"{prefix}.gate",
    )
    self.use_moe_router_bias = config.use_moe_router_bias
    assert self.use_moe_router_bias, "Only support use_moe_router_bias is true."
    self.routed_scaling_factor = config.moe_router_scaling_factor
    self.router_bias = nn.Parameter(
        torch.zeros(config.moe_num_experts, dtype=torch.float32),
        requires_grad=False,
    )
    self.need_fp32_gate = config.need_fp32_gate
    assert self.need_fp32_gate, "Router logits must use FP32 precision for numerical stability."

    activation = "silu"
    swiglu_limits = config.swiglu_limits or []
    swiglu_limit = swiglu_limits[self.layer_idx] if self.layer_idx < len(swiglu_limits) else None
    if swiglu_limit not in (None, 0):
        swiglu_limit = float(swiglu_limit)
        assert swiglu_limit == 7.0, "Swiglu limit in fused moe block only support 7.0 now."
        activation = "swiglustep"
        logger.debug(
            "step3p5 layer_idx: %s, activation: %s, limit: %s",
            self.layer_idx,
            activation,
            swiglu_limit,
        )

    self.share_expert = Step3p5MLP(
        config=config,
        hidden_size=self.hidden_size,
        intermediate_size=config.share_expert_dim,
        hidden_act="silu",
        reduce_results=False,
        quant_config=quant_config,
        prefix=f"{prefix}.share_expert",
    )
    self.experts = FusedMoEFactory(
        shared_experts=self.share_expert,
        gate=self.gate,
        num_experts=config.moe_num_experts,
        top_k=config.moe_top_k,
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        renormalize=config.norm_expert_weight,
        quant_config=quant_config,
        activation=activation,
        prefix=f"{prefix}.experts",
        scoring_func=getattr(config, "moe_router_activation", "sigmoid"),
        e_score_correction_bias=self.router_bias,
        routed_scaling_factor=config.moe_router_scaling_factor,
        enable_eplb=self.enable_eplb,
        num_redundant_experts=self.n_redundant_experts,
        router_logits_dtype=torch.float32,
        is_sequence_parallel=self.is_sequence_parallel,
    )


# Step3p5DecoderLayer.__init__: replicate upstream + SP flags. The attention
# o_proj reduction is disabled by setting the instance attribute after
# construction (RowParallelLinear only reads self.reduce_results at forward
# time), which avoids replacing Step3p5Attention.__init__.
def _patched_decoder_layer_init(
    self,
    vllm_config: VllmConfig,
    prefix: str = "",
) -> None:
    # Module-level replacement: zero-arg super() is unavailable outside a
    # class body, so call the base explicitly.
    nn.Module.__init__(self)
    config = vllm_config.model_config.hf_config
    self.hidden_size = config.hidden_size
    layer_idx = extract_layer_index(prefix)
    self.layer_idx = layer_idx
    cache_config = vllm_config.cache_config
    quant_config = vllm_config.quant_config
    parallel_config = vllm_config.parallel_config
    if cache_config is not None:
        cache_config.sliding_window = None
    if config.att_impl_type == "GQA":
        num_attention_heads = None
        num_attention_groups = None
        head_dim = None
        if (
            getattr(config, "attention_other_setting", None)
            and getattr(config, "layer_types", [])
            and config.layer_types[layer_idx] == config.attention_other_setting["attention_type"]
        ):
            num_attention_heads = config.attention_other_setting["num_attention_heads"]
            num_attention_groups = config.attention_other_setting["num_attention_groups"]
            head_dim = config.attention_other_setting["head_dim"]
        partial_rotary_factors = getattr(config, "partial_rotary_factors", [])
        self.self_attn = Step3p5Attention(
            hidden_size=self.hidden_size,
            num_heads=num_attention_heads if num_attention_heads else config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=num_attention_groups if num_attention_groups else config.num_attention_groups,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=head_dim if head_dim else getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=getattr(config, "rope_scaling", None),
            sliding_window=getattr(config, "sliding_window", None),
            use_head_wise_attn_gate=getattr(config, "use_head_wise_attn_gate", False),
            layer_types=getattr(config, "layer_types", []),
            use_rope_layers=getattr(config, "use_rope_layers", []),
            yarn_only_types=getattr(config, "yarn_only_types", []),
            partial_rotary_factor=partial_rotary_factors[layer_idx] if partial_rotary_factors else 1.0,
            prefix=f"{prefix}.self_attn",
        )
    else:
        raise ValueError(f"Unsupported attention implementation: {config.att_impl_type}")
    self.use_moe = False
    self.tp_group = get_tp_group()
    self.use_fused_all_reduce = get_tensor_model_parallel_world_size() > 1 and get_dp_group().world_size == 1
    if self.use_fused_all_reduce:
        logger.warning_once("Enable custom fused all reduce...")
    else:
        logger.warning_once("Disable custom fused all reduce...")

    moe_layers_enum = getattr(config, "moe_layers_enum", None)
    if moe_layers_enum is not None:
        moe_layers_idx = [int(i) for i in moe_layers_enum.strip().split(",")]
    else:
        moe_layers_idx = [i for i in range(1, config.num_hidden_layers)]
    is_moe_layer = layer_idx in moe_layers_idx
    # SP-for-MoE (mirrors DeepseekV2DecoderLayer.use_sequence_parallel_moe).
    self.use_sequence_parallel_moe = (
        parallel_config.use_sequence_parallel_moe and parallel_config.pipeline_parallel_size == 1 and is_moe_layer
    )
    # On SP layers, keep the attention output as a partial sum (skip the
    # o_proj TP all-reduce) so it can be reduce_scattered to per-rank token
    # shards before the expert layer.
    self.self_attn.o_proj.reduce_results = not self.use_sequence_parallel_moe

    self.use_moe = is_moe_layer
    if is_moe_layer:
        self.moe = FusedMoEBlock(
            vllm_config,
            prefix=f"{prefix}.moe",
            is_sequence_parallel=self.use_sequence_parallel_moe,
        )
    else:
        self.mlp = Step3p5MLP(
            config=config,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act="silu",
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.mlp",
        )
    self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
    self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
    self.prefix = prefix


# Step3p5DecoderLayer.forward: replicate upstream + SP branch. Runs inside
# torch.compile (fullgraph), so tensor ops only, no logging.
def _patched_decoder_layer_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    full_num_tokens = positions.shape[0]
    input_is_sequence_parallel = self.use_sequence_parallel_moe and hidden_states.shape[0] != full_num_tokens
    if hidden_states.shape[0] != full_num_tokens and not self.use_moe:
        # Dense layer receiving an SP-sliced input (previous MoE-SP layer
        # output): restore the full sequence so attention/MLP run on
        # complete data.
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[:full_num_tokens]

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if input_is_sequence_parallel:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[:full_num_tokens]

    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
    )

    if self.use_sequence_parallel_moe:
        tp_world_size = get_tensor_model_parallel_world_size()
        # small trick using minus, e.g. -17 % 8 = 7
        sp_pad = (-hidden_states.shape[0]) % tp_world_size
        # pad if not divisible by world size
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
        hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
        if not input_is_sequence_parallel:
            residual = sequence_parallel_chunk(residual)

    hidden_states += residual
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)

    if self.use_moe:
        ffn_output = self.moe(hidden_states)
    else:
        ffn_output = self.mlp(hidden_states)
    hidden_states = ffn_output + residual
    return hidden_states


# Step3p5Model.forward: wrap upstream, gather the full sequence back when the
# last layer was an SP MoE layer (its output is a per-rank shard). Pure tensor
# ops so it traces under torch.compile.
_orig_step3p5_model_forward = Step3p5Model.forward


def _patched_step3p5_model_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors=None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    hidden_states = _orig_step3p5_model_forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
    )
    if isinstance(hidden_states, torch.Tensor) and hidden_states.shape[0] != positions.shape[0]:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[: positions.shape[0]]
    return hidden_states


# ---------------------------------------------------------------------------
# Install the patches (importing this module applies them).
# ---------------------------------------------------------------------------
Step3p5Attention.forward = _patched_attention_forward
FusedMoEBlock.__init__ = _patched_fused_moe_block_init
Step3p5DecoderLayer.__init__ = _patched_decoder_layer_init
Step3p5DecoderLayer.forward = _patched_decoder_layer_forward
Step3p5Model.forward = _patched_step3p5_model_forward
