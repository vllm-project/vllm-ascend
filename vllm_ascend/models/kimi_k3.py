# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 model adapters for vLLM 0.27 on Ascend.

vLLM owns Kimi's configuration, multimodal processor, weight mappings, and
model-level forward contract.  This module composes those upstream pieces with
the generic MLA/MoE implementation and the Ascend KDA backend.
"""

import math
from copy import copy
from typing import NamedTuple

import torch
import vllm.envs as envs
from torch import nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    init_vllm_registered_model,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.kimi_k3.amd.linear import (
    KimiDecoderLayer as UpstreamKimiDecoderLayer,
)
from vllm.models.kimi_k3.amd.linear import KimiLinearForCausalLM as UpstreamKimiLinearForCausalLM
from vllm.models.kimi_k3.amd.linear import KimiLinearModel as UpstreamKimiLinearModel
from vllm.models.kimi_k3.amd.linear import (
    KimiMLAAttention as UpstreamKimiMLAAttention,
)
from vllm.models.kimi_k3.amd.linear import (
    KimiMLP,
    KimiRoutedOutputTransform,
)
from vllm.models.kimi_k3.amd.model import (
    KimiK3ForConditionalGeneration as UpstreamKimiK3ForConditionalGeneration,
)
from vllm.models.kimi_k3.common.mm_preprocess import (
    KimiK3DummyInputsBuilder,
    KimiK3MultiModalProcessor,
    KimiK3ProcessingInfo,
)
from vllm.models.kimi_k3.nvidia.model import (
    KimiLinearModel as UpstreamPackedKimiLinearModel,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import HAS_TRITON
from vllm.utils.math_utils import cdiv

from vllm_ascend import envs as ascend_envs
from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention  # type: ignore[import-untyped]
from vllm_ascend.utils import get_rotation_path

if HAS_TRITON:
    from vllm_ascend.ops.triton.kimi_k3.attention_residual import (  # type: ignore[import-untyped]
        apply_attn_res,
    )
else:
    apply_attn_res = None  # type: ignore[assignment]


def _apply_ascend_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: ReplicatedLinear,
    norm: RMSNorm,
    num_valid_blocks: int,
) -> torch.Tensor:
    """Apply Kimi's canonical learned residual mixture with native ops."""
    if num_valid_blocks <= 0:
        return prefix_sum

    if apply_attn_res is not None and prefix_sum.device.type == "npu" and prefix_sum.numel() > 0:
        return apply_attn_res(
            prefix_sum,
            block_residual,
            proj,
            norm,
            num_valid_blocks,
        )

    values = torch.cat(
        (
            block_residual[:, :num_valid_blocks, :],
            prefix_sum.unsqueeze(1),
        ),
        dim=1,
    )
    values_fp32 = values.float()
    inverse_rms = torch.rsqrt(values_fp32.square().mean(-1, keepdim=True) + norm.variance_epsilon)
    normalized_without_gamma = values_fp32 * inverse_rms
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (normalized_without_gamma * score_weight).sum(-1)
    probabilities = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probabilities, values_fp32).squeeze(1).to(values.dtype)


class AttnResPhase1Stats(NamedTuple):
    """Historical statistics for all slots in one K3 AttnRes block."""

    inter_numerator: torch.Tensor
    inter_max: torch.Tensor
    inter_exp_sum: torch.Tensor


class AttnResPhase2Slot(NamedTuple):
    """Query and historical statistics selected for one AttnRes slot."""

    effective_query: torch.Tensor
    inter_numerator: torch.Tensor
    inter_max: torch.Tensor
    inter_exp_sum: torch.Tensor


try:
    from cann_ops_transformer.ops import (  # type: ignore[import-not-found]
        block_attn_res_prepare as _block_attn_res_prepare_fused,
    )
    from cann_ops_transformer.ops import (  # type: ignore[import-not-found]
        block_attn_res_update as _block_attn_res_update_fused,
    )
except ImportError:
    _block_attn_res_prepare_fused = None  # type: ignore[assignment]
    _block_attn_res_update_fused = None  # type: ignore[assignment]


def _use_fused_attn_res() -> bool:
    """Whether the CANNBot DSL fused AttnRes backend is usable."""
    return (
        ascend_envs.VLLM_ASCEND_KIMI_K3_ATTNRES_FUSED_ENABLED
        and _block_attn_res_prepare_fused is not None
        and _block_attn_res_update_fused is not None
    )


def _prepare_attn_res_phase1(
    block_residual: torch.Tensor,
    effective_queries: torch.Tensor,
    epsilon: float,
) -> AttnResPhase1Stats:
    """Prepare FP32 Online Softmax statistics for every slot in one block.

    block_residual holds the completed block-start states on its (dynamic)
    block axis, so every row is a valid candidate. The slot RMS normalization
    is query-independent, so the per-slot statistics are computed once and
    shared by every Attention/MLP sublayer of the block (and reused across the
    block's applications that merge the running partial in Phase 2).
    """
    values_float = block_residual.float()
    inv_rms = torch.rsqrt(values_float.square().mean(dim=-1) + epsilon)
    inter_logits = torch.matmul(values_float, effective_queries.transpose(0, 1)).permute(2, 0, 1) * inv_rms.unsqueeze(0)
    inter_max = inter_logits.max(dim=2).values
    inter_exp = torch.exp(inter_logits - inter_max.unsqueeze(2))
    inter_exp_sum = inter_exp.sum(dim=2)
    inter_numerator = torch.matmul(inter_exp.permute(1, 0, 2), values_float).permute(1, 0, 2)
    return AttnResPhase1Stats(
        inter_numerator=inter_numerator,
        inter_max=inter_max,
        inter_exp_sum=inter_exp_sum,
    )


def _merge_attn_res_slot(
    partial_float: torch.Tensor,
    slot: AttnResPhase2Slot,
    epsilon: float,
) -> torch.Tensor:
    """Merge one candidate (the running partial) with Phase 1 statistics.

    Mirrors the reference Online Softmax step: the partial candidate's logit
    is the RMS-normalized learned query score and is folded into the shared
    softmax over the historical block states.
    """
    input_logit = torch.matmul(partial_float, slot.effective_query) * torch.rsqrt(
        partial_float.square().mean(dim=-1) + epsilon
    )
    merged_max = torch.maximum(slot.inter_max, input_logit)
    inter_scale = torch.exp(slot.inter_max - merged_max)
    input_scale = torch.exp(input_logit - merged_max)
    merged_exp_sum = inter_scale * slot.inter_exp_sum + input_scale
    merged_numerator = inter_scale.unsqueeze(-1) * slot.inter_numerator + input_scale.unsqueeze(-1) * partial_float
    return merged_numerator / merged_exp_sum.unsqueeze(-1)


def _update_attn_res_phase2(
    partial_block: torch.Tensor,
    partial_delta: torch.Tensor,
    slot: AttnResPhase2Slot,
    epsilon: float,
) -> torch.Tensor:
    """Update partial in place, then merge one selected slot with Online Softmax."""
    partial_updated = (partial_block.float() + partial_delta.float()).to(partial_block.dtype)
    partial_block.copy_(partial_updated)
    return _merge_attn_res_slot(partial_block.float(), slot, epsilon).to(partial_block.dtype)


def _merge_attn_res_partial(
    partial_block: torch.Tensor,
    slot: AttnResPhase2Slot,
    epsilon: float,
) -> torch.Tensor:
    """Merge the already-accumulated partial without folding in a delta.

    Used when a pipeline rank starts in the middle of a block: the incoming
    hidden_states already contain every delta, so the partial must not be
    updated again before the first Attention phase-2 merge.
    """
    return _merge_attn_res_slot(partial_block.float(), slot, epsilon).to(partial_block.dtype)


class AscendKimiMLP(KimiMLP):
    """Keep TP-sharded dense weights compatible with sequence-sharded tokens."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
        use_sequence_parallel: bool = False,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            reduce_results=False if use_sequence_parallel else reduce_results,
            prefix=prefix,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
        )
        self.use_sequence_parallel = use_sequence_parallel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_sequence_parallel:
            # All weight shards must operate on the same tokens. Reducing
            # different sequence shards would mix live and padding rows.
            x = sp_all_gather(x)
        x = super().forward(x)
        if self.use_sequence_parallel:
            x = sp_reduce_scatter(x)
        return x


class AscendKimiMoE(nn.Module):
    """Kimi K3 MoE assembled from the standard vLLM MoE interfaces."""

    def __init__(
        self,
        *,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        moe_intermediate_size = config.moe_intermediate_size
        num_experts = config.num_experts
        num_experts_per_token = config.num_experts_per_token
        assert moe_intermediate_size is not None
        assert num_experts is not None
        assert num_experts_per_token is not None

        routed_expert_hidden_size = config.routed_expert_hidden_size
        self.use_latent_moe = routed_expert_hidden_size is not None
        self.moe_hidden_size = routed_expert_hidden_size or hidden_size
        self.latent_moe_use_norm = config.latent_moe_use_norm
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_shared_experts = config.num_shared_experts
        activation_situ_beta = config.activation_situ_beta if config.hidden_act == "situ" else None
        activation_situ_linear_beta = config.activation_situ_linear_beta if config.hidden_act == "situ" else None

        self.gate = GateLinear(
            input_size=hidden_size,
            output_size=num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(num_experts, dtype=torch.float32))

        if self.num_shared_experts is not None:
            self.shared_experts = KimiMLP(
                hidden_size=hidden_size,
                intermediate_size=moe_intermediate_size * self.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                activation_situ_beta=activation_situ_beta,
                activation_situ_linear_beta=activation_situ_linear_beta,
            )
        else:
            self.shared_experts = None

        latent_quant_config = quant_config if quant_config is not None and quant_config.get_name() == "ascend" else None
        if self.use_latent_moe:
            self.routed_expert_down_proj = ReplicatedLinear(
                hidden_size,
                self.moe_hidden_size,
                bias=False,
                quant_config=latent_quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            self.routed_expert_norm = (
                RMSNorm(self.moe_hidden_size, eps=config.rms_norm_eps) if self.latent_moe_use_norm else None
            )
            self.routed_expert_up_proj = ReplicatedLinear(
                self.moe_hidden_size,
                hidden_size,
                bias=False,
                quant_config=latent_quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
            self.routed_output_transform = KimiRoutedOutputTransform(
                self.routed_expert_norm,
                self.routed_expert_up_proj,
            )
        else:
            self.routed_expert_down_proj = None
            self.routed_expert_norm = None
            self.routed_expert_up_proj = None
            self.routed_output_transform = None

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            num_experts=num_experts,
            top_k=num_experts_per_token,
            hidden_size=self.moe_hidden_size,
            intermediate_size=moe_intermediate_size,
            activation=config.hidden_act,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
            renormalize=config.moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.moe_router_activation_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            routed_input_transform=self.routed_expert_down_proj,
            routed_output_transform=self.routed_output_transform,
            is_sequence_parallel=use_sequence_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return final_hidden_states.view(num_tokens, hidden_size)


class AscendKimiMLAAttention(UpstreamKimiMLAAttention):
    """Extend vLLM's generic Kimi MLA only for DSpark RoPE metadata."""

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        use_output_gate: bool,
        use_rope: bool,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        non_causal_multi_token_decode: bool = False,
        disable_mlapo: bool = False,
    ) -> None:
        upstream_config = copy(config)
        upstream_config.mla_use_output_gate = use_output_gate
        super().__init__(
            config=upstream_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            use_nope=True,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        attention_layer = self._attention_layer
        if disable_mlapo:
            attention_layer.impl.enable_mlapo = False
        if not use_rope and not non_causal_multi_token_decode:
            return

        rotary_emb = None
        if use_rope:
            rope_parameters = dict(config.rope_parameters)
            if rope_parameters["rope_type"] != "default":
                rope_parameters["rope_type"] = (
                    "deepseek_yarn" if rope_parameters.get("apply_yarn_scaling", True) else "deepseek_llama_scaling"
                )
            rotary_emb = get_rope(
                qk_rope_head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=rope_parameters,
                is_neox_style=False,
            )
            if rope_parameters["rope_type"] == "deepseek_yarn":
                scaling_factor = float(rope_parameters["factor"])
                mscale_all_dim = float(rope_parameters.get("mscale_all_dim", 0.0))
                if scaling_factor > 1 and mscale_all_dim:
                    mscale = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                    self.scaling *= mscale * mscale

        # The upstream Kimi module has already constructed the platform-
        # registered MLA wrapper, including all projections and weight loaders.
        # Configure that existing Ascend attention layer for DSpark instead of
        # constructing and registering a second wrapper with the same prefix.
        attention_layer.scale = self.scaling
        attention_layer.non_causal_multi_token_decode = non_causal_multi_token_decode
        attention_layer.impl.scale = float(self.scaling)
        attention_layer.impl.rotary_emb = rotary_emb
        attention_layer.impl.use_mla_rope = use_rope

    @property
    def _attention_layer(self):
        return self.mla_attn.mla_attn

    @property
    def is_vl_first_layer(self) -> bool:
        return self.mla_attn.is_vl_first_layer

    @property
    def layer_name(self) -> str:
        return self._attention_layer.layer_name

    @property
    def impl(self):
        return self._attention_layer.impl

    @property
    def kv_cache(self):
        return self._attention_layer.kv_cache

    @property
    def kv_cache_dtype(self):
        return self._attention_layer.kv_cache_dtype

    @property
    def _k_scale(self):
        return self._attention_layer._k_scale

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)


class AscendKimiDecoderLayer(UpstreamKimiDecoderLayer):
    """Upstream Kimi decoder structure with Ascend attention backends."""

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_sequence_parallel: bool = False,
    ) -> None:
        """Select KDA or no-RoPE MLA and configure the layer residual path."""
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.is_moe = config.is_moe
        self.use_sequence_parallel = use_sequence_parallel
        layer_idx = self.layer_idx
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        if config.is_kda_layer(layer_idx):
            self.self_attn = AscendKimiK3DeltaAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            qk_nope_head_dim = config.qk_nope_head_dim
            qk_rope_head_dim = config.qk_rope_head_dim
            v_head_dim = config.v_head_dim
            kv_lora_rank = config.kv_lora_rank
            assert qk_nope_head_dim is not None
            assert qk_rope_head_dim is not None
            assert v_head_dim is not None
            assert kv_lora_rank is not None
            assert config.mla_use_nope is True
            self.self_attn = AscendKimiMLAAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                use_output_gate=bool(config.mla_use_output_gate),
                use_rope=False,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )

        self.is_moe_layer = (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        if self.is_moe_layer:
            self.block_sparse_moe = AscendKimiMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
                use_sequence_parallel=use_sequence_parallel,
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = AscendKimiMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                use_sequence_parallel=use_sequence_parallel,
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        attn_res_block_size = config.attn_res_block_size
        self.use_attn_residuals = attn_res_block_size is not None
        if attn_res_block_size is not None:
            self.attn_res_block_size = attn_res_block_size
            self.is_block_write_layer = layer_idx % attn_res_block_size == 0
            self.block_write_idx = layer_idx // attn_res_block_size
            self.prev_valid_blocks = cdiv(layer_idx, attn_res_block_size)
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.mlp_res_norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.self_attention_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.mlp_res_proj",
            )

        if self.use_sequence_parallel:
            self.self_attn.o_proj.reduce_results = False

    def _run_self_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Ascend attention returns its output instead of filling an AMD buffer.
        return self.self_attn(positions=positions, hidden_states=hidden_states)

    def _attention(self, positions: torch.Tensor, attention_input: torch.Tensor) -> torch.Tensor:
        """Run the attention delta for a pre-computed AttnRes blend."""
        hidden_states = self.input_layernorm(attention_input)
        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)
            hidden_states = hidden_states[: positions.shape[0]]
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
        )
        if self.use_sequence_parallel:
            hidden_states = sp_reduce_scatter(hidden_states)
        return hidden_states

    def _mlp(self, mlp_input: torch.Tensor) -> torch.Tensor:
        """Run the MLP/MoE delta for a pre-computed AttnRes blend."""
        return self.mlp(self.post_attention_layernorm(mlp_input))

    def forward_attn_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Kimi attention residuals with Ascend attention and MoE."""
        prefix_sum: torch.Tensor | None = hidden_states
        hidden_states = _apply_ascend_attn_res(
            prefix_sum,
            block_residual,
            self.self_attention_res_proj,
            self.self_attention_res_norm,
            self.prev_valid_blocks,
        )
        if self.is_block_write_layer:
            assert prefix_sum is not None
            block_residual[:, self.block_write_idx, :].copy_(prefix_sum)
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)
            hidden_states = hidden_states[: positions.shape[0]]
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
        )
        if self.use_sequence_parallel:
            hidden_states = sp_reduce_scatter(hidden_states)

        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        mlp_valid_blocks = self.prev_valid_blocks + (1 if self.is_block_write_layer else 0)
        hidden_states = _apply_ascend_attn_res(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            mlp_valid_blocks,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = prefix_sum + hidden_states
        return hidden_states, block_residual


class AscendKimiLinearModel(UpstreamKimiLinearModel):
    """Kimi text model assembled from the Ascend decoder layer."""

    packed_modules_mapping = {
        name: list(shards) for name, shards in UpstreamPackedKimiLinearModel.packed_modules_mapping.items()
    }
    packed_modules_mapping["fused_bfg_proj"] = [
        "b_proj",
        "f_a_proj",
        "g_proj",
    ]
    # Legacy Qwen3 GQA DSpark checkpoints consume the materialized input
    # to each selected Kimi layer. MLA DSpark checkpoints consume the raw
    # prefix-sum stream used by upstream vLLM, so keep that as the default.
    dspark_aux_capture_materialized = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size
        parallel_config = vllm_config.parallel_config
        # vLLM's generic MoE SP switch currently requires DP > 1. K3 also
        # needs the same rank-local token layout for the TP/EP, DP=1 topology
        # that FlashComm used before the standard SP operators were available.
        self.use_sequence_parallel = (
            parallel_config.pipeline_parallel_size == 1
            and parallel_config.enable_expert_parallel
            and parallel_config.tensor_parallel_size > 1
        )

        attn_res_mode = getattr(config, "attn_res_mode", "fused")
        if attn_res_mode == "original":
            self.attn_res_mode = "original"
        elif attn_res_mode == "fused":
            if _use_fused_attn_res():
                self.attn_res_mode = "fused"
            else:
                logger.warning_once(
                    "Kimi K3 'fused' AttnRes backend requested but unavailable; falling back to 'two_phase'"
                )
                self.attn_res_mode = "two_phase"
        else:
            self.attn_res_mode = "two_phase"
        logger.info("Kimi K3 AttnRes mode: %s", self.attn_res_mode)
        self.attn_res_effective_queries: torch.Tensor | None = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            return AscendKimiDecoderLayer(
                config,
                vllm_config,
                prefix,
                use_sequence_parallel=self.use_sequence_parallel,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.attn_res_block_size is not None:
                self.output_attn_res_norm = RMSNorm(
                    config.hidden_size,
                    eps=config.rms_norm_eps,
                )
                self.output_attn_res_proj = ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.output_attn_res_proj",
                )
        else:
            self.norm = PPMissingLayer()
            if config.attn_res_block_size is not None:
                self.output_attn_res_norm = PPMissingLayer()
                self.output_attn_res_proj = PPMissingLayer()

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, "num_attention_heads must be divisible by world_size"

    def load_weights(self, weights):
        """Route mixed-precision KDA gates through vLLM's packed loader."""
        params_dict = dict(self.named_parameters())
        gate_mapping = (
            (".b_proj.weight", ".fused_bfg_proj.weight", 0),
            (".f_a_proj.weight", ".fused_bfg_proj.f_a_weight", None),
            (".f_b_proj.weight", ".fused_bfg_proj.f_b_weight", None),
            (".g_proj.weight", ".fused_bfg_proj.weight", 2),
        )

        def remap_mixed_gate_weights():
            for args in weights:
                name, loaded_weight = args[:2]
                for source, target, shard_id in gate_mapping:
                    if source not in name:
                        continue
                    mapped_name = name.replace(source, target)
                    if mapped_name in params_dict:
                        kwargs = dict(args[2]) if len(args) > 2 else {}
                        kwargs["loaded_shard_id"] = shard_id
                        yield mapped_name, loaded_weight, kwargs
                        break
                else:
                    yield args

        result = super().load_weights(remap_mixed_gate_weights())
        if self.config.attn_res_block_size is not None and self.attn_res_mode != "original":
            self._prepare_attn_res_effective_queries()
        return result

    def _prepare_attn_res_effective_queries(self) -> None:
        """Precompute q * RMSNorm gain once after checkpoint loading."""
        layers = self.layers[self.start_layer : self.end_layer]
        if not layers:
            return
        first_weight = layers[0].self_attention_res_norm.weight
        effective_queries = torch.empty(
            2 * len(layers),
            self.config.hidden_size,
            dtype=torch.float32,
            device=first_weight.device,
        )
        for local_idx, layer in enumerate(layers):
            effective_queries[2 * local_idx].copy_(
                (
                    layer.self_attention_res_norm.weight.float()
                    * layer.self_attention_res_proj.weight.squeeze(0).float()
                ).detach()
            )
            effective_queries[2 * local_idx + 1].copy_(
                (layer.mlp_res_norm.weight.float() * layer.mlp_res_proj.weight.squeeze(0).float()).detach()
            )
        self.attn_res_effective_queries = effective_queries

    def _run_attn_res_phase1(
        self,
        block_residual: torch.Tensor,
        block_queries: torch.Tensor,
        epsilon: float,
    ) -> AttnResPhase1Stats:
        """Run the Phase-1 (historical block statistics) backend.

        ``fused`` dispatches to the CANNBot DSL kernel when available and
        falls back to the pure-Torch implementation otherwise. The pure Torch
        path treats every row of ``block_residual`` as a valid candidate,
        while the DSL kernel needs an explicit ``valid_blocks`` count for its
        static UB buffers.
        """
        if self.attn_res_mode == "fused" and _block_attn_res_prepare_fused is not None and block_residual.shape[1] > 0:
            # The CANNBot DSL kernel keeps V resident in FP32; the plugin's
            # block_residual is bf16 because it also travels through the PP
            # IntermediateTensors channel.
            v_fp32 = block_residual.float().contiguous()
            valid_blocks = torch.tensor([v_fp32.shape[1]], dtype=torch.uint64, device=block_residual.device)
            inter_numerator, inter_max, inter_exp_sum = _block_attn_res_prepare_fused(
                v_fp32,
                valid_blocks,
                block_queries,
                eps=epsilon,
            )
            return AttnResPhase1Stats(
                inter_numerator=inter_numerator,
                inter_max=inter_max,
                inter_exp_sum=inter_exp_sum,
            )
        return _prepare_attn_res_phase1(block_residual, block_queries, epsilon)

    def _run_attn_res_phase2(
        self,
        partial_block: torch.Tensor,
        partial_delta: torch.Tensor,
        slot: AttnResPhase2Slot,
        epsilon: float,
    ) -> torch.Tensor:
        """Fold the running ``partial_block`` delta into one slot.

        ``fused`` dispatches to the CANNBot DSL kernel which updates the
        partial in place and returns the merged output plus the updated
        partial buffer. The pure Torch path updates the partial in place
        and returns the merged slot output.
        """
        if self.attn_res_mode == "fused" and _block_attn_res_update_fused is not None:
            merged_output = _block_attn_res_update_fused(
                partial_block,
                partial_delta,
                slot.effective_query,
                slot.inter_numerator,
                slot.inter_max,
                slot.inter_exp_sum
            )
            return merged_output
        return _update_attn_res_phase2(partial_block, partial_delta, slot, epsilon)

    def _forward_attn_res_blocks(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_valid_blocks: int,
        positions: torch.Tensor,
        aux_hidden_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the two-phase AttnRes over the rank-local AttnRes blocks.

        A rank may start mid-block under pipeline parallelism; finish that
        in-flight block first (continuation), then process the remainder on
        block boundaries.
        """
        block_size = self.config.attn_res_block_size
        end_layer = self.end_layer

        next_layer = self.start_layer
        if next_layer % block_size != 0 and next_layer < end_layer:
            piece_end = min(((next_layer // block_size) + 1) * block_size, end_layer)
            hidden_states, num_valid_blocks = self._forward_attn_res_block(
                next_layer,
                piece_end,
                starts_in_rank=False,
                hidden_states=hidden_states,
                block_residual=block_residual,
                num_valid_blocks=num_valid_blocks,
                positions=positions,
                aux_hidden_states=aux_hidden_states,
            )
            next_layer = piece_end
        while next_layer < end_layer:
            block_end = min(next_layer + block_size, end_layer)
            hidden_states, num_valid_blocks = self._forward_attn_res_block(
                next_layer,
                block_end,
                starts_in_rank=True,
                hidden_states=hidden_states,
                block_residual=block_residual,
                num_valid_blocks=num_valid_blocks,
                positions=positions,
                aux_hidden_states=aux_hidden_states,
            )
            next_layer = block_end
        return hidden_states, aux_hidden_states

    def _forward_attn_res_block(
        self,
        start_layer_idx: int,
        end_layer_idx: int,
        starts_in_rank: bool,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_valid_blocks: int,
        positions: torch.Tensor,
        aux_hidden_states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        """Process one AttnRes block with the two-phase backend.

        Phase 1 computes the FP32 inter-block softmax statistics once for all
        slots of the block. Phase 2 folds the running partial block into each
        slot through an Online Softmax merge. ``starts_in_rank`` selects
        between a fresh block (the current partial becomes the newest block
        start) and an in-flight continuation whose partial is carried in
        ``hidden_states``.
        """
        layers = self.layers[start_layer_idx:end_layer_idx]
        if len(layers) == 0:
            return hidden_states, num_valid_blocks
        if self.attn_res_effective_queries is None or self.attn_res_effective_queries.device != hidden_states.device:
            self._prepare_attn_res_effective_queries()
        if self.attn_res_effective_queries is None:
            raise RuntimeError("Kimi K3 AttnRes effective queries are not initialized")
        if starts_in_rank:
            block_residual[:, num_valid_blocks, :].copy_(hidden_states)
            num_valid_blocks += 1
            partial_block = torch.zeros_like(
                hidden_states, dtype=torch.float32 if self.attn_res_mode == "fused" else hidden_states.dtype
            )
        else:
            partial_block = hidden_states.clone().to(
                torch.float32 if self.attn_res_mode == "fused" else hidden_states.dtype
            )
        block_queries = self.attn_res_effective_queries[
            2 * (start_layer_idx - self.start_layer) : 2 * (end_layer_idx - self.start_layer)
        ].contiguous()
        epsilon = self.config.rms_norm_eps
        phase1 = self._run_attn_res_phase1(block_residual[:, :num_valid_blocks], block_queries, epsilon)

        previous_mlp_delta = None
        for layer_offset, layer in enumerate(layers):
            attention_slot = 2 * layer_offset
            mlp_slot = attention_slot + 1
            layer_idx = start_layer_idx + layer_offset
            if previous_mlp_delta is None:
                if starts_in_rank:
                    attention_input = (
                        phase1.inter_numerator[attention_slot] / phase1.inter_exp_sum[attention_slot].unsqueeze(-1)
                    ).to(hidden_states.dtype)
                else:
                    attention_input = _merge_attn_res_partial(
                        partial_block,
                        AttnResPhase2Slot(
                            effective_query=block_queries[attention_slot],
                            inter_numerator=phase1.inter_numerator[attention_slot],
                            inter_max=phase1.inter_max[attention_slot],
                            inter_exp_sum=phase1.inter_exp_sum[attention_slot],
                        ),
                        epsilon,
                    )
            else:
                attention_slot_stats = AttnResPhase2Slot(
                    effective_query=block_queries[attention_slot],
                    inter_numerator=phase1.inter_numerator[attention_slot],
                    inter_max=phase1.inter_max[attention_slot],
                    inter_exp_sum=phase1.inter_exp_sum[attention_slot],
                )
                attention_input = self._run_attn_res_phase2(
                    partial_block,
                    previous_mlp_delta.contiguous(),
                    attention_slot_stats,
                    epsilon,
                ).to(hidden_states.dtype)
            if self.dspark_aux_capture_materialized and layer_idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(attention_input)
            attention_output = layer._attention(positions, attention_input)
            mlp_slot_stats = AttnResPhase2Slot(
                effective_query=block_queries[mlp_slot],
                inter_numerator=phase1.inter_numerator[mlp_slot],
                inter_max=phase1.inter_max[mlp_slot],
                inter_exp_sum=phase1.inter_exp_sum[mlp_slot],
            )
            mlp_input = self._run_attn_res_phase2(
                partial_block,
                attention_output.contiguous(),
                mlp_slot_stats,
                epsilon,
            ).to(hidden_states.dtype)
            previous_mlp_delta = layer._mlp(mlp_input)
            if not self.dspark_aux_capture_materialized and (layer_idx + 1) in self.aux_hidden_state_layers:
                self._maybe_add_hidden_state(
                    aux_hidden_states,
                    layer_idx + 1,
                    (partial_block + previous_mlp_delta).to(hidden_states.dtype),
                    None,
                )

        if previous_mlp_delta is not None:
            partial_block.add_(previous_mlp_delta)
        return partial_block.to(hidden_states.dtype), num_valid_blocks

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if self.config.attn_res_block_size is None:
            return super().forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding,
                    hidden_states,
                )
            hidden_states = sp_shard(hidden_states)
            assert residual is None, "Sequence parallelism is not supported with pipeline parallelism"

        if self.dspark_aux_capture_materialized:
            aux_hidden_states: list[torch.Tensor] = []
        else:
            aux_hidden_states = self._maybe_add_hidden_state(
                [],
                self.start_layer,
                hidden_states,
                residual,
            )
        attn_res_block_num = cdiv(
            self.end_layer,
            self.config.attn_res_block_size,
        )
        block_residual = hidden_states.new_empty(
            hidden_states.size(0),
            attn_res_block_num,
            hidden_states.size(1),
        )
        if residual is not None:
            block_residual[:, : residual.size(1), :].copy_(residual)
        num_valid_blocks = residual.size(1) if residual is not None else 0
        residual = block_residual

        if self.attn_res_mode != "original":
            hidden_states, aux_hidden_states = self._forward_attn_res_blocks(
                hidden_states,
                block_residual,
                num_valid_blocks,
                positions,
                aux_hidden_states,
            )
        else:
            for layer_idx, layer in enumerate(
                self.layers[self.start_layer : self.end_layer],
                start=self.start_layer,
            ):
                if self.dspark_aux_capture_materialized and layer_idx in self.aux_hidden_state_layers:
                    aux_hidden_states.append(
                        _apply_ascend_attn_res(
                            hidden_states,
                            residual,
                            layer.self_attention_res_proj,
                            layer.self_attention_res_norm,
                            layer.prev_valid_blocks,
                        )
                    )
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )
                if not self.dspark_aux_capture_materialized and (layer_idx + 1) in self.aux_hidden_state_layers:
                    self._maybe_add_hidden_state(
                        aux_hidden_states,
                        layer_idx + 1,
                        hidden_states,
                        residual,
                    )

        if not get_pp_group().is_last_rank:
            assert not self.use_sequence_parallel, "Sequence parallelism is not supported with pipeline parallelism"
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        hidden_states = _apply_ascend_attn_res(
            hidden_states,
            residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            attn_res_block_num,
        )
        if self.use_sequence_parallel:
            if aux_hidden_states:
                hidden_size = hidden_states.shape[-1]
                packed_hidden_states = torch.cat(
                    [hidden_states, *aux_hidden_states],
                    dim=-1,
                )
                packed_hidden_states = sp_all_gather(packed_hidden_states)
                packed_hidden_states = packed_hidden_states[:full_num_tokens]
                hidden_states, *aux_hidden_states = packed_hidden_states.split(
                    hidden_size,
                    dim=-1,
                )
            else:
                hidden_states = sp_all_gather(hidden_states)
                hidden_states = hidden_states[:full_num_tokens]
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class AscendKimiLinearForCausalLM(UpstreamKimiLinearForCausalLM):
    """Causal-LM wrapper retaining vLLM 0.27 state/cache interfaces."""

    packed_modules_mapping = AscendKimiLinearModel.packed_modules_mapping

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = AscendKimiLinearModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            scale=getattr(self.config, "logit_scale", 1.0),
        )

    def set_dspark_aux_capture_materialized(self, enabled: bool) -> None:
        self.model.dspark_aux_capture_materialized = enabled


class AscendKimiK3MultiModalProjector(KimiK25MultiModalProjector):
    """Kimi projector with the optional ModelSlim output rotation."""

    def __init__(
        self,
        config,
        *args,
        prefix: str = "",
        enable_rotation: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(config, *args, prefix=prefix, **kwargs)
        self.rot_proj: ReplicatedLinear | None = None
        if enable_rotation:
            output_size = config.text_hidden_size
            self.rot_proj = ReplicatedLinear(
                output_size,
                output_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.rot_proj",
            )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = super().forward(image_features)
        rot_proj = self.rot_proj
        if rot_proj is not None:
            hidden_states = rot_proj(hidden_states)[0]
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    KimiK3MultiModalProcessor,
    info=KimiK3ProcessingInfo,
    dummy_inputs=KimiK3DummyInputsBuilder,
)
class AscendKimiK3ForConditionalGeneration(UpstreamKimiK3ForConditionalGeneration):
    """Upstream Kimi K3 multimodal wrapper with Ascend text/projector layers."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        model_config = vllm_config.model_config
        self.config = model_config.hf_config
        self.quant_config = vllm_config.quant_config
        multimodal_config = model_config.multimodal_config
        assert multimodal_config is not None

        self.use_data_parallel = is_vit_use_data_parallel(
            self.config.vision_config.num_attention_heads,
        )
        self.hidden_size = self.config.text_config.hidden_size
        self.device = current_platform.current_device()
        vision_quant_config = self._maybe_ignore_quant_config(self.quant_config)

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = MoonViT3dPretrainedModel(
                self.config.vision_config,
                quant_config=vision_quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            if vision_quant_config is not None:
                self.vision_tower = self.vision_tower.to(device=self.device)
            else:
                self.vision_tower = self.vision_tower.to(
                    device=self.device,
                    dtype=model_config.dtype,
                )

            self.mm_projector = AscendKimiK3MultiModalProjector(
                self.config.vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=vision_quant_config,
                prefix=maybe_prefix(prefix, "mm_projector"),
                enable_rotation=get_rotation_path(vllm_config) is not None,
            )
        if vision_quant_config is not None:
            self.mm_projector = self.mm_projector.to(device=self.device)
        else:
            self.mm_projector = self.mm_projector.to(
                device=self.device,
                dtype=model_config.dtype,
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["KimiLinearForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder = self.config.media_placeholder_token_id

    def set_dspark_aux_capture_materialized(self, enabled: bool) -> None:
        self.language_model.set_dspark_aux_capture_materialized(enabled)
