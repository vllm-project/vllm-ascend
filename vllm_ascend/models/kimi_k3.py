# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 model adapters for vLLM 0.27 on Ascend.

vLLM owns Kimi's configuration, multimodal processor, weight mappings, and
model-level forward contract.  This module composes those upstream pieces with
the generic MLA/MoE implementation and the Ascend KDA backend.
"""

import math
from bisect import bisect_left
from collections.abc import Iterable
from copy import copy
from typing import Any

import torch
import vllm.envs as envs
from torch import nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import model_parallel_is_initialized
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
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
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    get_spec_layer_idx_from_weight_name,
    init_vllm_registered_model,
    is_pp_missing_parameter,
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
from vllm.utils.math_utils import cdiv

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.utils import mark_fused_preprocess_weights
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention  # type: ignore[import-untyped]
from vllm_ascend.ops.linear_op import KimiOProjMMReduceScatterOp
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod
from vllm_ascend.utils import get_rotation_path, is_950
from vllm_ascend.worker.v2.pp_transport import (
    PPTransportDataType,
    add_pp_transport_tensors,
    get_pp_transport_tensors,
)
from vllm_ascend.worker.v2.pp_transport import (
    make_empty_intermediate_tensors as make_pp_empty_intermediate_tensors,
)


def _use_attn_res_prefill_cache() -> bool:
    """Enable weight caching only for explicit pure-prefill A5 requests."""
    if not is_forward_context_available():
        return False
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return False
    for layer_metadata in metadata.values():
        prefills = getattr(layer_metadata, "num_prefills", None)
        decodes = getattr(layer_metadata, "num_decodes", None)
        if type(prefills) is not int or type(decodes) is not int or prefills <= 0 or decodes != 0:
            return False
        if getattr(layer_metadata, "spec_sequence_masks", None) is not None:
            return False
    return is_950()


class AscendKimiRoutedOutputTransform(KimiRoutedOutputTransform):
    """Fuse the latent RMSNorm and MXFP8 activation quantization on A5."""

    def __init__(self, norm: RMSNorm | None, up_proj: ReplicatedLinear) -> None:
        super().__init__(norm, up_proj)
        self._supports_mx_norm_fusion = get_current_hardware_profile().supports(
            HardwareCapability.DYNAMIC_MX_QUANT_FUSION
        ) and hasattr(torch.ops.npu, "npu_rms_norm_dynamic_mx_quant")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scheme = getattr(getattr(self.up_proj, "quant_method", None), "quant_method", None)
        if (
            self.norm is not None
            and self._supports_mx_norm_fusion
            and isinstance(scheme, AscendW8A8MXFP8DynamicLinearMethod)
            and scheme.group_size == 32
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.shape[-1] % 64 == 0
            and getattr(self.norm, "bias", None) is None
        ):
            quantized, scale, _ = torch.ops.npu.npu_rms_norm_dynamic_mx_quant(
                hidden_states,
                self.norm.weight,
                epsilon=self.norm.variance_epsilon,
                scale_alg=scheme.dynamic_mx_quant_scale_alg,
                dst_type=torch.float8_e4m3fn,
            )
            hidden_states, _ = self.up_proj((quantized, scale))
            return hidden_states
        return super().forward(hidden_states)


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


def is_moe_layer_idx(config, layer_idx: int) -> bool:
    """Whether the Kimi layer at ``layer_idx`` hosts routed experts."""
    return bool(
        config.is_moe
        and config.num_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )


class AscendKimiMoE(nn.Module):
    """Kimi K3 MoE assembled from the standard vLLM MoE interfaces."""

    def __init__(
        self,
        *,
        config,
        vllm_config: VllmConfig,
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

        # Load balancing settings.
        parallel_config = vllm_config.parallel_config
        self.enable_eplb = parallel_config.enable_eplb
        eplb_config = parallel_config.eplb_config
        self.n_routed_experts: int = num_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = get_ep_group().device_group.size()
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

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
        self.routed_output_transform: KimiRoutedOutputTransform | None
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
            self.routed_output_transform = AscendKimiRoutedOutputTransform(
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
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
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
            mark_fused_preprocess_weights(attention_layer.impl)
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

        self.is_moe_layer = is_moe_layer_idx(config, layer_idx)
        if self.is_moe_layer:
            self.block_sparse_moe = AscendKimiMoE(
                config=config,
                vllm_config=vllm_config,
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
                reduce_results=not use_sequence_parallel,
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

        self.fuse_o_proj_mm_reduce_scatter = self._enable_o_proj_mm_reduce_scatter(vllm_config)

    def _enable_o_proj_mm_reduce_scatter(self, vllm_config: VllmConfig) -> bool:
        if not self.use_sequence_parallel or not self.use_attn_residuals:
            return False
        # Fusion uses each PP stage's existing TP group and returns the same
        # SP token shard as the separate projection and ReduceScatter.
        if not get_current_hardware_profile().supports(HardwareCapability.MM_REDUCE_SCATTER_AI_CPU_INFERENCE):
            return False
        if get_ascend_config().weight_nz_mode == 2:
            return False
        if vllm_config.lora_config is not None:
            return False
        o_proj = self.self_attn.o_proj
        if KimiOProjMMReduceScatterOp.unsupported_reason(o_proj) is not None:
            return False
        o_proj.custom_op = KimiOProjMMReduceScatterOp(o_proj)
        if isinstance(self.self_attn, AscendKimiMLAAttention):
            # The MLA custom op writes into a caller-owned output buffer. Its
            # token dimension must match the fused projection's TP shard.
            self.self_attn.mla_attn.output_token_shard_size = o_proj.tp_size
        return True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        prepared_attn_input: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        defer_mlp_add: bool = False,
        optimize_prefill: bool = False,
        **kwargs,
    ):
        if self.use_attn_residuals:
            assert residual is not None
            return self.forward_attn_residual(
                positions,
                hidden_states,
                residual,
                prepared_attn_input=prepared_attn_input,
                defer_mlp_add=defer_mlp_add,
                optimize_prefill=optimize_prefill,
            )
        return super().forward(positions, hidden_states, residual, **kwargs)

    def _run_self_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Ascend attention returns its output instead of filling an AMD buffer.
        return self.self_attn(positions=positions, hidden_states=hidden_states)

    def prepare_attn_residual(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        addend: torch.Tensor | None = None,
        return_materialized: bool = False,
        optimize_prefill: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The previous MLP add is rounded to BF16 before it becomes either a
        # DSpark raw prefix or an AttnRes input. A block boundary stores that
        # same prefix, while the mixture reads only the older valid slots.
        return torch.ops._C_ascend.attn_res_fwd(
            prefix_sum,
            addend,
            block_residual,
            self.self_attention_res_proj.weight,
            self.self_attention_res_norm.weight,
            self.self_attention_res_norm.variance_epsilon,
            self.prev_valid_blocks,
            self.input_layernorm.weight,
            self.input_layernorm.variance_epsilon,
            self.block_write_idx if self.is_block_write_layer else -1,
            return_materialized,
            optimize_prefill=optimize_prefill,
        )

    def forward_attn_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        prepared_attn_input: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        defer_mlp_add: bool = False,
        optimize_prefill: bool = False,
    ):
        """Each residual point runs one native add/AttnRes/RMSNorm kernel."""
        if prepared_attn_input is None:
            prepared_attn_input = self.prepare_attn_residual(
                hidden_states, block_residual, optimize_prefill=optimize_prefill
            )
        hidden_states, prefix_sum, _ = prepared_attn_input
        if self.is_block_write_layer:
            prefix_sum = None

        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)
            hidden_states = hidden_states[: positions.shape[0]]
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
        )
        if self.use_sequence_parallel and not self.fuse_o_proj_mm_reduce_scatter:
            hidden_states = sp_reduce_scatter(hidden_states)

        mlp_valid_blocks = self.prev_valid_blocks + int(self.is_block_write_layer)
        op = torch.ops._C_ascend.attn_res_fwd
        hidden_states, prefix_sum, _ = op(
            hidden_states if prefix_sum is None else prefix_sum,
            None if prefix_sum is None else hidden_states,
            block_residual,
            self.mlp_res_proj.weight,
            self.mlp_res_norm.weight,
            self.mlp_res_norm.variance_epsilon,
            mlp_valid_blocks,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
            optimize_prefill=optimize_prefill,
        )
        mlp_output = self.mlp(hidden_states)
        if defer_mlp_add:
            # Only the enclosing model uses this contract. It materializes the
            # prefix in the next fused residual point before any aux capture,
            # or at the PP/final-output boundary before publishing a tensor.
            return prefix_sum, block_residual, mlp_output
        # Standalone decoder calls have no next residual point to absorb this.
        hidden_states, _, _ = op(
            prefix_sum,
            mlp_output,
            block_residual,
            self.mlp_res_proj.weight,
            self.mlp_res_norm.weight,
            self.mlp_res_norm.variance_epsilon,
            0,
            mix=False,
            optimize_prefill=optimize_prefill,
        )
        return hidden_states, block_residual


class AscendKimiLinearModel(UpstreamKimiLinearModel):
    """Kimi text model assembled from the Ascend decoder layer."""

    # The Ascend forward path carries cumulative auxiliary states across PP
    # via IntermediateTensors (see pp_transport.PPTransportDataType).
    supports_aux_hidden_states_over_pp = True
    # Upstream's reserve_aux_intermediate_tensor_slots / relay_aux_hidden_states
    # read this via getattr to name and forward the receive-buffer aux slots;
    # it must equal pp_transport's transport key prefix so both sides agree.
    AUX_HIDDEN_STATE_KEY = "pp_transport_aux_hidden_states_"

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
        self.vllm_config = vllm_config
        self.vocab_size = config.vocab_size
        # Only the last PP stage loads the drafter. Select its aux contract
        # on every stage before upstream reserves receive buffers/relay slots.
        spec_config = vllm_config.speculative_config
        if spec_config is not None and spec_config.method == "dspark":
            draft_config = spec_config.draft_model_config.hf_config
            self.dspark_aux_capture_materialized = draft_config.model_type == "qwen3" and any(
                arch in ("DSparkDraftModel", "Qwen3DSparkModel") for arch in (draft_config.architectures or ())
            )
        # vLLM's generic MoE SP switch currently requires DP > 1. K3 also
        # needs the same rank-local token layout for the TP/EP, DP=1 topology
        # that FlashComm used before the standard SP operators were available.
        parallel_config = vllm_config.parallel_config
        # Physical expert slots appended by EPLB; load_weights needs the count
        # so the duplicated initial expert placements receive weights.
        self.n_redundant_experts = (
            parallel_config.eplb_config.num_redundant_experts if parallel_config.enable_eplb else 0
        )
        self.use_sequence_parallel = (
            (parallel_config.pipeline_parallel_size == 1 or vllm_config.use_v2_model_runner)
            and parallel_config.enable_expert_parallel
            and parallel_config.tensor_parallel_size > 1
        )

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

    def _cache_aux_pp_layout(self) -> None:
        super()._cache_aux_pp_layout()
        if (
            self.config.attn_res_block_size is None
            or not self.dspark_aux_capture_materialized
            or not model_parallel_is_initialized()
        ):
            return
        pp = get_pp_group()
        if not pp.is_first_rank:
            # A materialized state at start_layer belongs to this stage,
            # unlike the raw state emitted by the preceding stage.
            self._aux_slot_base_cached = bisect_left(self.aux_hidden_state_layers, self.start_layer)
        if pp.is_last_rank:
            self._aux_upstream_total_cached = self._aux_slot_base_cached

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        if self.config.attn_res_block_size is None:
            return super().make_empty_intermediate_tensors(batch_size, dtype, device)
        # Materialized DSpark states are captured before a layer; raw states
        # are captured after the preceding layer. Handle a PP cut at either.
        return make_pp_empty_intermediate_tensors(
            self,
            super().make_empty_intermediate_tensors,
            include_start_layer=not self.dspark_aux_capture_materialized,
        )(batch_size, dtype, device)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor] | tuple[str, torch.Tensor, dict[str, Any]]],
    ) -> set[str]:
        """Load Kimi weights through vLLM's packed KDA-gate loader.

        Mirrors upstream ``KimiLinearModel.load_weights`` with one addition:
        ``n_redundant_experts`` is propagated into the expert mapping so the
        duplicated initial expert placements receive weights when EPLB is
        enabled.
        """
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

        kda_config = self.config.linear_attn_config
        use_full_rank_gate = bool(kda_config and kda_config.get("use_full_rank_gate", False))
        beta_shard_id = 5 if use_full_rank_gate else 3
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".in_proj_qkvgfab", ".q_proj", 0),
            (".in_proj_qkvgfab", ".k_proj", 1),
            (".in_proj_qkvgfab", ".v_proj", 2),
            (".in_proj_qkvgfab", ".b_proj", beta_shard_id),
            (".in_proj_qkvgfab", ".f_a_proj", 4),
            (".conv1d", ".q_conv1d", 0),
            (".conv1d", ".k_conv1d", 1),
            (".conv1d", ".v_conv1d", 2),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        if use_full_rank_gate:
            stacked_params_mapping.append((".in_proj_qkvgfab", ".g_proj", 3))
        if getattr(self.config, "q_lora_rank", None) is not None:
            stacked_params_mapping += [
                (".fused_qkv_a_proj", ".q_a_proj", 0),
                (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            ]
        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_experts,
                num_redundant_experts=self.n_redundant_experts,
            )
        else:
            expert_params_mapping = []
        # Under the MXFP4 quant interface the routed experts register unpacked
        # params (``w13_weight``), while the compressed-tensors checkpoint names
        # them ``.weight_packed``. Rebind so the expert mapping resolves; scales
        # already share the ``.weight_scale`` suffix.
        experts_unpacked = not any(n.endswith("w13_weight_packed") for n in params_dict)
        loaded_params: set[str] = set()
        for args in remap_mixed_gate_weights():
            name, loaded_weight = args[0], args[1]
            kwargs: dict[str, Any] = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue
            if experts_unpacked and name.endswith(".weight_packed"):
                name = name.replace(".weight_packed", ".weight")

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
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
                name_mapped = name.replace(weight_name, param_name)
                # Packed projections are only present on compatible layers.
                if name_mapped not in params_dict:
                    continue
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
                for (
                    expert_param_name,
                    expert_weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if expert_weight_name not in name:
                        continue
                    name = name.replace(expert_weight_name, expert_param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict and not self.config.is_linear_attn:  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        return loaded_params

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

        full_num_tokens = positions.shape[0]
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        materialized_aux = self.dspark_aux_capture_materialized
        aux_hidden_states = get_pp_transport_tensors(intermediate_tensors, PPTransportDataType.AUX_HIDDEN_STATES)
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                # Every stage starts with a full-token padding mask, even
                # though only the first stage starts with full activations.
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding,
                    positions,
                )
            # Every stage starts from full-sequence tensors - the previous
            # stage closed its sequence-parallel region at the boundary -
            # and keeps its own shard afterwards, matching DeepSeek V4.
            hidden_states = sp_shard(hidden_states)
            if residual is not None:
                residual = sp_shard(residual)
            aux_hidden_states = [sp_shard(aux) for aux in aux_hidden_states]
        if not materialized_aux and get_pp_group().is_first_rank:
            aux_hidden_states = self._maybe_add_hidden_state(
                aux_hidden_states,
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
        residual = block_residual

        optimize_attn_res_prefill = _use_attn_res_prefill_cache()
        pending_mlp_output = None
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            prepared = layer.prepare_attn_residual(
                hidden_states,
                residual,
                pending_mlp_output,
                return_materialized=materialized_aux and layer_idx in self.aux_hidden_state_layers,
                optimize_prefill=optimize_attn_res_prefill,
            )
            if layer_idx > self.start_layer and not materialized_aux:
                self._maybe_add_hidden_state(aux_hidden_states, layer_idx, prepared[1], None)
            if materialized_aux and layer_idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(prepared[2])
            hidden_states, residual, pending_mlp_output = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                prepared_attn_input=prepared,
                defer_mlp_add=True,
                optimize_prefill=optimize_attn_res_prefill,
            )

        if not get_pp_group().is_last_rank and pending_mlp_output is not None:
            hidden_states = hidden_states + pending_mlp_output
            if not materialized_aux:
                # Publish the completed raw prefix at the PP boundary once.
                self._maybe_add_hidden_state(aux_hidden_states, self.end_layer, hidden_states, None)
        if not get_pp_group().is_last_rank:
            if self.use_sequence_parallel:
                # The next PP rank expects full-sequence tensors; close the
                # sequence-parallel region before crossing the boundary so the
                # upstream PP transport only sees replicated tensors.
                hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
                residual = sp_all_gather(residual)[:full_num_tokens]
                aux_hidden_states = [sp_all_gather(aux)[:full_num_tokens] for aux in aux_hidden_states]
            return add_pp_transport_tensors(
                IntermediateTensors({"hidden_states": hidden_states, "residual": residual}),
                PPTransportDataType.AUX_HIDDEN_STATES,
                aux_hidden_states,
            )

        hidden_states, final_prefix, _ = torch.ops._C_ascend.attn_res_fwd(
            hidden_states,
            pending_mlp_output,
            residual,
            self.output_attn_res_proj.weight,
            self.output_attn_res_norm.weight,
            self.output_attn_res_norm.variance_epsilon,
            attn_res_block_num,
            optimize_prefill=optimize_attn_res_prefill,
        )
        if not materialized_aux and pending_mlp_output is not None:
            self._maybe_add_hidden_state(aux_hidden_states, self.end_layer, final_prefix, None)
        if materialized_aux and self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states)
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


class KimiMixtureOfExperts(MixtureOfExperts):
    """MoE bookkeeping shared by the Ascend Kimi K3 adapters."""

    moe_mlp_layers: list[AscendKimiMoE]

    def extract_moe_parameters(self, example_moe: AscendKimiMoE | None) -> None:
        if example_moe is None:
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
            self.num_shared_experts = example_moe.num_shared_experts or 0
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
            moe.n_physical_experts = num_physical_experts
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class AscendKimiLinearForCausalLM(UpstreamKimiLinearForCausalLM, KimiMixtureOfExperts):
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
        # Set MoE hyperparameters for the EPLB registration.
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "num_expert_group", None) or 1
        # Global count: EPLB state is sized over every MoE layer of the model,
        # while ``moe_layers`` below only holds this PP rank's layers.
        self.num_moe_layers = sum(
            1 for layer_idx in range(self.config.num_hidden_layers) if is_moe_layer_idx(self.config, layer_idx)
        )
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, AscendKimiDecoderLayer)
            if isinstance(layer.mlp, AscendKimiMoE):
                # Pick the last one since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

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

    def get_language_model(self) -> AscendKimiLinearForCausalLM:
        return self.language_model

    def set_dspark_aux_capture_materialized(self, enabled: bool) -> None:
        self.language_model.set_dspark_aux_capture_materialized(enabled)
