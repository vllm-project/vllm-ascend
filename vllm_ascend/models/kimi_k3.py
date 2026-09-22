# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 model adapters for vLLM 0.27 on Ascend.

vLLM owns Kimi's configuration, multimodal processor, weight mappings, and
model-level forward contract.  This module composes those upstream pieces with
the generic MLA/MoE implementation and the Ascend KDA backend.
"""

import math
from copy import copy

import torch
import vllm.envs as envs
from torch import nn
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    DCPGroupColumnParallelLinear,
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
from vllm.utils.math_utils import cdiv

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention  # type: ignore[import-untyped]
from vllm_ascend.ops.linear_op import KimiOProjMMReduceScatterOp
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod
from vllm_ascend.utils import enable_kimi_k3_sp, get_rotation_path, is_950
from vllm_ascend.worker.v2.pp_utils import (
    PPTransportDataType,
    add_pp_transport_tensors,
    get_pp_transport_tensors,
)
from vllm_ascend.worker.v2.pp_utils import (
    make_empty_intermediate_tensors as make_pp_empty_intermediate_tensors,
)


def _use_attn_res_prefill_kernel() -> bool:
    """Select the prefill-only kernel from explicit CPU request counts."""
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
    return is_950() and hasattr(torch.ops._C_ascend.attn_res_fwd, "fused_prefill")


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
            # The prequantized linear input contract returns BF16.
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 2
            and hidden_states.shape[-1] % 64 == 0
            and getattr(self.norm, "bias", None) is None
            and getattr(self.norm, "variance_size_override", None) is None
            and getattr(self.up_proj, "custom_op", None) is None
            and not any(getattr(self.up_proj, "mxfp8_tp_padding", (0, 0)))
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


def _apply_ascend_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: ReplicatedLinear,
    norm: RMSNorm,
    num_valid_blocks: int,
) -> torch.Tensor:
    """Apply Kimi's learned residual mixture through native AttnResFwd."""
    if num_valid_blocks <= 0:
        return prefix_sum
    return torch.ops._C_ascend.attn_res_fwd(
        prefix_sum,
        block_residual[:, :num_valid_blocks, :],
        proj.weight,
        norm.weight,
        norm.variance_epsilon,
    )


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
        parallel_config = get_current_vllm_config().parallel_config
        qrep_requested = (
            envs.VLLM_DCP_Q_REPLICATE if envs.is_set("VLLM_DCP_Q_REPLICATE") else parallel_config.dcp_q_replicate
        )
        if (
            qrep_requested
            and parallel_config.decode_context_parallel_size > 1
            and parallel_config.prefill_context_parallel_size == 1
        ):
            # Replace the projection before checkpoint loading. Keep its model
            # attribute and the backend reference on the same weight module.
            name = "q_b_proj" if q_lora_rank is not None else "q_proj"
            projection = DCPGroupColumnParallelLinear(
                q_lora_rank if q_lora_rank is not None else hidden_size,
                num_heads * (qk_nope_head_dim + qk_rope_head_dim),
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.{name}",
            )
            setattr(self, name, projection)
            attention_layer.impl.q_proj = projection
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

    @kv_cache.setter
    def kv_cache(self, value):
        # ModelRunner V2 detaches cache storage through every model module
        # during shutdown. Forward that assignment to the owning MLA layer.
        self._attention_layer.kv_cache = value

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
                reduce_results=not use_sequence_parallel,
                prefix=f"{prefix}.mlp",
                # Decoder _run_mlp owns the SP gather/reduce-scatter pair.
                use_sequence_parallel=False,
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

        self.fuse_o_proj_mm_reduce_scatter = (
            get_ascend_config().enable_kimi_o_proj_mm_reduce_scatter
            and self._enable_o_proj_mm_reduce_scatter(vllm_config)
        )

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

    def _run_self_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Ascend attention returns its output instead of filling an AMD buffer.
        return self.self_attn(positions=positions, hidden_states=hidden_states)

    def _run_mlp(self, hidden_states: torch.Tensor, full_num_tokens: int) -> torch.Tensor:
        # Routed MoE already consumes SP shards. Dense layers still have
        # TP-sharded weights and need the same AG/RS boundaries as attention.
        dense_sp = self.use_sequence_parallel and not self.is_moe_layer
        if dense_sp:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        hidden_states = self.mlp(hidden_states)
        if dense_sp:
            hidden_states = sp_reduce_scatter(hidden_states)
        return hidden_states

    def forward(self, positions, hidden_states, residual, **kwargs):
        if self.use_attn_residuals:
            return self.forward_attn_residual(positions, hidden_states, residual, **kwargs)
        if not self.use_sequence_parallel:
            return super().forward(positions, hidden_states, residual, **kwargs)

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]
        hidden_states = self._run_self_attn(positions, hidden_states)
        hidden_states = sp_reduce_scatter(hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self._run_mlp(hidden_states, positions.shape[0])
        return hidden_states, residual

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
        op = (
            torch.ops._C_ascend.attn_res_fwd.fused_prefill
            if optimize_prefill
            else torch.ops._C_ascend.attn_res_fwd.fused
        )
        return op(
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
        hidden_states = self.self_attn(hidden_states=hidden_states, positions=positions)
        if self.use_sequence_parallel and not self.fuse_o_proj_mm_reduce_scatter:
            hidden_states = sp_reduce_scatter(hidden_states)

        mlp_valid_blocks = self.prev_valid_blocks + int(self.is_block_write_layer)
        op = (
            torch.ops._C_ascend.attn_res_fwd.fused_prefill
            if optimize_prefill
            else torch.ops._C_ascend.attn_res_fwd.fused
        )
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
        )
        mlp_output = self._run_mlp(hidden_states, positions.shape[0])
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
        )
        return hidden_states, block_residual


class AscendKimiLinearModel(UpstreamKimiLinearModel):
    """Kimi text model assembled from the Ascend decoder layer."""

    # The Ascend forward path carries cumulative auxiliary states across PP.
    # Use the same keys for MRV2 receive buffers and intermediate-stage relay.
    supports_aux_hidden_states_over_pp = True
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
        self.vocab_size = config.vocab_size
        # vLLM's generic MoE SP switch currently requires DP > 1. K3 also
        # needs the same rank-local token layout for the TP/EP, DP=1 topology
        # that FlashComm used before the standard SP operators were available.
        self.use_sequence_parallel = enable_kimi_k3_sp(vllm_config)

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

    def _maybe_add_hidden_state(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if layer_idx in self.aux_hidden_state_layers:
            # MLA DSpark consumes the raw prefix sum, not the 3-D AttnRes bank.
            value = hidden_states
            if residual is not None and residual.ndim == 2:
                value = value + residual
            aux_hidden_states.append(value)
        return aux_hidden_states

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        # Materialized DSpark states are captured before a layer; raw states
        # are captured after the preceding layer. Handle a PP cut at either.
        return make_pp_empty_intermediate_tensors(
            self,
            super().make_empty_intermediate_tensors,
            include_start_layer=not (
                self.dspark_aux_capture_materialized and self.config.attn_res_block_size is not None
            ),
        )(batch_size, dtype, device)

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

        return super().load_weights(remap_mixed_gate_weights())

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        full_num_tokens = positions.shape[0]
        optimize_attn_res_prefill = self.config.attn_res_block_size is not None and _use_attn_res_prefill_kernel()
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                # Every stage starts with a full-token padding mask, even
                # though only the first stage starts with full activations.
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding,
                    positions,
                )
            if get_pp_group().is_first_rank:
                hidden_states = sp_shard(hidden_states)

        materialized_aux = self.dspark_aux_capture_materialized and self.config.attn_res_block_size is not None
        aux_hidden_states = get_pp_transport_tensors(intermediate_tensors, PPTransportDataType.AUX_HIDDEN_STATES)
        if not materialized_aux and get_pp_group().is_first_rank:
            aux_hidden_states = self._maybe_add_hidden_state(
                aux_hidden_states,
                self.start_layer,
                hidden_states,
                residual,
            )
        if self.config.attn_res_block_size is not None:
            attn_res_block_num = cdiv(self.end_layer, self.config.attn_res_block_size)
            block_residual = hidden_states.new_empty(hidden_states.size(0), attn_res_block_num, hidden_states.size(1))
            if residual is not None:
                block_residual[:, : residual.size(1), :].copy_(residual)
            residual = block_residual

        pending_mlp_output = None
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            if self.config.attn_res_block_size is not None:
                prepared = layer.prepare_attn_residual(
                    hidden_states,
                    residual,
                    pending_mlp_output,
                    return_materialized=materialized_aux and layer_idx in self.aux_hidden_state_layers,
                    optimize_prefill=optimize_attn_res_prefill,
                )
                # Raw capture logically belongs to the previous layer's tail.
                # The first stage's input and preceding PP stage's tail were
                # already captured, so do not append that boundary twice.
                if layer_idx > self.start_layer and not materialized_aux:
                    self._maybe_add_hidden_state(aux_hidden_states, layer_idx, prepared[1], residual)
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
            else:
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )
                if not materialized_aux:
                    self._maybe_add_hidden_state(aux_hidden_states, layer_idx + 1, hidden_states, residual)

        if (
            self.config.attn_res_block_size is not None
            and not get_pp_group().is_last_rank
            and pending_mlp_output is not None
        ):
            op = (
                torch.ops._C_ascend.attn_res_fwd.fused_prefill
                if optimize_attn_res_prefill
                else torch.ops._C_ascend.attn_res_fwd.fused
            )
            hidden_states, _, _ = op(
                hidden_states,
                pending_mlp_output,
                residual,
                layer.mlp_res_proj.weight,
                layer.mlp_res_norm.weight,
                layer.mlp_res_norm.variance_epsilon,
                0,
                mix=False,
            )
            if not materialized_aux:
                self._maybe_add_hidden_state(aux_hidden_states, self.end_layer, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            if residual is None:
                # An explicitly empty first PP stage has no additive residual
                # yet. A zero tensor preserves its semantics in the PP buffers.
                if self.config.attn_res_block_size is not None:
                    residual = hidden_states.new_zeros(hidden_states.shape[0], 0, hidden_states.shape[-1])
                else:
                    residual = torch.zeros_like(hidden_states)
            return add_pp_transport_tensors(
                IntermediateTensors({"hidden_states": hidden_states, "residual": residual}),
                PPTransportDataType.AUX_HIDDEN_STATES,
                aux_hidden_states,
            )

        if self.config.attn_res_block_size is not None:
            op = (
                torch.ops._C_ascend.attn_res_fwd.fused_prefill
                if optimize_attn_res_prefill
                else torch.ops._C_ascend.attn_res_fwd.fused
            )
            hidden_states, final_prefix, _ = op(
                hidden_states,
                pending_mlp_output,
                residual,
                self.output_attn_res_proj.weight,
                self.output_attn_res_norm.weight,
                self.output_attn_res_norm.variance_epsilon,
                attn_res_block_num,
            )
            if not materialized_aux and pending_mlp_output is not None:
                self._maybe_add_hidden_state(aux_hidden_states, self.end_layer, final_prefix, residual)
        elif residual is not None:
            hidden_states = hidden_states + residual
        # Materialized draft capture at the final boundary follows AttnRes.
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


def _get_kimi_k3_vision_config(config):
    """Keep K3 position interpolation aligned with its reference model."""
    vision_config = copy(config)
    # The released K3 reference constructor leaves the interpolation mode at
    # MoonVision3dPatchEmbed's bicubic default. Its config carries a bilinear
    # field that the reference model never consumes.
    vision_config.pos_emb_interpolation_mode = "bicubic"
    return vision_config


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

        vision_config = _get_kimi_k3_vision_config(self.config.vision_config)
        self.use_data_parallel = is_vit_use_data_parallel(
            vision_config.num_attention_heads,
        )
        self.hidden_size = self.config.text_config.hidden_size
        self.device = current_platform.current_device()
        vision_quant_config = self._maybe_ignore_quant_config(self.quant_config)

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = MoonViT3dPretrainedModel(
                vision_config,
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
                vision_config,
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
