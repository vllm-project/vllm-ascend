# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 model adapters for vLLM 0.27 on Ascend.

vLLM owns Kimi's configuration, multimodal processor, weight mappings, and
model-level forward contract.  This module composes those upstream pieces with
the generic MLA/MoE implementation and the Ascend KDA backend.
"""

import math
from collections.abc import Iterable
from copy import copy

import torch
from torch import nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
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
    AutoWeightsLoader,
    PPMissingLayer,
    init_vllm_registered_model,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.models.kimi_k3.amd.linear import (
    KimiDecoderLayer as UpstreamKimiDecoderLayer,
)
from vllm.models.kimi_k3.amd.linear import KimiLinearForCausalLM as UpstreamKimiLinearForCausalLM
from vllm.models.kimi_k3.amd.linear import KimiLinearModel as UpstreamKimiLinearModel
from vllm.models.kimi_k3.amd.linear import (
    KimiMLP,
    KimiRoutedOutputTransform,
)
from vllm.models.kimi_k3.amd.linear import (
    KimiMoE as UpstreamKimiMoE,
)
from vllm.models.kimi_k3.amd.model import (
    KimiK3ForConditionalGeneration as UpstreamKimiK3ForConditionalGeneration,
)
from vllm.models.kimi_k3.common.mm_preprocess import (
    KimiK3DummyInputsBuilder,
    KimiK3MultiModalProcessor,
    KimiK3ProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.activation import AscendSituAndMul  # type: ignore[attr-defined]
from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention  # type: ignore[import-untyped]


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
    # Avoid materializing a broadcasted FP32 tensor as large as the entire
    # normalized residual stack.
    scores = torch.matmul(normalized_without_gamma, score_weight)
    probabilities = scores.softmax(-1).unsqueeze(1)
    mixed = torch.matmul(probabilities, values_fp32).squeeze(1).to(values.dtype)
    if _EXTRA_CTX.flash_comm_v1_enabled:
        mixed = torch.ops.vllm.maybe_chunk_residual(prefix_sum, mixed)
    return mixed


class AscendKimiMLP(KimiMLP):
    """Use the Ascend SiTU module for dense and shared-expert MLPs."""

    def __init__(
        self,
        *args,
        hidden_act: str,
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            hidden_act=hidden_act,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
            **kwargs,
        )
        if hidden_act == "situ":
            self.act_fn = AscendSituAndMul(
                beta=activation_situ_beta or 1.0,
                linear_beta=activation_situ_linear_beta,
            )


class AscendKimiMoE(UpstreamKimiMoE):
    """Adapt Kimi K3 latent MoE construction to Ascend.

    The upstream AMD implementation pads small expert partitions at the model
    layer before the backend MoE factory resolves TP versus EP.  Under EP this
    applies a TP-sized pad even though each rank owns complete experts.  Ascend
    routed experts already perform any backend-required size rounding, so keep
    Kimi's checkpoint intermediate size here and leave padding to that layer.

    Native ModelSlim checkpoints also quantize the latent down/up projections,
    while the upstream implementation always creates them unquantized.  Rebind
    those projections and the runner transforms to their Ascend quantized
    modules after the common Kimi structure has been assembled.
    """

    def __init__(
        self,
        *,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        ascend_config = copy(config)
        ascend_config.min_moe_intermediate_per_partition = 0
        super().__init__(
            config=ascend_config,
            quant_config=quant_config,
            prefix=prefix,
            **kwargs,
        )

        latent_quant_config = quant_config if quant_config is not None and quant_config.get_name() == "ascend" else None
        if not self.use_latent_moe or latent_quant_config is None:
            return

        self.routed_expert_down_proj = ReplicatedLinear(
            config.hidden_size,
            self.moe_hidden_size,
            bias=False,
            quant_config=latent_quant_config,
            prefix=f"{prefix}.routed_expert_down_proj",
        )
        self.routed_expert_up_proj = ReplicatedLinear(
            self.moe_hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=latent_quant_config,
            prefix=f"{prefix}.routed_expert_up_proj",
        )
        self.routed_output_transform = KimiRoutedOutputTransform(
            self.routed_expert_norm,
            self.routed_expert_up_proj,
        )
        self.experts.routed_input_transform = self.routed_expert_down_proj
        self.experts.routed_output_transform = self.routed_output_transform


class AscendKimiMLAAttention(nn.Module):
    """Generic vLLM MLA composition backed by Ascend's pluggable wrapper."""

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
    ) -> None:
        """Assemble Kimi's projections, optional RoPE, and generic MLA wrapper."""
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5

        self.fused_qkv_a_proj = None
        self.kv_a_proj_with_mqa = None
        self.q_a_layernorm = None
        self.q_b_proj = None
        self.q_proj = None
        if q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                hidden_size,
                [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(
                q_lora_rank,
                eps=config.rms_norm_eps,
            )
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                hidden_size,
                num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                hidden_size,
                kv_lora_rank + qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        self.kv_a_layernorm = RMSNorm(
            kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.g_proj = (
            ColumnParallelLinear(
                hidden_size,
                num_heads * v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
            if use_output_gate
            else None
        )
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = None
        if use_rope:
            rope_parameters = dict(config.rope_parameters)
            if rope_parameters["rope_type"] != "default":
                rope_parameters["rope_type"] = (
                    "deepseek_yarn" if rope_parameters.get("apply_yarn_scaling", True) else "deepseek_llama_scaling"
                )
            self.rotary_emb = get_rope(
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

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
            g_proj=self.g_proj,
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            hidden_size,
            self.num_local_heads,
            self.scaling,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
        )

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
    ) -> None:
        """Select KDA or no-RoPE MLA and configure the layer residual path."""
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.is_moe = config.is_moe
        layer_idx = self.layer_idx
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        if config.is_kda_layer(layer_idx):
            self.self_attn = AscendKimiK3DeltaAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )
            self._self_attn_writes_output = False
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
            self._self_attn_writes_output = False

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
                layer_idx=layer_idx,
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = AscendKimiMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
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

        self.is_vl_first_layer = self.self_attn.is_vl_first_layer

    def _run_self_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self._self_attn_writes_output:
            return self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        output = torch.empty_like(hidden_states)
        self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            output=output,
        )
        return output

    def forward_attn_residual(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run upstream attn-res with the multimodal FlashComm transition."""
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
        hidden_states = self._run_self_attn(positions, hidden_states)

        if self.is_vl_first_layer and _EXTRA_CTX.flash_comm_v1_enabled:
            block_residual = torch.ops.vllm.maybe_chunk_residual(
                hidden_states.unsqueeze(1),
                block_residual,
            )

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
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvgfab": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
        ],
        "conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size

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
        """Load mixed-precision KDA gates into the FLOAT packed module."""
        params_dict = dict(self.named_parameters())
        gate_mapping = (
            (".g_proj", ".in_proj_gfab", 0),
            (".f_a_proj", ".in_proj_gfab", 1),
            (".b_proj", ".in_proj_gfab", 2),
        )
        loaded_gate_params = set()

        def load_non_gate_weights():
            for args in weights:
                name, loaded_weight = args[:2]
                for source, target, shard_id in gate_mapping:
                    if source not in name:
                        continue
                    mapped_name = name.replace(source, target)
                    if mapped_name in params_dict:
                        param = params_dict[mapped_name]
                        module_name = mapped_name.rsplit(".", 1)[0]
                        module = self.get_submodule(module_name)
                        module.load_shard_weight(param, loaded_weight, shard_id)
                        loaded_gate_params.add(mapped_name)
                        break
                else:
                    yield args

        return super().load_weights(load_non_gate_weights()) | loaded_gate_params

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
        residual = block_residual

        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if (layer_idx + 1) in self.aux_hidden_state_layers:
                self._maybe_add_hidden_state(
                    aux_hidden_states,
                    layer_idx + 1,
                    hidden_states,
                    residual,
                )

        if not get_pp_group().is_last_rank:
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


class AscendKimiK3MultiModalProjector(KimiK25MultiModalProjector):
    """Kimi projector with the optional ModelSlim output rotation."""

    def __init__(self, config, *args, prefix: str = "", **kwargs) -> None:
        super().__init__(config, *args, prefix=prefix, **kwargs)
        output_size = config.text_hidden_size
        self.rot_proj: ReplicatedLinear | None = ReplicatedLinear(
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

    def _maybe_ignore_quant_config(
        self,
        quant_config: QuantizationConfig | None,
    ) -> QuantizationConfig | None:
        if isinstance(
            quant_config,
            compressed_tensors.CompressedTensorsConfig,
        ):
            return None
        return quant_config

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        rot_proj = self.mm_projector.rot_proj
        loader = AutoWeightsLoader(self)
        rot_proj_weight_names = (
            {name for name, _ in rot_proj.named_parameters(prefix="mm_projector.rot_proj")}
            if rot_proj is not None
            else set()
        )
        loaded_weights = loader.load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )
        if rot_proj is not None and rot_proj_weight_names.isdisjoint(loaded_weights):
            self.mm_projector.rot_proj = None
        return loaded_weights
