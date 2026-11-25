from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models import deepseek_v2 as deepseek_v2_module
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekAttention,
    DeepseekV2MLP,
    DeepseekV2ForCausalLM,
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV2MLAAttention,
    DeepseekV2Model,
    Indexer,
    _get_llama_4_scaling,
    yarn_get_mscale,
)
from vllm.model_executor.models.interfaces import EagleModelMixin
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    sequence_parallel_chunk,
    spec_decode_needs_target_embed,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.mla.index_group import SparseMLAIndexGroupBuilder

from vllm_ascend.attention.context_parallel.common_cp import use_dcp_q_replicate
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.layernorm import AscendRMSNorm
from vllm_ascend.ops.linear import AscendDCPGroupColumnParallelLinear, AscendMergedColumnParallelLinear
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.base import PreparedLinearInput
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import enable_custom_op, is_mtp_layer, should_reuse_topk, vllm_version_is
from vllm_ascend.worker.v2 import pp_transport
from vllm_ascend.worker.v2.pp_transport import (
    PPTransportDataType,
    configure_pp_topk_transport,
)


def _should_skip_indexer_init(
    config: DeepseekV2Config | DeepseekV3Config,
    prefix: str,
    skip_topk: bool,
) -> bool:
    if not skip_topk:
        return False

    layer_id = extract_layer_index(prefix)
    if is_mtp_layer(config, prefix):
        return False

    # GLM-5.2 describes checkpoint-level shared indexers explicitly. Runtime
    # IndexCache overrides on GLM-5.1 only skip top-k computation; its
    # checkpoint still contains an Indexer for every layer.
    indexer_types = getattr(config, "indexer_types", None)
    indexer_type = indexer_types[layer_id] if indexer_types is not None and layer_id < len(indexer_types) else None
    return isinstance(indexer_type, str) and indexer_type.lower() == "shared"


def _deepseek_v2_mla_attention_init(
    self,
    vllm_config: VllmConfig,
    config: DeepseekV2Config | DeepseekV3Config,
    hidden_size: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    q_lora_rank: int | None,
    kv_lora_rank: int,
    max_position_embeddings: int = 8192,
    cache_config: CacheConfig | None = None,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
    topk_indices_buffer: torch.Tensor | None = None,
    input_size: int | None = None,
    reduce_results: bool = True,
    index_group_builder: SparseMLAIndexGroupBuilder | None = None,
    non_causal_multi_token_decode: bool = False,
) -> None:
    # Zero-argument super() is unavailable because this function is defined
    # outside the class and assigned to DeepseekV2MLAAttention.__init__.
    nn.Module.__init__(self)

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
    self.max_position_embeddings = max_position_embeddings

    # Use input_size for projection input dimensions if provided,
    # otherwise default to hidden_size (used in Eagle3 Deepseek with MLA).
    proj_input_size = input_size if input_size is not None else self.hidden_size

    if self.q_lora_rank is not None:
        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            proj_input_size,
            [
                self.q_lora_rank,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
    else:
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            proj_input_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )

    dcp_q_replicate_enabled = use_dcp_q_replicate(vllm_config)
    q_proj_cls = AscendDCPGroupColumnParallelLinear if dcp_q_replicate_enabled else ColumnParallelLinear
    if self.q_lora_rank is not None:
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.q_b_proj = q_proj_cls(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
    else:
        self.q_proj = q_proj_cls(
            proj_input_size,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

    self.kv_a_layernorm = RMSNorm(
        self.kv_lora_rank,
        eps=config.rms_norm_eps,
    )

    self.kv_b_proj = ColumnParallelLinear(
        self.kv_lora_rank,
        self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.kv_b_proj",
    )

    self.o_proj = RowParallelLinear(
        self.num_heads * self.v_head_dim,
        self.hidden_size,
        bias=False,
        reduce_results=reduce_results,
        quant_config=quant_config,
        prefix=f"{prefix}.o_proj",
    )

    if config.rope_parameters["rope_type"] != "default":
        config.rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if config.rope_parameters.get(
                "apply_yarn_scaling",
                True,
            )
            else "deepseek_llama_scaling"
        )

    self.rotary_emb = get_rope(
        qk_rope_head_dim,
        max_position=max_position_embeddings,
        rope_parameters=config.rope_parameters,
        is_neox_style=False,
    )

    if config.rope_parameters["rope_type"] != "default" and config.rope_parameters["rope_type"] == "deepseek_yarn":
        mscale_all_dim = config.rope_parameters.get(
            "mscale_all_dim",
            False,
        )
        scaling_factor = config.rope_parameters["factor"]
        mscale = yarn_get_mscale(
            scaling_factor,
            float(mscale_all_dim),
        )
        self.scaling = self.scaling * mscale * mscale

    self.is_v32 = hasattr(config, "index_topk")

    # IndexCache config.
    #
    # skip_topk controls top-k reuse. Indexer initialization is skipped only
    # when the checkpoint marks this layer as sharing another layer's Indexer.
    layer_id = extract_layer_index(prefix)
    _skip_topk = should_reuse_topk(config, layer_id)

    # The skip pattern only governs backbone layers. MTP/nextn layers
    # (layer_id >= num_hidden_layers) must never start with skip_topk=True:
    # they compute their own indices at draft step 0 and toggle at runtime
    # via set_skip_topk (index_share_for_mtp_iteration). Matches upstream
    # deepseek_v2.py behavior.
    mtp_layer = is_mtp_layer(config, prefix)

    skip_indexer_init = _should_skip_indexer_init(config, prefix, _skip_topk)
    if self.is_v32 and not skip_indexer_init:
        self.indexer_rope_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=not getattr(
                config,
                "indexer_rope_interleave",
                False,
            ),
        )

        self.indexer = Indexer(
            vllm_config,
            config,
            hidden_size,
            q_lora_rank,
            quant_config,
            cache_config,
            topk_indices_buffer,
            f"{prefix}.indexer",
            is_inplace_rope=self.indexer_rope_emb.enabled(),
        )
    else:
        self.indexer_rope_emb = None
        self.indexer = None

    # Keep the shared model-level buffer reachable from every local layer so
    # the model can transport it when a PP boundary splits an IndexShare or
    # IndexCache group.
    self.topk_indices_buffer = topk_indices_buffer

    mla_modules = MLAModules(
        kv_a_layernorm=self.kv_a_layernorm,
        kv_b_proj=self.kv_b_proj,
        rotary_emb=self.rotary_emb,
        o_proj=self.o_proj,
        fused_qkv_a_proj=(self.fused_qkv_a_proj if self.q_lora_rank is not None else None),
        kv_a_proj_with_mqa=(self.kv_a_proj_with_mqa if self.q_lora_rank is None else None),
        q_a_layernorm=(self.q_a_layernorm if self.q_lora_rank is not None else None),
        q_b_proj=(self.q_b_proj if self.q_lora_rank is not None else None),
        q_proj=(self.q_proj if self.q_lora_rank is None else None),
        indexer=self.indexer,
        indexer_rotary_emb=self.indexer_rope_emb,
        is_sparse=self.is_v32,
        topk_indices_buffer=topk_indices_buffer,
        index_group_builder=index_group_builder,
    )

    self.mla_attn = MultiHeadLatentAttentionWrapper(
        self.hidden_size,
        self.num_local_heads,
        self.scaling,
        self.qk_nope_head_dim,
        self.qk_rope_head_dim,
        self.v_head_dim,
        self.q_lora_rank,
        self.kv_lora_rank,
        mla_modules,
        cache_config,
        quant_config,
        prefix,
        skip_topk=_skip_topk and not mtp_layer,
        non_causal_multi_token_decode=non_causal_multi_token_decode,
    )


DeepseekV2MLAAttention.__init__ = _deepseek_v2_mla_attention_init


_original_deepseek_v2_model_init = DeepseekV2Model.__init__


def _find_topk_indices_buffer(layers: Iterable[object]) -> torch.Tensor | None:
    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        topk_indices_buffer = getattr(self_attn, "topk_indices_buffer", None)
        if topk_indices_buffer is not None:
            return topk_indices_buffer
    return None


def _patched_deepseek_v2_model_init(self, *args, **kwargs):
    _original_deepseek_v2_model_init(self, *args, **kwargs)
    vllm_config = kwargs["vllm_config"]
    # GLM-5.x uses the generic DeepseekV2Model on Ascend. Provision the
    # last-stage embedding before the normal target checkpoint load, so
    # upstream DFlash sharing needs no override. Preserve an embedding that
    # a newer upstream constructor already provides. Draft QuaRot alignment
    # remains a separate post-processing step on private draft weights.
    if isinstance(self.embed_tokens, PPMissingLayer) and spec_decode_needs_target_embed(vllm_config):
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{kwargs.get('prefix', '')}.embed_tokens",
        )
    self.topk_indices_buffer = _find_topk_indices_buffer(self.layers)
    transport_data_types = (PPTransportDataType.TOPK_INDICES,)
    configure_pp_topk_transport(self, transport_data_types)
    self.make_empty_intermediate_tensors = pp_transport.make_empty_intermediate_tensors(
        self,
        self.make_empty_intermediate_tensors,
        transport_data_types,
    )


DeepseekV2Model.__init__ = _patched_deepseek_v2_model_init


# Reuse upstream slot bookkeeping; Ascend owns capture and TP gathering.
# v0.34.0's DeepseekV2Model does not inherit EagleModelMixin yet, so the mixin
# members have to be copied in place; the native inheritance only landed on main.
if vllm_version_is("0.34.0"):
    for _member in (
        "AUX_HIDDEN_STATE_KEY",
        "_aux_slot_base_cached",
        "_aux_upstream_total_cached",
        "_set_aux_hidden_state_layers",
        "_cache_aux_pp_layout",
        "pack_local_aux_hidden_states",
        "collect_remote_aux_hidden_states",
    ):
        setattr(DeepseekV2Model, _member, getattr(EagleModelMixin, _member))
    DeepseekV2Model.supports_aux_hidden_states_over_pp = True
else:

    class AscendEagleDeepseekV2Model(DeepseekV2Model, EagleModelMixin):
        """Expose the native decoder identity required by vLLM #66177.

        Inherit the decorated DeepSeek implementation and its Ascend patches;
        the main mixin supplies local slots, and the runner merges remote ones.
        Remove this replacement when native DeepSeek inherits EagleModelMixin.
        """

        supports_aux_hidden_states_over_pp = True

    deepseek_v2_module.DeepseekV2Model = AscendEagleDeepseekV2Model
    # Native constructors use this cached factory, not the module alias above.
    DeepseekV2ForCausalLM.model_cls = AscendEagleDeepseekV2Model


def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
    self.model._set_aux_hidden_state_layers(layers)


DeepseekV2ForCausalLM.set_aux_hidden_state_layers = _set_aux_hidden_state_layers


def _capture_aux_hidden_state(self, aux_hidden_states, layer_id, hidden_states, residual, positions):
    if layer_id not in self.aux_hidden_state_layers:
        return
    aux_hidden_state = hidden_states if residual is None else hidden_states + residual
    if aux_hidden_state.shape[0] != positions.shape[0]:
        aux_hidden_state = tensor_model_parallel_all_gather(aux_hidden_state, 0)
        aux_hidden_state = aux_hidden_state[: positions.shape[0]]
    aux_hidden_states.append(aux_hidden_state)


def _patched_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    pp_group = get_pp_group()
    if pp_group.is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided to DeepseekV2Model.forward")
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
        aux_hidden_states: list[torch.Tensor] = []
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]
        aux_hidden_states = self.collect_remote_aux_hidden_states(intermediate_tensors)

    llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
    llama_4_scaling: torch.Tensor | None
    if llama_4_scaling_config is not None:
        llama_4_scaling = _get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config["original_max_position_embeddings"],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    if pp_group.is_first_rank:
        _capture_aux_hidden_state(self, aux_hidden_states, 0, hidden_states, residual, positions)
    for idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        hidden_states, residual = layer(positions, hidden_states, residual, llama_4_scaling)
        # A boundary state belongs to the stage producing it, including end_layer.
        _capture_aux_hidden_state(self, aux_hidden_states, idx + 1, hidden_states, residual, positions)

    if not pp_group.is_last_rank:
        pp_output = IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        # vLLM's buffer-mutation check also sees names in traced-out branches.
        # Avoid leaving "update" in the forward's co_names for PP=1.
        pp_output = IntermediateTensors({**pp_output.tensors, **self.pack_local_aux_hidden_states(aux_hidden_states)})
        if self.send_pp_topk_indices:
            assert self.topk_indices_buffer is not None
            num_tokens = positions.shape[0]
            pp_transport.add_pp_transport_tensors(
                pp_output,
                PPTransportDataType.TOPK_INDICES,
                [self.topk_indices_buffer[:num_tokens]],
            )
        return pp_output

    if hidden_states.shape[0] != positions.shape[0]:
        combined_states = torch.cat([hidden_states, residual], dim=-1)
        combined_states = tensor_model_parallel_all_gather(combined_states, 0)
        combined_states = combined_states[: positions.shape[0]]
        hidden_size = self.hidden_size
        hidden_states, residual = combined_states.split([hidden_size, hidden_size], dim=-1)
        residual = residual.contiguous()

    hidden_states, _ = self.norm(hidden_states, residual)
    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states


_ORIGINAL_DECODER_FORWARD = DeepseekV2DecoderLayer.forward


def _supports_dense_norm_quant(layer, hidden_states):
    """Select the decoder path before running attention or mutating residuals."""
    if not isinstance(layer.mlp, DeepseekV2MLP):
        return False
    projection = layer.mlp.gate_up_proj
    adapter = getattr(projection, "quant_method", None)
    scheme = getattr(adapter, "quant_method", None)
    return (
        type(projection) in (MergedColumnParallelLinear, AscendMergedColumnParallelLinear)
        and type(adapter) is AscendLinearMethod
        and type(scheme) is AscendW8A8DynamicLinearMethod
        and scheme.act_quant_type == torch.int8
        and not getattr(projection, "requires_unquantized_input", False)
        and getattr(projection, "custom_op", None) is None
        and not layer.use_sequence_parallel_moe
        and hidden_states.dtype == torch.bfloat16
        and isinstance(layer.post_attention_layernorm, AscendRMSNorm)
        and enable_custom_op()
        and get_current_hardware_profile().supports(HardwareCapability.GRAPH_NORM_QUANT_FUSION)
    )


def _maybe_fused_add_rms_norm_quant(norm, hidden_states, residual):
    """Prepare the dense MLP input with fused add/norm/quant when supported."""
    if (
        hidden_states.ndim != 2
        or hidden_states.dtype != torch.bfloat16
        or residual.dtype != hidden_states.dtype
        or hidden_states.shape != residual.shape
        or not hidden_states.is_contiguous()
        or not residual.is_contiguous()
    ):
        return norm(hidden_states, residual)

    output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
        hidden_states,
        residual,
        norm.weight,
        beta=norm.bias,
        epsilon=norm.variance_epsilon,
        output_mask=[True, False],
    )
    prepared = PreparedLinearInput(
        quantized=output[0],
        scale=output[3],
        output_dtype=hidden_states.dtype,
    )
    return prepared, output[2]


def _decoder_forward_with_norm_quant(self, positions, hidden_states, residual, llama_4_scaling=None):
    """Explicit dense-MLP norm/quant fusion; remove when upstream owns it."""
    # Restrict the model-level override to the observed dense W8A8 path. Other
    # layers, sequence-parallel boundaries and unsupported consumers retain
    # upstream's complete forward, including its FP16 overflow handling.
    if not _supports_dense_norm_quant(self, hidden_states):
        return _ORIGINAL_DECODER_FORWARD(self, positions, hidden_states, residual, llama_4_scaling)

    # Keep this forward aligned with DeepseekV2DecoderLayer.forward. Only the
    # post-attention norm/quant call below intentionally changes its behavior.
    full_num_tokens = positions.shape[0]
    input_is_sequence_parallel = (
        self.use_sequence_parallel_moe and residual is not None and hidden_states.shape[0] != full_num_tokens
    )

    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    if input_is_sequence_parallel:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[:full_num_tokens]

    if self.use_mha:
        hidden_states = self.self_attn(positions, hidden_states)
    else:
        hidden_states = self.self_attn(positions, hidden_states, llama_4_scaling)

    if not isinstance(self.self_attn, DeepseekAttention) and hidden_states.dtype == torch.float16:
        # Match the upstream FP16 overflow handling after attention.
        hidden_states *= 1.0 / self.routed_scaling_factor
        if self.layer_idx == 0:
            residual *= 1.0 / self.routed_scaling_factor

    if self.use_sequence_parallel_moe:
        tp_world_size = get_tensor_model_parallel_world_size()
        sp_pad = (-hidden_states.shape[0]) % tp_world_size
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
        hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
        if not input_is_sequence_parallel:
            residual = sequence_parallel_chunk(residual)

    # Fully Connected
    hidden_states, residual = _maybe_fused_add_rms_norm_quant(
        self.post_attention_layernorm,
        hidden_states,
        residual,
    )
    if self.use_sequence_parallel_moe:
        hidden_states = self.mlp(hidden_states, already_sequence_parallel=True)
    else:
        hidden_states = self.mlp(hidden_states)

    if isinstance(self.mlp, DeepseekV2MLP) and hidden_states.dtype == torch.float16:
        # Match the upstream FP16 overflow handling after the dense MLP.
        hidden_states *= 1.0 / self.routed_scaling_factor

    return hidden_states, residual


DeepseekV2DecoderLayer.forward = _decoder_forward_with_norm_quant
DeepseekV2Model.forward = _patched_forward
