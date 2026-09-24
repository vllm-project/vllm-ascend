# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 text model and source-shared hybrid-cache graph."""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import vllm.envs as envs
from safetensors import safe_open
from torch import nn
from transformers import PretrainedConfig
from vllm.config import ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import PPMissingLayer, is_pp_missing_parameter, make_layers, maybe_prefix

# Upstream #56741 normalized the V4.1 model package name.
from vllm.models.deepseek_v41.common.engram import EngramLayout
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheLayer,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm_ascend.ops.dsa import AscendDeepseekSparseAttention, DSAModules
from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding
from vllm_ascend.ops.triton.mul_add import muls_add_triton
from vllm_ascend.utils import (
    enable_custom_op,
    enable_dsa_cp,
    normalize_deepseek_v41_config,
)

from .compressor import DeepseekV41Compressor
from .engram import (
    create_engram_hash_state,
    engram_cpu_offload,
    engram_dead_mask,
    engram_enabled,
)
from .engram.embedding import (
    AscendParallelEngramEmbedding,
    preflight_engram_checkpoint,
)
from .engram.layer import AscendEngram
from .engram.parallel import gather_engram_hashes, get_engram_dp_size
from .indexer import DeepseekV41Indexer


class DeepseekV41MLP(nn.Module):
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
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV41MoE(nn.Module):
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

            self.shared_experts = DeepseekV41MLP(
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
                    final_hidden_states = muls_add_triton(
                        final_hidden_states, shared_output, self.routed_scaling_factor
                    )
                else:
                    final_hidden_states *= self.routed_scaling_factor
            elif self.shared_experts is not None:
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


class DeepseekV41MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list[DeepseekV41MoE]
    """
    List of MoE MLP layers in the model.
    """

    def extract_moe_parameters(self, example_moe: DeepseekV41MoE | None):
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


@dataclass(frozen=True)
class DeepseekV41LayerRole:
    """The attention and future Engram responsibilities of one backbone layer."""

    layer_idx: int
    compress_ratio: int
    kv_source_layer: int | None
    index_source_layer: int | None
    is_kv_source: bool
    is_index_source: bool
    is_candidate_source: bool
    uses_candidate_filter: bool
    engram_slot: int | None

    @property
    def has_long_context(self) -> bool:
        return self.compress_ratio > 0


@dataclass(frozen=True)
class DeepseekV41Topology:
    """Validated, immutable model-wide source/consumer topology."""

    layers: tuple[DeepseekV41LayerRole, ...]
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_topk: int

    def layer(self, layer_idx: int) -> DeepseekV41LayerRole:
        return self.layers[layer_idx]

    def kv_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.kv_source_layer == source_layer)

    def index_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.index_source_layer == source_layer)


class DeepseekV41SharedAttentionState:
    """Per-forward handoff between index sources and their consumer layers."""

    def __init__(self, topk_indices, candidates):
        self.topk_indices = topk_indices
        self.candidates = candidates

    def reset(self):
        # Source layers overwrite the active rows before any consumer reads
        # them. Keeping the storage intact avoids replay depending on Python
        # state mutation and preserves a fixed address for ACL Graph.
        return None


def _latest_source(layer_idx: int, sources: tuple[int, ...]) -> int | None:
    return next((source for source in reversed(sources) if source <= layer_idx), None)


def build_layer_plan(config: Any) -> DeepseekV41Topology:
    """Build and validate the V4.1 layer-sharing graph from a text config.

    ``config`` is the parsed text-model config. Extra compression ratios for
    speculative layers are allowed,
    but only the first ``num_hidden_layers`` entries describe the backbone.
    """

    num_layers = int(config.num_hidden_layers)
    ratios = tuple(config.compress_ratios)
    kv_sources = tuple(config.kv_source_layer_ids)
    index_sources = tuple(config.index_source_layer_ids)
    engram_layers = tuple(config.engram_layer_ids)
    candidate_source = int(config.candidate_source_layer_id)
    candidate_topk_blocks = int(config.candidate_topk_blocks)
    candidate_block_size = int(config.candidate_block_size)
    index_topk = int(config.index_topk)

    ratios = ratios[:num_layers]

    engram_slots = {layer_idx: slot for slot, layer_idx in enumerate(engram_layers)}
    roles: list[DeepseekV41LayerRole] = []
    for layer_idx, ratio in enumerate(ratios):
        kv_source = _latest_source(layer_idx, kv_sources) if ratio else None
        index_source = _latest_source(layer_idx, index_sources) if ratio else None

        roles.append(
            DeepseekV41LayerRole(
                layer_idx=layer_idx,
                compress_ratio=ratio,
                kv_source_layer=kv_source,
                index_source_layer=index_source,
                is_kv_source=layer_idx in kv_sources,
                is_index_source=layer_idx in index_sources,
                is_candidate_source=layer_idx == candidate_source,
                # Consumer layers inherit the selection policy of their index
                # source.  For example, layer 26 reuses layer 24 TopK, and that
                # TopK was computed inside layer 20's candidate blocks.
                uses_candidate_filter=index_source is not None and index_source > candidate_source,
                engram_slot=engram_slots.get(layer_idx),
            )
        )

    return DeepseekV41Topology(
        layers=tuple(roles),
        kv_source_layer_ids=kv_sources,
        index_source_layer_ids=index_sources,
        candidate_source_layer_id=candidate_source,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        index_topk=index_topk,
    )


class AscendDeepseekV41SWACache(DeepseekV41CacheLayer):
    """Ascend SWA cache registered with the V4.1 allocator."""

    def __init__(self, head_dim, window_size, dtype, prefix, cache_config):
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        block_size = DSV4_BLOCK_SIZES[cache_config.block_size][0][1]
        spec = AscendSlidingWindowMLASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=head_dim,
            dtype=dtype,
            sliding_window=window_size,
            cache_dtype_str=cache_config.cache_dtype,
            model_version="deepseek_v41",
            alignment=None,
        )
        super().__init__(get_current_vllm_config(), prefix, spec)
        self.head_dim = head_dim
        self.window_size = window_size
        self.dtype = dtype
        self.block_size = block_size
        self.cache_config = cache_config


class DeepseekV41SWAAttention(nn.Module):
    """Projection and Ascend SWA execution shared by target and draft."""

    swa_cache_cls = AscendDeepseekV41SWACache

    def __init__(
        self,
        vllm_config,
        config,
        max_position_embeddings=0,
        cache_config=None,
        quant_config=None,
        prefix="",
        topk_indices_buffer=None,
        reduce_results=True,
        need_gather_q_kv=False,
        *,
        use_yarn=False,
    ):
        super().__init__()
        self.layer_idx = int(prefix.split(".")[-2])
        init_attention_projections(self, config, quant_config, prefix, reduce_results)
        self.compress_ratio = 0
        self.compressor = None
        self.indexer = None
        self.rotary_emb = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername=f"{prefix}.attn",
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            is_neox_style=False,
            scaling_factor=config.rope_parameters["factor"],
            base=config.compress_rope_theta if use_yarn else config.rope_theta,
            beta_fast=config.rope_parameters["beta_fast"],
            beta_slow=config.rope_parameters["beta_slow"],
            original_max_position_embeddings=max_position_embeddings,
            apply_yarn_scaling=use_yarn,
            rope_groups=["default"],
        )
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(vllm_config.cache_config.cache_dtype, vllm_config.model_config)
        swa_cache_layer = self.swa_cache_cls(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=kv_cache_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        dsa_modules = DSAModules(
            wq_a=self.wq_a,
            q_norm=self.q_norm,
            q_norm_without_weight=self.q_norm_without_weight,
            wq_b=self.wq_b,
            wkv=self.wkv,
            kv_norm=self.kv_norm,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            indexer=self.indexer,
            compressor=self.compressor,
            swa_cache_layer=swa_cache_layer,
        )

        self.dsa_attn = AscendDeepseekSparseAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            scale=self.scale,
            n_local_heads=self.n_local_heads,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            nope_head_dim=self.nope_head_dim,
            eps=self.eps,
            n_groups=self.n_groups,
            n_local_groups=self.n_local_groups,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            dsa_modules=dsa_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            # prefix=f'{prefix}.attn',
            prefix=f"{prefix}",
            need_gather_q_kv=need_gather_q_kv,
        )


class DeepseekV41Attention(DeepseekV41SWAAttention):
    """V4.1 source-shared attention with explicit cache ownership."""

    swa_cache_cls = AscendDeepseekV41SWACache

    def __init__(
        self,
        vllm_config,
        config,
        max_position_embeddings=0,
        cache_config=None,
        quant_config=None,
        prefix="",
        topk_indices_buffer=None,
        reduce_results=True,
        need_gather_q_kv=False,
    ):
        layer_idx = int(prefix.split(".")[-2])
        topology = build_layer_plan(config)
        role = topology.layer(layer_idx)
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=reduce_results,
            need_gather_q_kv=need_gather_q_kv,
            use_yarn=role.has_long_context,
        )
        block_size = vllm_config.cache_config.block_size
        self.role = role
        self.topology = topology
        self.shared_state = None
        self.prefix = prefix
        width = config.head_dim
        self.softmax_scale = width**-0.5
        if role.is_kv_source:
            self.long_kv_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.long_kv_cache",
                AscendMLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=width,
                    dtype=torch.bfloat16,
                    tokens_per_state=role.compress_ratio,
                    model_version="deepseek_v41",
                    storage_block_size=block_size // role.compress_ratio,
                ),
            )
        self.compressor = (
            DeepseekV41Compressor(config, role.compress_ratio, vllm_config, f"{prefix}.compressor")
            if role.is_kv_source
            else None
        )
        self.indexer = (
            DeepseekV41Indexer(
                config,
                role.is_kv_source,
                vllm_config,
                f"{prefix}.indexer",
                role.compress_ratio,
                quant_config=quant_config,
            )
            if role.is_index_source
            else None
        )
        root = prefix.rsplit(".layers.", 1)[0]
        source = f"{root}.layers.{role.kv_source_layer}.self_attn"
        self.long_kv_source_prefix = f"{source}.long_kv_cache" if role.has_long_context else None
        self.index_k_source_prefix = f"{source}.indexer.k_cache" if role.has_long_context else None
        self.index_source_layer = role.index_source_layer
        from vllm_ascend.attention.context_parallel.dsa_v41_cp import get_v41_cp_classes

        self.v41_impl = get_v41_cp_classes()[1](
            prefix=prefix,
            role=role,
            topology=topology,
            long_kv_source_prefix=self.long_kv_source_prefix,
            index_k_source_prefix=self.index_k_source_prefix,
        )
        self.v41_layer_name = f"{prefix}.v41_attn"
        context = vllm_config.compilation_config.static_forward_context
        context[self.v41_layer_name] = self

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.dsa_v41_forward(hidden_states, output, self.v41_layer_name)
        return output


class DeepseekV41DecoderLayer(nn.Module):
    """V4.1 block with the checkpoint's delayed mHC coefficient handoff."""

    attention_cls = DeepseekV41Attention

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        super().__init__()

        if config is None:
            config = normalize_deepseek_v41_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = config.rope_parameters["original_max_position_embeddings"]
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps
        self.use_sequence_parallel_moe = parallel_config.use_sequence_parallel_moe
        self.enable_dsa_cp = enable_dsa_cp()  # TODO: delete this when enable_dsa_cp is sunset.

        attn_cls = self.attention_cls

        self.self_attn = attn_cls(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=not self.use_sequence_parallel_moe,
            need_gather_q_kv=self.use_sequence_parallel_moe and self.enable_dsa_cp,
        )

        self.mlp = DeepseekV41MoE(
            config=config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            is_draft_layer=is_draft_layer,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.use_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
        # Leave the TP partial sums for the reduce-scatter below. The mHC
        # and MoE paths then stay sharded between attention calls.
        if self.use_sequence_parallel:
            self.self_attn.wo_b.reduce_results = False
        has_engram = engram_enabled(config)
        self.engram: AscendEngram | None
        if has_engram and not is_draft_layer and self.layer_idx in config.engram_layer_ids:
            self.engram = AscendEngram(config)
        else:
            self.engram = None

    def rms_norm_cast(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize once and provide the exact FP32 routing input."""
        if enable_custom_op():
            return torch.ops._C_ascend.npu_rms_norm_cast(
                hidden_states,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
            )
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, hidden_states.float()

    @staticmethod
    def hc_collapse(x, pre_mix):
        return (pre_mix.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base, pre_mix=None):
        return torch.ops._C_ascend.npu_hc_pre_v3(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            pre_mix,
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    def hc_post(self, x, residual, post, comb):
        return torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(0),
            residual.unsqueeze(0),
            post.unsqueeze(0),
            comb.unsqueeze(0),
        ).squeeze(0)

    def forward(
        self,
        positions,
        hidden_states,
        pre_mix,
        llama_4_scaling=None,
        input_ids=None,
    ):
        use_sequence_parallel = getattr(self, "use_sequence_parallel", False)
        residual = hidden_states
        x, attn_post, attn_comb, attn_pre = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            pre_mix,
        )
        x = self.input_layernorm(x)
        if use_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]
        x = self.self_attn(positions, x, llama_4_scaling)
        if use_sequence_parallel:
            x = sp_reduce_scatter(x)
        hidden_states = self.hc_post(x, residual, attn_post, attn_comb)

        residual = hidden_states
        x, ffn_post, ffn_comb, ffn_pre = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            attn_pre,
        )
        x, x_fp32 = self.rms_norm_cast(x)
        x = self.mlp(
            x,
            input_ids=input_ids,
            hidden_states_fp32=x_fp32,
            already_sequence_parallel=use_sequence_parallel,
        )
        hidden_states = self.hc_post(x, residual, ffn_post, ffn_comb)
        return hidden_states, ffn_pre


class DeepseekV41Model(nn.Module, EagleModelMixin):
    """V4.1 backbone with delayed HC collapse and shared attention state."""

    decoder_layer_cls = DeepseekV41DecoderLayer

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()

        config = normalize_deepseek_v41_config(vllm_config.model_config.hf_config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.use_sequence_parallel_moe = vllm_config.parallel_config.use_sequence_parallel_moe

        self.vocab_size = config.vocab_size
        if hasattr(config, "index_topk"):
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        # Expose at model level so spec_decode/llm_base_proposer can share
        # this buffer with the MTP draft via attribute replacement.
        self.topk_indices_buffer = topk_indices_buffer

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: self.decoder_layer_cls(vllm_config, prefix, topk_indices_buffer=topk_indices_buffer),
            prefix=f"{prefix}.layers",
        )
        self.needs_moe_input_ids = any(
            layer.mlp.gate.tid2eid is not None or layer.mlp.gate.bias_vl is not None
            for layer in islice(self.layers, self.start_layer, self.end_layer)
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_mult = config.hc_mult
        self._mtp_hidden_buffer = None
        self.make_empty_intermediate_tensors = self._make_empty_intermediate_tensors
        self.use_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
        topology = build_layer_plan(self.config)
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        candidate_buffer = torch.full(
            (max_tokens, 1, topology.candidate_topk_blocks),
            -1,
            dtype=torch.int32,
            device=self.topk_indices_buffer.device,
        )
        self.candidate_indices_buffer = candidate_buffer
        self.shared_attention_state = DeepseekV41SharedAttentionState(
            self.topk_indices_buffer,
            candidate_buffer,
        )
        for layer in self.layers:
            if isinstance(layer, DeepseekV41DecoderLayer):
                layer.self_attn.shared_state = self.shared_attention_state
        self.engram_root = vllm_config.model_config.model
        config = self.config
        self.engram_weight_root = self.engram_root
        # The table is INT8 with group-32 scales; whether it lives in host
        # memory is vLLM's EngramConfig choice.
        cpu_offload = engram_cpu_offload(vllm_config)
        self.engram_dp_shared_memory = bool(vllm_config.engram_config and vllm_config.engram_config.dp_shared_memory)
        self.engram_layout = EngramLayout.from_config(config) if engram_enabled(config) else None
        if self.engram_layout is not None:
            # Complete head buckets per rank, laid out over TP and the
            # node-local EDP group (upstream's, not one built from EP hosts).
            # Fail on an unreadable checkpoint before the first table exists:
            # the allocation below is per-rank 24-51 GiB, and discovering a
            # missing index/key during weight iteration would mean paying for
            # it first.  `dummy` reads no checkpoint at all.
            if vllm_config.load_config.load_format != "dummy":
                preflight_engram_checkpoint(
                    self.engram_weight_root, config.engram_layer_ids, AscendParallelEngramEmbedding
                )
            for slot, (layer_id, rows) in enumerate(zip(config.engram_layer_ids, config.engram_num_embeddings)):
                head_sizes = tuple(size for order in self.engram_layout.primes[slot] for size in order)
                embed = AscendParallelEngramEmbedding(
                    rows,
                    config.engram_head_dim,
                    head_sizes,
                    slot,
                    cpu_offload=cpu_offload,
                    dp_shared_memory=self.engram_dp_shared_memory,
                )
                embed.bind_checkpoint(self.engram_weight_root, f"layers.{layer_id}.engram.embed.weight")
                self.layers[layer_id].engram.embed_tokens = embed
        self.engram_hash = None
        self._engram_input_buffers = None
        self._engram_max_tokens = max(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        self.register_buffer("engram_rotation", torch.eye(32), persistent=False)
        if engram_enabled(config):
            if vllm_config.load_config.load_format != "dummy":
                with torch.device("cpu"):
                    with safe_open(Path(self.engram_root) / "optional/quarot.safetensors", framework="pt") as file:
                        rotation = file.get_tensor("global_rotation")
                    block = rotation[:32, :32].contiguous()
                self.engram_rotation.copy_(block)
            # Upstream owns the n-gram history; the adapter only hands it the
            # Ascend SWA slot metadata (see engram/hash_state.py).
            swa_cache_layer = self.layers[config.engram_layer_ids[0]].self_attn.dsa_attn.swa_cache_layer
            self.engram_hash = create_engram_hash_state(vllm_config, config, swa_cache_layer)

        self.decoder_replay_start = self.end_layer
        self.decoder_replay_layers = None
        if getattr(vllm_config.cache_config, "swa_bounded_replay", False):
            from .decoder_replay_layers import make_decoder_replay

            self.decoder_replay_layers = make_decoder_replay(self, vllm_config)

    def _make_empty_intermediate_tensors(self, batch_size, dtype, device):
        return IntermediateTensors(
            {
                "hidden_states": torch.empty(
                    (batch_size, self.hc_mult, self.config.hidden_size), dtype=dtype, device=device
                )
            }
        )

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def prepare_engram(
        self,
        input_ids,
        positions,
        lookback_token_ids=None,
        query_start_loc=None,
        slot_mapping=None,
        block_table=None,
    ):
        """Hash on device with upstream NgramHashState, then look up head shards.

        Calls without the device metadata (dummy runs) participate with empty
        hashes. History is the current chunk, then the runner's prompt
        lookback window, then the slot cache the hash state fills itself.
        """
        config = self.config
        if not engram_enabled(config):
            return {}, torch.empty(0, dtype=torch.bool, device=positions.device)
        device = positions.device
        hash_state = self.engram_hash
        hashing = (
            hash_state is not None
            and hash_state.ensure_cache()
            and query_start_loc is not None
            # A DP dummy batch (worker.execute_dummy_batch -> _dummy_run) has
            # one token and no requests: the runner hands over a single-element
            # query_start_loc and a zero-row block table. Hashing with that
            # metadata makes the request search clamp to -1, and the hash
            # kernel then indexes one row before the block table -- an address
            # the device faults on ("GM address ... exceeds 48 bits", seen when
            # a FULL decode graph replays on an idle rank). No request rows
            # means no hash rows: fall back to the dummy-hash path below, which
            # is what an idle replica does in every other graph mode.
            and query_start_loc.numel() > 1
            and block_table is not None
            and block_table.shape[0] > 0
        )
        # A DP-sharded lookup is collective, so a replica that skips the hash
        # still has to reach it: it participates with no valid rows and no
        # history update (upstream's dummy_hashes branch). Sharing has no
        # per-step collectives, so it opts out.
        participates = hashing or (
            self.engram_hash is not None and not self.engram_dp_shared_memory and get_engram_dp_size() > 1
        )
        hashes = None
        mask = torch.empty(0, dtype=torch.bool, device=device)
        if hashing:
            assert hash_state is not None
            image_token_id = config.image_token_id
            image_pad_token_id = getattr(config, "image_pad_token_id", image_token_id + 1)
            dead = engram_dead_mask(input_ids, image_token_id, image_pad_token_id)
            if lookback_token_ids is None:
                lookback_token_ids = input_ids.new_full((query_start_loc.numel() - 1, hash_state.lookback_depth), -1)
            hashes = hash_state(
                input_ids,
                positions,
                query_start_loc,
                dead,
                lookback_token_ids,
                engram_dead_mask(lookback_token_ids, image_token_id, image_pad_token_id),
                slot_mapping,
                block_table,
            )
            # Engram.forward takes True=keep.
            mask = ~dead
        elif participates:
            assert hash_state is not None
            hashes, mask = hash_state.dummy_hashes(input_ids)
        lookups = {}
        tables = [self.layers[layer_id].engram.embed_tokens for layer_id in config.engram_layer_ids]
        if participates:
            assert hashes is not None
            # One DP gather feeds every layer sharing the split table.
            gathered = gather_engram_hashes(hashes, dp_shared_memory=self.engram_dp_shared_memory)
            for slot, (layer_id, table) in enumerate(zip(config.engram_layer_ids, tables)):
                lookups[layer_id] = table.embed_gathered(gathered[:, slot], hashes.shape[0]).flatten(1)
        else:
            for layer_id, table in zip(config.engram_layer_ids, tables):
                lookups[layer_id] = torch.empty(
                    (0, table.n_hash_cols * table.dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
        return lookups, mask.to(positions.device)

    def prepare_engram_inputs(
        self,
        input_ids,
        positions,
        padded_tokens=None,
        lookback_token_ids=None,
        query_start_loc=None,
        slot_mapping=None,
        block_table=None,
    ):
        """Synchronously refresh the rows read by this forward, before replay."""
        graph_inputs = self.prepare_engram_graph_inputs(padded_tokens)
        if not graph_inputs["engram_lookups"]:
            return graph_inputs
        num_tokens = positions.shape[0]
        output_tokens = num_tokens if padded_tokens is None else padded_tokens
        lookups, mask = self.prepare_engram(
            input_ids,
            positions,
            lookback_token_ids,
            query_start_loc,
            slot_mapping,
            block_table,
        )
        buffers = graph_inputs["engram_lookups"]
        mask_buffer = graph_inputs["engram_mask"]
        mask_buffer[: mask.numel()].copy_(mask)
        mask_buffer[mask.numel() : output_tokens].zero_()
        for layer, values in lookups.items():
            buffers[layer][: values.shape[0]].copy_(values)
            buffers[layer][values.shape[0] : output_tokens].zero_()
        return graph_inputs

    def prepare_engram_graph_inputs(self, padded_tokens=None):
        """Capture fixed-address buffers without CPU history or routing work."""
        if not engram_enabled(self.config):
            return {"engram_lookups": {}, "engram_mask": self.engram_rotation.new_empty(0, dtype=torch.bool)}
        if self._engram_input_buffers is None:
            capacity = self._engram_max_tokens
            columns = (self.config.engram_max_ngram_size - 1) * self.config.engram_n_heads
            device = self.engram_rotation.device
            self._engram_input_buffers = (
                {
                    layer: torch.zeros(
                        (capacity, columns * self.layers[layer].engram.embed_tokens.dim),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    for layer in self.config.engram_layer_ids
                },
                torch.zeros(capacity, dtype=torch.bool, device=device),
            )
        buffers, mask_buffer = self._engram_input_buffers
        return {"engram_lookups": buffers, "engram_mask": mask_buffer}

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=None,
        engram_lookups=None,
        engram_mask=None,
        lookback_token_ids=None,
    ):
        use_sequence_parallel = getattr(self, "use_sequence_parallel", False)
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        if engram_lookups is None:
            lookups, token_mask = self.prepare_engram(input_ids, positions, lookback_token_ids=lookback_token_ids)
        else:
            lookups, token_mask = engram_lookups, engram_mask
        self.shared_attention_state.reset()
        full_num_tokens = positions.shape[0]
        # Slice capacity-sized graph buffers before SP splits the token axis.
        token_mask = token_mask[:full_num_tokens]
        lookups = {layer_idx: lookup[:full_num_tokens] for layer_idx, lookup in lookups.items()}
        if use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding,
                    hidden_states,
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)
            token_mask = sp_shard(token_mask)
            lookups = {layer_idx: sp_shard(lookup) for layer_idx, lookup in lookups.items()}
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        moe_input_ids = input_ids
        if self.needs_moe_input_ids:
            moe_input_ids = torch.where(input_ids == -1, 0, input_ids)
        hidden_states, pre_mix, aux_hidden_states = self._run_layers(
            range(self.start_layer, self.decoder_replay_start),
            positions,
            hidden_states,
            pre_mix,
            moe_input_ids,
            lookups,
            token_mask,
        )
        if self.decoder_replay_layers is not None:
            hidden_states, pre_mix, *late_aux = self.decoder_replay_layers(
                hidden_states,
                pre_mix,
                positions,
                moe_input_ids,
            )
            aux_hidden_states.extend(late_aux)
        hidden_states = self.layers[self.end_layer - 1].hc_collapse(hidden_states, pre_mix)
        if use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _run_layers(self, layer_indices, positions, hidden_states, pre_mix, moe_input_ids, lookups, token_mask):
        use_sequence_parallel = self.use_sequence_parallel
        full_num_tokens = positions.shape[0]
        aux_hidden_states = []
        for layer_idx in layer_indices:
            layer = self.layers[layer_idx]
            # DSpark consumes the residual stream entering its configured
            # target layers. The runner expresses checkpoint IDs as one-based.
            if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states.mean(dim=1)
                if use_sequence_parallel:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[:full_num_tokens]
                aux_hidden_states.append(aux_hidden_state)
            if layer.engram is not None and token_mask.numel():
                n = hidden_states.shape[0]
                # Graph captures keep lookup buffers at static capacity; the
                # model's actual token dimension remains scheduler-dynamic.
                lookup = lookups[layer.layer_idx][:n]
                active_mask = token_mask[:n]
                hidden_states[:n] = layer.engram(
                    hidden_states[:n],
                    lookup,
                    active_mask,
                    self.engram_rotation,
                )
            hidden_states, pre_mix = layer(positions, hidden_states, pre_mix, None, input_ids=moe_input_ids)
        return hidden_states, pre_mix, aux_hidden_states

    def _run_replay_layers(self, hidden_states, pre_mix, positions, input_ids):
        hidden_states, pre_mix, aux = self._run_layers(
            range(self.decoder_replay_start, self.end_layer),
            positions,
            hidden_states,
            pre_mix,
            input_ids,
            {},
            None,
        )
        return hidden_states, pre_mix, *aux

    def _new_replay_outputs(self, hidden_states, pre_mix, positions, input_ids):
        late_aux = (
            hidden_states.new_zeros((positions.shape[0], self.config.hidden_size))
            for layer_id in self.aux_hidden_state_layers
            if layer_id > self.decoder_replay_start
        )
        return torch.zeros_like(hidden_states), torch.zeros_like(pre_mix), *late_aux


class AscendDeepseekV41LLMForCausalLM(nn.Module, DeepseekV41MixtureOfExperts, SupportsPP, SupportsLoRA, SupportsEagle3):
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    model_cls = DeepseekV41Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = normalize_deepseek_v41_config(vllm_config.model_config.hf_config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = self.model_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        # Set MoE hyperparameters
        self.num_moe_layers = self.config.num_hidden_layers
        self.set_moe_parameters()
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

        self.moe_comm_methods = {kind: get_moe_comm_method(kind) for kind in MoECommType}

    requires_raw_input_tokens = True
    _DEFERRED_WEIGHT_MARKERS: tuple[str, ...] = ()
    _DEFERRED_WEIGHT_PREFIXES = ("aligner.", "vision.", "image_", "mtp.")

    def prepare_engram_inputs(
        self,
        input_ids,
        positions,
        padded_tokens=None,
        lookback_token_ids=None,
        query_start_loc=None,
        slot_mapping=None,
        block_table=None,
    ):
        return self.model.prepare_engram_inputs(
            input_ids,
            positions,
            padded_tokens,
            lookback_token_ids,
            query_start_loc,
            slot_mapping,
            block_table,
        )

    def prepare_engram_graph_inputs(self, padded_tokens=None):
        return self.model.prepare_engram_graph_inputs(padded_tokens)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        engram_lookups=None,
        engram_mask=None,
        lookback_token_ids=None,
    ):
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            engram_lookups=engram_lookups,
            engram_mask=engram_mask,
            lookback_token_ids=lookback_token_ids,
        )

    @property
    def decoder_replay_layers(self):
        return self.model.decoder_replay_layers

    @property
    def token_lookback_depth(self) -> int:
        """Tokens before a chunk start the engram hash needs; the runner passes
        them as ``lookback_token_ids``."""
        engram_hash = self.model.engram_hash
        return engram_hash.lookback_depth if engram_hash is not None else 0

    @classmethod
    def _is_milestone_weight(cls, name):
        return not name.startswith(cls._DEFERRED_WEIGHT_PREFIXES) and not any(
            marker in name for marker in cls._DEFERRED_WEIGHT_MARKERS
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if not engram_enabled(self.model.config):
            return self._load_model_weights((name, tensor) for name, tensor in weights if ".engram." not in name)
        loaded = self._load_model_weights((name, tensor) for name, tensor in weights if self._is_milestone_weight(name))
        return loaded

    def set_moe_parameters(self):
        self.expert_weights = []

        self.num_expert_groups = getattr(self.config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            if isinstance(layer.mlp, DeepseekV41MoE):
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

    def _load_model_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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
            if name.endswith(".engram.embed.scale"):
                name = name.removesuffix(".scale") + ".weight_scale_inv"
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

    @property
    def engram_cache_layer_name(self) -> str | None:
        if not engram_enabled(self.model.config):
            return None
        return self.model.layers[0].self_attn.dsa_attn.swa_cache_layer.prefix
