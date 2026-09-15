# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 text model and source-shared hybrid-cache graph."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import vllm.envs as envs
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3, SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import PPMissingLayer, make_layers, maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.dsa_attn_kv_plan import get_dsv4_attn_kv_dtype
from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheLayer,
)
from vllm_ascend.config_utils import normalize_deepseek_v41_config
from vllm_ascend.core.deepseek_v41_kv_cache import (
    DeepseekV41FullSpec,
    DeepseekV41SWASpec,
    validate_cache_runtime,
)
from vllm_ascend.models.common.deepseek import DeepseekForCausalLMMixin, DeepseekMoE, init_attention_projections
from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm_ascend.ops.dsa import AscendDeepseekSparseAttention, DSAModules
from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding
from vllm_ascend.utils import enable_custom_op, enable_dsa_cp

from .compressor import DeepseekV41Compressor, _read, text_config_of
from .engram_gate import engram_gate
from .engram_hash import PagedNgramHistory, engram_history_metadata
from .engram_hbm import EngramQueryGroup, NodeShardedEngram
from .indexer import DeepseekV41Indexer


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
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    candidate_source_layer: int
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


def _as_int_tuple(config: Any, name: str) -> tuple[int, ...]:
    value = _read(config, name)
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"DeepSeek V4.1 {name} must be a list of integers")
    return tuple(value)


def _latest_source(layer_idx: int, sources: tuple[int, ...]) -> int | None:
    return next((source for source in reversed(sources) if source <= layer_idx), None)


def build_layer_plan(config: Any) -> DeepseekV41Topology:
    """Build and validate the V4.1 layer-sharing graph from a text config.

    ``config`` may be a Transformers config object or the raw ``text_config``
    dictionary.  Extra compression ratios for speculative layers are allowed,
    but only the first ``num_hidden_layers`` entries describe the backbone.
    """

    config = text_config_of(config)
    num_layers = int(_read(config, "num_hidden_layers"))
    ratios = _as_int_tuple(config, "compress_ratios")
    kv_sources = _as_int_tuple(config, "kv_source_layers")
    index_sources = _as_int_tuple(config, "index_source_layers")
    engram_layers = _as_int_tuple(config, "engram_layer_ids")
    candidate_source = int(_read(config, "candidate_source_layer"))
    candidate_topk_blocks = int(_read(config, "candidate_topk_blocks"))
    candidate_block_size = int(_read(config, "candidate_block_size"))
    index_topk = int(_read(config, "index_topk"))

    if num_layers <= 0:
        raise ValueError("DeepSeek V4.1 num_hidden_layers must be positive")
    if len(ratios) < num_layers:
        raise ValueError(
            "DeepSeek V4.1 compress_ratios must cover every backbone layer: "
            f"got {len(ratios)} ratios for {num_layers} layers"
        )
    ratios = ratios[:num_layers]
    if any(ratio not in (0, 1, 2) for ratio in ratios):
        raise ValueError(f"DeepSeek V4.1 backbone only supports compression ratios 0, 1 and 2; got {ratios}")

    for name, sources in (("kv_source_layers", kv_sources), ("index_source_layers", index_sources)):
        if tuple(sorted(set(sources))) != sources:
            raise ValueError(f"DeepSeek V4.1 {name} must be sorted and unique")
        if any(source < 0 or source >= num_layers for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} contains a layer outside the backbone")
        if any(ratios[source] == 0 for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} cannot point to a local-only layer")

    if not set(kv_sources).issubset(index_sources):
        raise ValueError("Every DeepSeek V4.1 KV source must also be an index source")
    if candidate_source not in kv_sources:
        raise ValueError("DeepSeek V4.1 candidate_source_layer must be a KV source")
    if candidate_topk_blocks <= 0 or candidate_block_size <= 0 or index_topk <= 0:
        raise ValueError("DeepSeek V4.1 candidate and index TopK values must be positive")
    if len(set(engram_layers)) != len(engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids must be unique")
    if any(layer < 0 or layer >= num_layers for layer in engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids contains a layer outside the backbone")

    engram_slots = {layer_idx: slot for slot, layer_idx in enumerate(engram_layers)}
    roles: list[DeepseekV41LayerRole] = []
    for layer_idx, ratio in enumerate(ratios):
        kv_source = _latest_source(layer_idx, kv_sources) if ratio else None
        index_source = _latest_source(layer_idx, index_sources) if ratio else None
        if ratio and (kv_source is None or index_source is None):
            raise ValueError(f"DeepSeek V4.1 layer {layer_idx} has long-context attention but no source layer")
        if kv_source is not None and ratios[kv_source] != ratio:
            raise ValueError(
                f"DeepSeek V4.1 layer {layer_idx} has ratio {ratio}, but its KV source "
                f"layer {kv_source} has ratio {ratios[kv_source]}"
            )

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
        kv_source_layers=kv_sources,
        index_source_layers=index_sources,
        candidate_source_layer=candidate_source,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        index_topk=index_topk,
    )


class AscendDeepseekV41SWACache(DeepseekV41CacheLayer):
    """Ascend SWA cache registered with the V4.1 allocator."""

    def __init__(self, head_dim, window_size, dtype, prefix, cache_config):
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        block_size = DSV4_BLOCK_SIZES[cache_config.block_size][0][1]
        spec = DeepseekV41SWASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=head_dim,
            dtype=dtype,
            sliding_window=window_size,
            cache_dtype_str=cache_config.cache_dtype,
            model_version="deepseek_v4",
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
            original_seq_len=max_position_embeddings if use_yarn else 0,
            rope_groups=["default"],
        )
        k_dtype = get_dsv4_attn_kv_dtype(vllm_config)
        swa_cache_layer = self.swa_cache_cls(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=k_dtype,
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
        config = text_config_of(config)
        validate_cache_runtime(vllm_config)
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
        if block_size <= 0 or block_size % 2:
            raise ValueError("V4.1 logical block_size must be a positive multiple of two")
        owned: list[str] = []
        if role.is_kv_source:
            owned.extend((f"{prefix}.long_kv_cache", f"{prefix}.indexer.k_cache"))
            if role.compress_ratio == 2:
                owned.append(f"{prefix}.compressor.state_cache")
        duplicates = set(owned) & vllm_config.compilation_config.static_forward_context.keys()
        if duplicates:
            raise ValueError(f"Duplicate V4.1 cache prefixes: {sorted(duplicates)}")
        self.role = role
        self.topology = topology
        self.shared_state = None
        self.prefix = prefix
        width = _read(config, "head_dim")
        self.softmax_scale = width**-0.5
        if role.is_kv_source:
            self.long_kv_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.long_kv_cache",
                DeepseekV41FullSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=width,
                    dtype=torch.bfloat16,
                    tokens_per_state=role.compress_ratio,
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
        if self.v41_layer_name in context:
            raise ValueError(f"Duplicate V4.1 attention layer: {self.v41_layer_name}")
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

        self.mlp = DeepseekMoE(
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
        engram_enabled = get_ascend_config().enable_engram
        if engram_enabled and not is_draft_layer and self.layer_idx in config.engram_layer_ids:
            self.engram = torch.nn.Module()
            self.engram.wkv = torch.nn.Linear(
                (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim,
                (config.hc_mult + 1) * config.hidden_size,
                bias=False,
                dtype=torch.bfloat16,
            )
            self.engram.q_weight = torch.nn.Parameter(
                torch.empty(config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
            )
            self.engram.k_weight = torch.nn.Parameter(
                torch.empty(config.hc_mult, config.hidden_size, dtype=torch.bfloat16)
            )
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
        if (
            get_ascend_config().enable_engram
            and vllm_config.load_config.load_format != "dummy"
            and vllm_config.load_config.safetensors_load_strategy != "lazy"
        ):
            raise ValueError("Engram HBM shards require --safetensors-load-strategy lazy")
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
        # Target storage is a loader/runtime choice.  Checkpoint metadata is
        # used only by load_checkpoint to validate the source representation.
        # Read the storage choice after AscendConfig validation.
        ascend_config = get_ascend_config()
        storage_format = ascend_config.engram_storage
        if ascend_config.enable_engram:
            query_group = EngramQueryGroup.from_vllm(vllm_config.parallel_config)
            for layer_id, rows in zip(config.engram_layer_ids, config.engram_num_embeddings):
                self.layers[layer_id].engram.embed = NodeShardedEngram(
                    rows,
                    config.engram_head_dim,
                    query_group,
                    storage_format=storage_format,
                )
        self.engram_history = None
        self._engram_input_buffers = None
        self._engram_max_tokens = max(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        self.register_buffer("engram_rotation", torch.eye(32), persistent=False)
        if ascend_config.enable_engram and vllm_config.load_config.load_format != "dummy":
            with torch.device("cpu"):
                tokenizer = AutoTokenizer.from_pretrained(self.engram_root)
                self.engram_history = PagedNgramHistory(config, tokenizer)
                with safe_open(Path(self.engram_root) / "optional/quarot.safetensors", framework="pt") as file:
                    rotation = file.get_tensor("global_rotation")
                block = rotation[:32, :32].contiguous()
                if not torch.equal(rotation, torch.block_diag(*[block] * (config.hidden_size // 32))):
                    raise ValueError("Engram gate requires repeated block32 global rotation")
            self.engram_rotation.copy_(block)

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

    def prepare_engram(self, input_ids, positions):
        """Eager boundary: every DP participates, including metadata-free dummies."""
        config = self.config
        if not get_ascend_config().enable_engram:
            return {}, torch.empty(0, dtype=torch.bool, device=positions.device)
        columns = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        hashes = torch.empty((0, len(config.engram_layer_ids), columns), dtype=torch.int64, device="cpu")
        mask = torch.empty(0, dtype=torch.bool, device="cpu")
        metadata = get_forward_context().attn_metadata
        if metadata is not None and self.engram_history is not None:
            first = self.layers[0].self_attn.dsa_attn.swa_cache_layer
            meta = metadata[first.prefix]
            boundaries, block_table, block_size = engram_history_metadata(meta)
            n = int(boundaries[-1])
            requests = torch.repeat_interleave(torch.arange(len(boundaries) - 1, device="cpu"), boundaries.diff())
            hashes, mask = self.engram_history.update(
                input_ids[:n].cpu().long(),
                positions[:n].cpu().long(),
                requests,
                block_table,
                block_size,
            )
        lookups = {}
        tables = [self.layers[layer_id].engram.embed for layer_id in config.engram_layer_ids]
        ids_list = [hashes[:, slot] for slot in range(len(tables))]
        if hasattr(tables[0], "route_many"):
            routed = tables[0].route_many(tables, ids_list)
        else:
            routed = [table(ids) for table, ids in zip(tables, ids_list)]
        for layer_id, values in zip(config.engram_layer_ids, routed):
            lookups[layer_id] = values.flatten(1)
        return lookups, mask.to(positions.device)

    def prepare_engram_inputs(self, input_ids, positions, padded_tokens=None):
        """Refresh persistent inputs before main-model capture or replay."""
        lookups, mask = self.prepare_engram(input_ids, positions)
        num_tokens = positions.shape[0]
        # The compiled V4.1 backbone uses the scheduler's static token
        # capacity for decode graphs (typically max_num_batched_tokens), even
        # when the current request has one token.  Keep lookup tensors at that
        # capacity so every captured graph sees the same Engram shape.
        output_tokens = max(self._engram_max_tokens, padded_tokens or 0)
        if output_tokens < num_tokens:
            raise ValueError("Engram padded token count is smaller than the input")
        if self._engram_input_buffers is None:
            capacity = self._engram_max_tokens
            self._engram_input_buffers = (
                {layer: values.new_zeros((capacity, values.shape[1])) for layer, values in lookups.items()},
                mask.new_zeros(capacity),
            )
        buffers, mask_buffer = self._engram_input_buffers
        padded_mask = mask_buffer[:output_tokens]
        padded_mask.zero_()
        padded_mask[: mask.numel()].copy_(mask)
        padded_lookups = {}
        for layer, values in lookups.items():
            padded = buffers[layer][:output_tokens]
            padded.zero_()
            padded[: values.shape[0]].copy_(values)
            padded_lookups[layer] = padded
        return {"engram_lookups": padded_lookups, "engram_mask": padded_mask}

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=None,
        engram_lookups=None,
        engram_mask=None,
    ):
        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
            raise NotImplementedError("V4.1 eager milestone currently requires PP=1")
        use_sequence_parallel = getattr(self, "use_sequence_parallel", False)
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        if engram_lookups is None:
            lookups, token_mask = self.prepare_engram(input_ids, positions)
        else:
            lookups, token_mask = engram_lookups, engram_mask
        self.shared_attention_state.reset()
        full_num_tokens = positions.shape[0]
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
        last_layer = None
        aux_hidden_states = []
        moe_input_ids = input_ids
        if self.needs_moe_input_ids:
            moe_input_ids = torch.where(input_ids == -1, 0, input_ids)
        for layer in self.layers:
            last_layer = layer
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
                kv = layer.engram.wkv(lookup)
                key, value = kv.split([self.hc_mult * self.config.hidden_size, self.config.hidden_size], -1)
                hidden_states[:n] = engram_gate(
                    hidden_states[:n],
                    key.view(n, self.hc_mult, self.config.hidden_size),
                    value,
                    layer.engram.q_weight.float() * layer.engram.k_weight.float(),
                    self.engram_rotation,
                    active_mask,
                    self.config.rms_norm_eps,
                )
            hidden_states, pre_mix = layer(positions, hidden_states, pre_mix, None, input_ids=moe_input_ids)
        assert last_layer is not None
        hidden_states = last_layer.hc_collapse(hidden_states, pre_mix)
        if use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class AscendDeepseekV41LLMForCausalLM(nn.Module, DeepseekForCausalLMMixin, SupportsPP, SupportsLoRA, SupportsEagle3):
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

    def prepare_engram_inputs(self, input_ids, positions, padded_tokens=None):
        return self.model.prepare_engram_inputs(input_ids, positions, padded_tokens)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        engram_lookups=None,
        engram_mask=None,
    ):
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            engram_lookups=engram_lookups,
            engram_mask=engram_mask,
        )

    @classmethod
    def _is_milestone_weight(cls, name):
        return not name.startswith(cls._DEFERRED_WEIGHT_PREFIXES) and not any(
            marker in name for marker in cls._DEFERRED_WEIGHT_MARKERS
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if not get_ascend_config().enable_engram:
            return super().load_weights((name, tensor) for name, tensor in weights if ".engram." not in name)
        engram_loaded: set[str] = set()

        def milestone_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if ".engram." in name:
                    # Bypass V4's generic embed -> embed_tokens remapping and TP loader.
                    local_name = name.removeprefix("model.")
                    # FP8/MXFP8 Engram scales are consumed by the CPU loader.
                    if local_name.endswith(".engram.embed.scale"):
                        continue
                    parameter_name = "model." + local_name
                    if local_name.endswith(".engram.embed.weight"):
                        layer_id = int(local_name.split(".")[1])
                        self.model.layers[layer_id].engram.embed.load_checkpoint(self.model.engram_root, local_name)
                    else:
                        param = self.get_parameter(parameter_name)
                        if tensor.dtype != torch.bfloat16 or tensor.shape != param.shape:
                            raise ValueError(f"Unexpected BF16 Engram parameter: {name}")
                        param.data.copy_(tensor)
                    engram_loaded.add(parameter_name)
                elif self._is_milestone_weight(name):
                    yield name, tensor

        loaded = super().load_weights(milestone_weights())
        expected = {name for name, _ in self.named_parameters() if ".engram." in name}
        if engram_loaded != expected:
            raise ValueError(f"Missing Engram weights: {expected - engram_loaded}")
        return loaded | engram_loaded
