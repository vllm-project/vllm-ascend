# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Eager text execution for HY4 Flash on Ascend.

The model-specific constructors adapt the vLLM HY4 Flash NVIDIA modules to
Ascend while retaining their checkpoint layout and inherited weight loaders.
"""

from copy import deepcopy
from typing import Any

import torch
import torch_npu
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import make_attention_sink
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.hy_v4.nvidia.hc import HYV4HCLayer
from vllm.models.hy_v4.nvidia.mm_preprocess import HYV4VLDummyInputsBuilder, HYV4VLMultiModalProcessor
from vllm.models.hy_v4.nvidia.model import HYV4ForCausalLM, HYV4Model
from vllm.models.hy_v4.nvidia.vl_model import HYV4VLForConditionalGeneration
from vllm.models.hy_v4_flash.configs import SLIDING_ATTENTION, HYV4FlashConfig
from vllm.models.hy_v4_flash.nvidia.attention import HYV4FlashAttention
from vllm.models.hy_v4_flash.nvidia.engram import FlashEmbedding, FlashEngram, FlashNgramHashState
from vllm.models.hy_v4_flash.nvidia.model import (
    HYV4FlashDecoderLayer,
    HYV4FlashForCausalLM,
    HYV4FlashModel,
)
from vllm.models.hy_v4_flash.nvidia.moe import HYV4FlashFeedForward, HYV4FlashMoE
from vllm.models.hy_v4_flash.nvidia.vl_model import (
    HYV4FlashVLForConditionalGeneration,
    HYV4FlashVLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionState,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import attention_transfer_window
from vllm_ascend.models.deepseek_v41.engram.hash_state import AscendNgramHashState
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.models.hy4_flash.long_context import (
    AscendHYV4FlashIndexPool,
    absorb_kv_projection,
    make_paged_cache,
    run_paged_sparse_attention,
)
from vllm_ascend.worker.v2.model_states.default import AscendModelState


def _dense_flash_limit(config):
    # One IndexPool group is reserved for the current, incomplete tail.
    pool_limit = (config.compressed_indexer_group_topk - 1) * config.compressed_indexer_compress_ratio
    return min(config.sliding_window, pool_limit, config.index_topk)


def validate_ascend_hy4_flash(vllm_config: VllmConfig) -> None:
    config = vllm_config.model_config
    text = config.hf_text_config
    text.validate_flash_config()
    text.verify_parallel_config(vllm_config.parallel_config)
    if current_platform.device_type != "npu":
        raise ValueError("Ascend HY4 Flash requires an NPU.")
    if config.dtype != torch.bfloat16:
        raise ValueError("Ascend HY4 Flash requires BF16 compute.")
    if vllm_config.kv_transfer_config is not None or vllm_config.lora_config is not None:
        raise ValueError("Ascend HY4 Flash does not support KV transfer or LoRA.")
    if text.index_topk % 128 or text.index_topk < text.sliding_window:
        raise ValueError("index_topk must be 128-aligned and cover sliding_window.")
    if text.engram_layer_ids and not text.enable_ihc:
        raise ValueError("Flash Engram requires the iHC residual stream.")
    quantization = vllm_config.model_config.quantization
    if quantization == "compressed-tensors":
        from vllm.models.hy_v4.dfly_config import _validate_checkpoint_quantization

        _validate_checkpoint_quantization(vllm_config.model_config.model_arch_config.quantization_config)
    elif quantization == "fp8":
        fp8_quant = vllm_config.quant_config
        if list(getattr(fp8_quant, "weight_block_size", None) or []) != [128, 128]:
            raise ValueError("HYV4 Flash requires FP8 128x128 weight blocks")
    elif quantization == "online":
        from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8Static128BlockSym,
        )

        quant = vllm_config.model_config.quantization_config
        if isinstance(quant, dict):
            quant = QuantizationConfigArgs(**quant)
        if quant != QuantizationConfigArgs(moe=QuantSpec(weight=kFp8Static128BlockSym)):
            raise ValueError("Flash online quantization requires MoE-only fp8_per_block")
    elif quantization is not None:
        raise ValueError("HYV4 Flash supports blockwise FP8 or compressed-tensors MXFP4")
    if config.max_model_len > _dense_flash_limit(text) and vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("Ascend HY4 Flash long-context execution requires disabling prefix caching.")
    if (
        not config.enforce_eager
        or vllm_config.speculative_config
        or vllm_config.parallel_config.data_parallel_size != 1
        or vllm_config.parallel_config.prefill_context_parallel_size != 1
        or vllm_config.parallel_config.decode_context_parallel_size != 1
    ):
        raise ValueError("Ascend HY4 Flash requires eager execution, DP=PCP=DCP=1 and no speculative decoding.")
    if config.is_multimodal_model and any(
        config.get_multimodal_config().get_limit_per_prompt(m) for m in ("image", "video")
    ):
        raise ValueError("Ascend HY4 Flash currently requires image and video limits of zero.")
    if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("Ascend HY4 Flash requires BF16 KV cache.")
    if not vllm_config.engram_config or not vllm_config.engram_config.cpu_offload:
        raise ValueError("Ascend HY4 Flash requires CPU-offloaded Engram tables.")


class AscendHYV4FlashAttentionImpl(AscendAttentionBackendImpl):
    def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
        with attention_transfer_window():
            key, value, block_size, block_table, seq_lens = self._get_fia_params(key, value, attn_metadata, kv_cache)
            query_ends = attn_metadata.actual_seq_lengths_q
            num_tokens = query_ends[-1]
            is_decode = attn_metadata.attn_state == AscendAttentionState.DecodeOnly and attn_metadata.max_query_len == 1
            if block_table is None:
                key = key[:num_tokens]
                value = value[:num_tokens]
            attended, lse = torch_npu.npu_fused_infer_attention_score_v2(
                query[:num_tokens].contiguous(),
                key.contiguous(),
                value.contiguous(),
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                actual_seq_qlen=query_ends,
                actual_seq_kvlen=seq_lens,
                atten_mask=None if is_decode else attn_metadata.attn_mask,
                sparse_mode=0 if is_decode else 3,
                block_table=block_table,
                block_size=block_size,
                softmax_scale=self.scale,
                return_softmax_lse=True,
            )
            # A zero-value sink with logit s rescales the attention output by
            # exp(LSE) / (exp(LSE) + exp(s)). FIA cannot take a sink at D=256.
            sink_weight = torch.sigmoid(lse - self.sinks.view(1, -1, 1))
            output[:num_tokens].copy_(attended.float() * sink_weight)
            return output


class AscendHYV4FlashAttentionBackend(AscendAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[AscendHYV4FlashAttentionImpl]:
        return AscendHYV4FlashAttentionImpl


class AscendHYV4FlashAttention(HYV4FlashAttention):
    def __init__(self, vllm_config, config, layer_idx, _topk_indices_buffer, prefix):
        nn.Module.__init__(self)
        self.is_window_layer = config.layer_types[layer_idx] == SLIDING_ATTENTION
        self.config = deepcopy(config)
        self.config.rope_parameters = config.rope_for_layer(layer_idx)
        self.hidden_size = config.hidden_size
        self.layer_idx = self.layer_id = layer_idx
        self.prefix = prefix
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.max_position_embeddings = vllm_config.model_config.max_model_len
        self.num_heads = config.swa_num_attention_heads if self.is_window_layer else config.num_attention_heads
        self.num_local_heads = self.num_heads // get_tensor_model_parallel_world_size()
        self.qk_nope_head_dim = config.swa_qk_nope_head_dim if self.is_window_layer else config.qk_nope_head_dim
        self.qk_rope_head_dim = config.swa_qk_rope_head_dim if self.is_window_layer else config.qk_rope_head_dim
        self.v_head_dim = config.swa_v_head_dim if self.is_window_layer else config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.qk_head_dim**-0.5
        self.gated_mla = config.gated_mla
        self.learnable_sink = config.learnable_sink
        if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
            raise ValueError("Flash sparse MLA requires latent512 + rotary64 KV")
        if not config.learnable_sink or not config.gated_mla or config.gating_type != "elementwise":
            raise ValueError("Flash requires learnable sinks and elementwise gated MLA")
        if any(getattr(config, name, None) is not None for name in ("gate_lora_rank", "o_lora_rank")):
            raise ValueError("Flash requires direct gate/output projections")
        if (
            getattr(config, "gate_groups", 1) != 1
            or getattr(config, "o_groups", 1) != 1
            or getattr(config, "full_nope", False)
            or not getattr(config, "qk_norm", True)
        ):
            raise ValueError("Flash requires normalized ungrouped RoPE MLA")
        quant = vllm_config.quant_config
        self.fused_qkv_a_proj = None
        self.q_a_proj = ReplicatedLinear(
            config.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_a_proj",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.linear_gate = ColumnParallelLinear(
            config.hidden_size,
            self.num_heads * self.v_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.linear_gate",
        )
        self.use_hpc_gated_mla = False
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=self.config.rope_parameters,
            is_neox_style=False,
        )
        self.learnable_sink_param = make_attention_sink(self.num_local_heads, torch.float32)
        self.long_context = self.max_position_embeddings > _dense_flash_limit(config)
        if self.long_context:
            self.attn = make_paged_cache(vllm_config, f"{prefix}.attn", self.kv_lora_rank, self.qk_rope_head_dim)
            self.register_buffer("_key_projection", None, persistent=False)
            self.register_buffer("_value_projection", None, persistent=False)
            if not self.is_window_layer:
                self.indexer = AscendHYV4FlashIndexPool(vllm_config, self.config, f"{prefix}.indexer")
        else:
            self.attn = Attention(
                self.num_local_heads,
                self.qk_head_dim,
                self.scaling,
                num_kv_heads=self.num_local_heads,
                head_size_v=self.v_head_dim,
                cache_config=vllm_config.cache_config,
                prefix=f"{prefix}.attn",
                attn_backend=AscendHYV4FlashAttentionBackend,
                sinks=self.learnable_sink_param,
            )
        self.post_proj_norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps)
            if config.sandwich_norm and (self.is_window_layer or not config.dsa_skip_attn_post_norm)
            else None
        )

    def process_weights_after_loading(self) -> None:
        if self.long_context:
            absorb_kv_projection(self, self.attn.act_dtype)

    def _indexer_and_attn(self, hidden_states, q_c, positions, q, kv_c_normed, k_pe, out) -> None:
        if self.long_context:
            run_paged_sparse_attention(self, hidden_states, q_c, positions, q, kv_c_normed, k_pe, out)
            return
        del hidden_states, q_c, positions
        # Expand the latent into its two per-head projections. Attention over
        # the expanded keys and values is equivalent to absorbed MLA.
        kv = self.kv_b_proj(kv_c_normed)[0].reshape(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value = kv.split((self.qk_nope_head_dim, self.v_head_dim), -1)
        key = torch.cat((k_nope, k_pe.expand(-1, self.num_local_heads, -1)), dim=-1)
        out.copy_(self.attn(q, key, value))


class AscendFlashNgramHashState(FlashNgramHashState):
    def prepare_hashes(self, input_ids, lookback, metadata):
        tokens = input_ids.numel()
        if metadata is None or tokens == 0 or metadata.num_actual_tokens == 0 or metadata.seq_lens_gpu.numel() == 0:
            return input_ids.new_full(
                (len(self.layout.layer_ids), tokens, self.offsets.shape[1]),
                -1,
                dtype=torch.int32,
            )
        if self.layout.lookup_table is None:
            raise ValueError("Missing checkpoint Engram token map")
        if lookback is None or lookback.shape[1] != self.lookback_depth:
            raise ValueError("Flash Engram requires V2 lookback_token_ids")
        starts = metadata.query_start_loc
        rows = torch.arange(tokens, device=input_ids.device)
        requests = torch.searchsorted(starts[1:], rows, right=True)
        requests = requests.clamp_max(starts.numel() - 2).long()
        logical = (
            metadata.seq_lens_gpu[requests] - (starts[requests + 1] - starts[requests]) + rows - starts[requests]
        ).long()
        dead = (input_ids < 0) | (input_ids >= self.token_map.numel())
        safe_ids = input_ids.clamp(0, self.token_map.numel() - 1)
        lookback_dead = lookback >= self.token_map.numel()
        safe_lookback = lookback.clamp(-1, self.token_map.numel() - 1)
        hashes = AscendNgramHashState.forward(
            self,
            safe_ids,
            logical,
            starts,
            dead,
            safe_lookback,
            lookback_dead,
        )
        offsets = self.offsets.unsqueeze(0)
        primes = self.primes.flatten(1).unsqueeze(0)
        hashes = torch.where(hashes < offsets, hashes + primes, hashes).int()
        no_hash = dead | (rows >= metadata.num_actual_tokens) | (rows >= starts[-1])
        no_hash |= input_ids == self.layout.pad_token_id
        for token_id in self.visual_token_ids:
            no_hash |= input_ids == token_id
        hashes.masked_fill_(no_hash[:, None, None], -1)
        return hashes.permute(1, 0, 2).contiguous()


class AscendFlashEmbedding(FlashEmbedding):
    def forward(self, hashes):
        weight = self.embedding.weight
        indices = hashes[:, self.head_start : self.head_end].to(device=weight.device, dtype=torch.long)
        local = indices - self.vocab_start
        valid = (local >= 0) & (local < weight.shape[0])
        safe = local.masked_fill(~valid, 0)
        if self.fp8:
            rows = weight.view(torch.uint8)[safe].view(torch.float8_e4m3fn).float()
            rows = rows * self.embedding.weight_scale_inv[safe].float().unsqueeze(-1)
        else:
            rows = weight[safe].float()
        return rows.masked_fill(~valid.unsqueeze(-1), 0).to(hashes.device)


class AscendFlashEngram(FlashEngram):
    def __init__(self, vllm_config, layout, layer_id, prefix):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        index = layout.layer_ids.index(layer_id)
        sizes = [p for order in layout.primes[index] for p in order]
        storage = vllm_config.engram_config
        if storage.disk_offload or storage.embedding_across_dp or storage.dp_shared_memory:
            raise ValueError(
                "Flash Engram requires private DP replicas with TP-local tables; "
                "DP sharing, cross-DP sharding and disk offload are unsupported"
            )
        dtype = None
        if vllm_config.load_config.load_format == "dummy":
            dtype = torch.float8_e4m3fn if config.engram_fp8_embedding else torch.bfloat16
        self.multi_head_embedding = AscendFlashEmbedding(sizes, layout.head_dim, dtype, storage.cpu_offload)
        self.hidden_size, self.hc_mult = config.hidden_size, config.hc_mult
        local = self.multi_head_embedding.head_end - self.multi_head_embedding.head_start
        self.lookup_per_head_norms = nn.ModuleList([nn.RMSNorm(layout.head_dim) for _ in range(local)])
        self.value_proj = nn.Linear(local * layout.head_dim, self.hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(local * layout.head_dim, self.hidden_size) for _ in range(self.hc_mult)]
        )
        self.norm_key = nn.ModuleList([nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)])
        self.norm_query = nn.ModuleList([nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)])
        self.unified_pre_gamma = nn.Parameter(torch.empty(self.hidden_size))
        self.unified_per_hc_scales = nn.ParameterList(
            [nn.Parameter(torch.empty(self.hidden_size)) for _ in range(self.hc_mult)]
        )
        self.total_width = len(sizes) * layout.head_dim
        self.start = self.multi_head_embedding.head_start * layout.head_dim
        self.stop = self.multi_head_embedding.head_end * layout.head_dim
        self.rank = get_tensor_model_parallel_rank()
        for projection in [self.value_proj, *self.key_projs]:
            set_weight_attrs(projection.weight, {"weight_loader": self._load_projection})
            set_weight_attrs(projection.bias, {"weight_loader": self._load_bias})


class AscendHYV4FlashDecoderLayer(HYV4FlashDecoderLayer):
    def __init__(
        self,
        config: HYV4FlashConfig,
        vllm_config: VllmConfig,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        ihc_sequence_parallel: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.enable_ihc = config.enable_ihc
        assert not ihc_sequence_parallel
        self.defer_reduction = False
        self.self_attn = AscendHYV4FlashAttention(
            vllm_config,
            config,
            self.layer_idx,
            topk_indices_buffer,
            f"{prefix}.self_attn",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        dense = (config.mlp_layer_types or ())[self.layer_idx] == "dense"
        mlp_cls = HYV4FlashFeedForward if dense else HYV4FlashMoE
        self.mlp = mlp_cls(config, quant_config, f"{prefix}.mlp")
        self.block_type = "feedforward" if dense else "moe"
        self.hc_attn_layer = HYV4HCLayer(config, self.layer_idx, prefix=f"{prefix}.hc_attn_layer")
        self.hc_mlp_layer = HYV4HCLayer(config, self.layer_idx, prefix=f"{prefix}.hc_mlp_layer")


class AscendHYV4FlashModel(HYV4FlashModel):
    decoder_layer_cls = AscendHYV4FlashDecoderLayer

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        HYV4Model.__init__(self, vllm_config=vllm_config, prefix=prefix)
        self.loaded_parts: dict[str, set[Any]] = {}
        self.engram_hash = None
        if self.config.engram_layer_ids:
            self.engram_hash = AscendFlashNgramHashState(
                self.config, dummy=vllm_config.load_config.load_format == "dummy"
            )
            for layer_id in self.config.engram_layer_ids:
                self.layers[layer_id - 1].engram_layer = AscendFlashEngram(
                    vllm_config,
                    self.engram_hash.layout,
                    layer_id,
                    f"{prefix}.layers.{layer_id - 1}.engram_layer",
                )
        self.token_lookback_depth = self.config.engram_max_ngram_size - 1 if self.engram_hash is not None else 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.engram_hash is None:
            return HYV4Model.forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)
        if input_ids is None:
            raise ValueError("Flash Engram requires original input token IDs")
        hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        hashes = input_ids.new_empty(
            (
                len(self.config.engram_layer_ids),
                input_ids.numel(),
                self.engram_hash.offsets.shape[1],
            ),
            dtype=torch.int32,
        )
        self.engram_hash.prepare_hashes_with_output(
            input_ids,
            lookback_token_ids,
            self.layers[0].self_attn.attn.layer_name,
            hashes,
        )
        hash_indices = {layer_id - 1: index for index, layer_id in enumerate(self.config.engram_layer_ids)}
        for index, layer in enumerate(self.layers):
            if index in hash_indices:
                hidden = layer.hc_attn_layer.prepare_input(hidden)
                hidden = layer.engram_layer(hidden, hashes[hash_indices[index]])
            hidden, _ = layer(positions, hidden, None)
        return self.norm(self.hc_head(hidden))

    def load_weights(self, weights):
        if not self.layers[0].self_attn.long_context:
            # IndexPool scoring cannot change selection inside the dense limit.
            weights = ((name, value) for name, value in weights if ".self_attn.indexer." not in name)
        return super().load_weights(weights)


class AscendHYV4FlashModelState(EngramModelState):
    def prepare_engram(self):
        # HY4 reads the lookback window and computes its hashes inside forward.
        self._engram_inputs = None
        return {}

    def finish_execution(self, *, failed: bool) -> None:
        self._engram_inputs = None
        AscendModelState.finish_execution(self, failed=failed)


class _AscendHYV4FlashHooks:
    @staticmethod
    def get_model_state_cls():
        return AscendHYV4FlashModelState

    @staticmethod
    def prepare_engram_graph_inputs(num_tokens):
        # The eager route only consumes the lookback window.
        return {}


class AscendHYV4FlashForCausalLM(_AscendHYV4FlashHooks, HYV4FlashForCausalLM):
    model_cls = AscendHYV4FlashModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        validate_ascend_hy4_flash(vllm_config)
        HYV4ForCausalLM.__init__(self, vllm_config=vllm_config, prefix=prefix)
        self.token_lookback_depth = self.model.token_lookback_depth
        self._weight_loading_depth = 0
        self._seen_weights: set[str] = set()
        self._loaded_weights: set[str] = set()
        self._load_config = vllm_config.load_config


@MULTIMODAL_REGISTRY.register_processor(
    HYV4VLMultiModalProcessor,
    info=HYV4FlashVLProcessingInfo,
    dummy_inputs=HYV4VLDummyInputsBuilder,
)
class AscendHYV4FlashVLForConditionalGeneration(_AscendHYV4FlashHooks, HYV4FlashVLForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        validate_ascend_hy4_flash(vllm_config)
        HYV4VLForConditionalGeneration.__init__(self, vllm_config=vllm_config, prefix=prefix)
        self.token_lookback_depth = self.language_model.token_lookback_depth
        engram_hash = self.language_model.model.engram_hash
        if engram_hash is not None:
            engram_hash.visual_token_ids = (
                self.config.image_token_id,
                self.config.video_token_id,
            )
        for name, value in vars(self.config).items():
            if name.startswith("dfly_"):
                setattr(self.language_model.config, name, value)
