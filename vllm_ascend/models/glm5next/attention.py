# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn
from vllm.config import (
    CacheConfig,
    VllmConfig,
)
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
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
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    yarn_get_mscale,
)

from vllm_ascend.models.glm5next.config import Glm5NextConfig
from vllm_ascend.models.glm5next.kv_cache import KpoolTailSpec
from vllm_ascend.models.glm5next.ops.kpool_compress import fwht128_quant_fp8
from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool
from vllm_ascend.utils import vllm_version_is

# Paged MQA page sizes the kpool tail cache aligns against. Upstream reads this
# from `vllm.utils.deep_gemm`, which is a CUDA-only module on Ascend.
PAGED_MQA_PAGE_SIZES = (32, 64)

# Shared torch.compile config for the indexer's small-kernel leaves. The MLA
# indexer runs under breakable-CG (CompilationMode.NONE), which blocks FX-graph
# fusion of the surrounding eager ops; carving each cluster into its own
# @torch.compile leaf (backend==inductor) still fuses them. Matches the
# grouped_topk / _cast_sigmoid leaf pattern.
_INDEXER_COMPILE = dict(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)


@torch.compile(**_INDEXER_COMPILE)
def _fused_indexer_k_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, dim: int, eps: float
) -> torch.Tensor:
    # Fuse fp32 cast + layer_norm + cast-back (was 3 kernels) into one.
    return F.layer_norm(x.float(), (dim,), weight, bias, eps).type_as(x)


@torch.compile(**_INDEXER_COMPILE)
def _fused_indexer_weight_scale(weights: torch.Tensor, q_scale: torch.Tensor, scale: float) -> torch.Tensor:
    # Fuse the weight-scaling muls (was 2 kernels) into one. `scale` folds
    # softmax_scale (head_dim**-0.5) and n_head**-0.5 into a single constant.
    return (weights.unsqueeze(-1) * q_scale * scale).squeeze(-1)


@torch.compile(**_INDEXER_COMPILE)
def _pad_indexer_heads(x: torch.Tensor, pad: int) -> torch.Tensor:
    # DeepGEMM MQA-logits needs num_heads in {32,64}; zero-pad the head dim.
    # Fuse new_zeros + cat (was 2 kernels) into one. Pad values are zero (exact
    # in fp8 e4m3 and zero-weight in the logits sum), so numerically a no-op.
    return torch.cat([x, x.new_zeros(x.shape[0], pad, *x.shape[2:])], dim=1)


class Glm5NextIndexerCache(DeepseekV32IndexerCache):
    """Indexer K cache that stores kpool-compressed entries.

    Setting ``compress_ratio`` / ``tokens_per_state`` (= index_kpool) on the KV
    cache spec makes vLLM's indexer metadata builder emit pool-granular
    ``slot_mapping`` / ``seq_lens`` / ``cu_seq_lens`` / ``page_table`` for free,
    and shrinks the cache allocation store one state per ``index_kpool`` tokens.
    The pool *content* (softmax-weighted sum vs keep-every-Nth) is computed by
    the kpool compress kernel inside the indexer op — the cache only provides
    the addressing, which is identical for both schemes.

    The indexer shares one block with the co-located MLA (a single
    ``MLAAttentionSpec`` / block_table), so ``block_size`` is the model-wide
    ``cache_config.block_size``. DeepGEMM's paged-MQA kernel
    (``csrc/apis/attention.hpp``) requires ``block_kv`` to be exactly 32 or
    64, so the storage block is virtually split into pool pages of the
    largest such size that tiles it (``storage_kernel_block_size``); this
    needs ``block_size`` to be a multiple of ``index_kpool * 32`` (512 for
    ``index_kpool = 16``). A smaller block (e.g. the default 64) silently
    collapses ``storage_block_size`` (64 // 16 = 4) and only fails later at
    the opaque C++ assert; ``get_kv_cache_spec`` guards this up front
    instead.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        index_kpool: int,
    ):
        super().__init__(head_dim=head_dim, dtype=dtype, prefix=prefix, cache_config=cache_config)
        assert index_kpool > 1, "Glm5NextIndexerCache expects index_kpool > 1"
        # Keep chunked-prefill boundaries aligned to complete pools.
        assert cache_config.block_size % index_kpool == 0, (
            "Glm5NextIndexerCache: cache_config.block_size "
            f"({cache_config.block_size}) must be a multiple of index_kpool "
            f"({index_kpool}) so chunked-prefill boundaries stay pool-aligned."
        )
        self._index_kpool = index_kpool

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        from dataclasses import replace

        spec = super().get_kv_cache_spec(vllm_config)
        # vLLM #51718 renamed MLAAttentionSpec.compress_ratio to
        # AttentionSpec.tokens_per_state on main; both express kpool compression.
        assert isinstance(spec, MLAAttentionSpec)
        if vllm_version_is("0.28.0"):
            spec = replace(spec, compress_ratio=self._index_kpool)
        else:
            spec = replace(spec, tokens_per_state=self._index_kpool)

        # DeepGEMM paged-MQA takes block_kv in {32, 64}; the storage block
        # (= block_size // index_kpool) is virtually split into pool pages of
        # the largest such size that tiles it, so it must be a multiple of 32.
        storage_block_size = spec.block_size // self._index_kpool
        assert spec.block_size % self._index_kpool == 0 and storage_block_size % 32 == 0, (
            "Glm5NextIndexerCache: kpool indexer requires cache block_size to "
            f"be a multiple of index_kpool * 32 ({self._index_kpool * 32}) so "
            "that DeepGEMM paged-MQA pool pages (32 or 64 entries) tile the "
            f"storage block, got block_size={spec.block_size} -> "
            f"storage_block_size={storage_block_size}."
        )
        max_page_size = max(PAGED_MQA_PAGE_SIZES)
        min_page_size = min(PAGED_MQA_PAGE_SIZES)
        if storage_block_size <= max_page_size:
            page_size = storage_block_size
        elif storage_block_size % max_page_size == 0:
            page_size = max_page_size
        else:
            page_size = min_page_size
        return replace(
            spec,
            storage_block_size=page_size * self._index_kpool,
        )


class Glm5NextTailCache(DeepseekV32IndexerCache):
    """Paged circular buffer for the kpool indexer's in-progress (tail) pool.

    Holds the trailing incomplete pool's raw K + gate score: one block of
    ``index_kpool`` slots per request, overwritten in place by ``pos % kpool``
    as decode/spec-decode advances. Prefill seeds it (instead of discarding the
    tail raw K+gate); the connector transfers it across PD; decode reads it to
    compress the boundary pool correctly. ``KpoolTailSpec`` /
    ``KpoolTailManager`` provide the no-prune, 1-block/req allocation that lets
    the in-progress pool survive across steps and across transfer.

    Stores raw bf16 K (``head_dim``) as the "K" half of each block and the
    bf16 gate score (``head_dim``) as the "V" half -- not the fp8-compressed
    entry, which lives in ``Glm5NextIndexerCache``.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        index_kpool: int,
    ):
        super().__init__(head_dim=head_dim, dtype=dtype, prefix=prefix, cache_config=cache_config)
        assert index_kpool > 1, "Glm5NextTailCache expects index_kpool > 1"
        self._index_kpool = index_kpool

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        # The two head slots form [K, gate score] in the generic
        # [block, head, state, content] cache view.
        return KpoolTailSpec(
            block_size=self._index_kpool,
            num_kv_heads=2,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=self._index_kpool,
        )

    def get_attn_backend(self):
        from vllm.v1.attention.backends.mla.indexer import KpoolTailBackend

        return KpoolTailBackend
from vllm_ascend.models.glm5next.kv_cache import (
    Glm5NextIndexerCache,
    Glm5NextStateCache,
)


class Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        # Indexer is only constructed for v32 configs, where these sparse-indexer
        # fields are guaranteed populated; narrow away the `int | None` declared
        # on Glm5NextConfig for the optional-indexer case.
        assert config.index_topk is not None
        assert config.index_n_heads is not None
        assert config.index_head_dim is not None
        assert config.index_kpool is not None
        assert cache_config is not None
        assert topk_indices_buffer is not None
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.index_kpool = config.index_kpool
        self.q_lora_rank = q_lora_rank  # 1536

        # kpool
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32)
        )
        # Keep the checkpoint name ``index_kpool_compress_gate`` without a
        # ``.weight`` suffix. torch.mm consumes its [head_dim, hidden_size] shape.
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.head_dim, hidden_size, dtype=torch.bfloat16)
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # FP8 wk weights are upcasted to BF16 during loading to maintain fusion.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = None
        self.quant_block_size = self.head_dim
        self.topk_indices_buffer = topk_indices_buffer
        # FP32 weight copies for the fp32 K/gate/weights projections. Cached
        # lazily after the checkpoint weights are loaded so the quantitative
        # work of the first forward matches the last.
        self._wk_weight_f32: torch.Tensor | None = None
        self._gate_weight_f32: torch.Tensor | None = None

        # Completed pools store BF16 vectors without quantization scales.
        self.k_cache = Glm5NextIndexerCache(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            cache_role="indexer",
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.index_kpool,
        )
        # Absolute-position FP32 K/gate pages retain the incomplete pool.
        self.state_cache = Glm5NextStateCache(
            state_dim=2 * self.head_dim,
            dtype=torch.float32,
            prefix=f"{prefix}.compressor.state_cache",
            cache_config=cache_config,
            compress_ratio=self.index_kpool,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = SparseAttnIndexerKpool(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            state_cache=self.state_cache,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions,
        rotary_emb,
    ) -> torch.Tensor:
        # Match glm-next-0806's kpool contract: the scoring chain consumes a
        # raw bf16 Q (no FWHT/fp8 rotation), an fp32 K that survives the norm
        # and the compressed-K insertion, raw head weights scaled only by the
        # softmax/head factors, and an fp32 gate. FP32 K/gate keep near-tie
        # pool rankings stable on long-context tasks.
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        hidden_f32 = hidden_states.float()
        if self._wk_weight_f32 is None:
            self._wk_weight_f32 = self.wk_weights_proj.weight.detach().float()
            self._gate_weight_f32 = self.index_kpool_compress_gate.detach().float()
        kw = torch.mm(hidden_f32, self._wk_weight_f32.t())
        weights = kw[:, self.head_dim :].to(torch.bfloat16)
        k = self.k_norm(kw[:, : self.head_dim])

        if self.rope_dim > 0:
            q_pe, q_nope = torch.split(
                q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
            k_pe, k_nope = torch.split(
                k,
                [self.rope_dim, self.head_dim - self.rope_dim],
                dim=-1,
            )
            # RoPE runs on the bf16 half; the nope half keeps fp32 so the cache
            # write matches the FP32 state the compressor reads back.
            k_pe = k_pe.to(torch.bfloat16)
            q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
            # Note: RoPE (NeoX) can introduce extra leading dimensions during
            # compilation so we need to reshape back to token-flattened shapes.
            q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
            k_pe = k_pe.reshape(-1, 1, self.rope_dim).float()
            # `rotary_emb` is shape-preserving; `q_pe` is already
            # [num_tokens, n_head, rope_dim].
            q = torch.cat([q_pe, q_nope], dim=-1)
            # `k_pe` is [num_tokens, 1, rope_dim] (MQA); keep the pe rounded to
            # bf16 but embed it back into the fp32 K row.
            k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)
        # else: qk_rope_head_dim=0 — no rope component. q is already
        # [num_tokens, n_head, head_dim] and k is [num_tokens, head_dim] (all
        # nope), so skip the rope split / rotary / cat entirely; otherwise the
        # split/reshape would build 0-element tensors (breaks dynamo tracing).

        q = q.to(torch.bfloat16)
        weights = weights * (self.softmax_scale * self.n_head**-0.5)
        gate_score = torch.mm(hidden_f32, self._gate_weight_f32.t())

        return self.indexer_op(
            hidden_states,
            q,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=self.index_kpool_compress_ape,
            index_kpool=self.index_kpool,
            positions=positions,
        )


class Glm5NextMLAAttention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextConfig,
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
        skip_rope: bool | None = False,
    ) -> None:
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
        self.max_position_embeddings = max_position_embeddings

        # Use input_size for projection input dimensions if provided,
        # otherwise default to hidden_size (used in Eagle3 Deepseek with MLA)
        proj_input_size = input_size if input_size is not None else self.hidden_size

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
                proj_input_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
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

        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                proj_input_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
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
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if not skip_rope:
            assert config.rope_parameters is not None
            if config.rope_parameters["rope_type"] != "default":
                config.rope_parameters["rope_type"] = (
                    "deepseek_yarn"
                    if config.rope_parameters.get("apply_yarn_scaling", True)
                    else "deepseek_llama_scaling"
                )

            self.rotary_emb: RotaryEmbedding | None = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=False,
            )

            if (
                config.rope_parameters["rope_type"] != "default"
                and config.rope_parameters["rope_type"] == "deepseek_yarn"
            ):
                mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
                scaling_factor = config.rope_parameters["factor"]
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb = None

        self.is_v32 = config.index_topk is not None

        if self.is_v32:
            self.indexer_rope_emb: RotaryEmbedding | None = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=not config.indexer_rope_interleave,
            )
            # The sparse indexer projects from the MLA q-lora rank, which is
            # always set for v32 MLA configs; narrow away the `int | None`.
            assert q_lora_rank is not None
            self.indexer: Indexer | None = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )

        else:
            self.indexer_rope_emb = None
            self.indexer = None

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj if self.q_lora_rank is not None else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa if self.q_lora_rank is None else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=self.is_v32,
            topk_indices_buffer=topk_indices_buffer,
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
            skip_topk=False,
            fuse_qkv_rmsnorm=True,
        )
        # The pluggable MLA wrapper owns the AttentionLayerBase registered in
        # the static forward context. Publish GLM-Next's cache contract on that
        # layer, matching the dedicated cache layer used by the source branch,
        # so the model runner can remain model agnostic.
        mla_cache_layer = self.mla_attn.mla_attn
        mla_cache_layer.model_version = "glm5_next"
        mla_cache_layer.indexes_kv_by_block_stride = True

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # The wrapper also runs the sparse indexer before MLA attention.
        return self.mla_attn(positions, hidden_states)
