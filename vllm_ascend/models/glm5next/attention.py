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
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32))
        # Keep the checkpoint name ``index_kpool_compress_gate`` without a
        # ``.weight`` suffix. torch.mm consumes its [head_dim, hidden_size] shape.
        self.index_kpool_compress_gate = nn.Parameter(torch.empty(self.head_dim, hidden_size, dtype=torch.bfloat16))

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
        self.prefix = prefix

    def get_ascend_indexer_backend_cls(self):
        # Lazy import keeps the model module independent from the shared ops
        # registry during process startup.
        from vllm_ascend.attention.indexer_kpool_backend import (
            Glm5NextKPoolIndexerBackend,
        )

        return Glm5NextKPoolIndexerBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions,
        rotary_emb,
    ) -> torch.Tensor:
        del hidden_states, qr, positions, rotary_emb
        raise RuntimeError("GLM-Next Indexer must run through IndexerWrapper's Ascend backend.")


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
