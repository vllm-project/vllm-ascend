# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 sparse attention and indexer backends for Ascend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn.parameter import Parameter
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    adjust_block_scale_shard,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.parameter import BasevLLMParameter, BlockQuantScaleParameter
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.logger import init_logger
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)
from vllm_ascend.attention.msa_m3_npu import minimax_m3_sparse_attn
from vllm_ascend.attention.msa_m3_triton import (
    minimax_m3_sparse_attn_decode,
)
from vllm_ascend.attention.minimax_triton_indexer import (
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)

from vllm_ascend.ops.linear import AscendColumnParallelLinear
from vllm_ascend.ops.linear_op import get_parallel_op


logger = init_logger(__name__)

_SPARSE_ATTN_LOGGED = False


class AscendMiniMaxM3IndexerBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "ASCEND_MINIMAX_M3_SPARSE_INDEXER"

    @staticmethod
    def get_impl_cls() -> type["AscendMiniMaxM3IndexerImpl"]:
        return AscendMiniMaxM3IndexerImpl

    @staticmethod
    def get_builder_cls() -> type["AscendMiniMaxM3IndexerMetadataBuilder"]:
        return AscendMiniMaxM3IndexerMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError
        return (0, 1, 2, 3, 4)


class AscendMiniMaxM3IndexerCache(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        prefix: str,
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.dtype = torch.bfloat16
        self.prefix = prefix
        self.cache_config = cache_config
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.dtype,
        )

    def forward(self) -> None: ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendMiniMaxM3IndexerBackend


@dataclass
class AscendMiniMaxM3IndexerPrefillMetadata:
    cu_seqlens_q: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int


@dataclass
class AscendMiniMaxM3IndexerDecodeMetadata:
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    max_seq_len: int
    decode_query_len: int


@dataclass
class AscendMiniMaxM3IndexerMetadata(AttentionMetadata):
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    prefill: AscendMiniMaxM3IndexerPrefillMetadata | None = None
    decode: AscendMiniMaxM3IndexerDecodeMetadata | None = None


class AscendMiniMaxM3IndexerMetadataBuilder(
    AttentionMetadataBuilder[AscendMiniMaxM3IndexerMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.context_len_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMiniMaxM3IndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        context_lens = self.context_len_buffer[:num_reqs]
        context_lens.copy_(
            common_attn_metadata.compute_num_computed_tokens(), non_blocking=True
        )

        prefill_metadata: AscendMiniMaxM3IndexerPrefillMetadata | None = None
        if num_prefills > 0:
            prefill_metadata = AscendMiniMaxM3IndexerPrefillMetadata(
                cu_seqlens_q=(query_start_loc[num_decodes:] - num_decode_tokens).to(
                    torch.int32
                ),
                seq_lens=seq_lens[num_decodes:],
                context_lens=context_lens[num_decodes:],
                block_table=block_table[num_decodes:],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
            )

        decode_metadata: AscendMiniMaxM3IndexerDecodeMetadata | None = None
        if num_decodes > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            decode_metadata = AscendMiniMaxM3IndexerDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
            )

        return AscendMiniMaxM3IndexerMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )


class AscendMiniMaxM3IndexerImpl(nn.Module):
    def __init__(
        self,
        *,
        num_kv_heads: int,
        scale: float,
        topk_blocks: int,
        sparse_block_size: int,
        num_index_heads: int,
        index_head_dim: int,
        prefix: str,
        init_blocks: int = 0,
        local_blocks: int = 0,
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.num_index_heads = num_index_heads
        self.index_head_dim = index_head_dim
        self.index_cache = AscendMiniMaxM3IndexerCache(
            head_dim=index_head_dim,
            prefix=f"{prefix}.index_cache",
            cache_config=cache_config,
        )

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None
        index_md = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, AscendMiniMaxM3IndexerMetadata)
        num_tokens = index_md.num_actual_tokens
        nd = index_md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None
        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            decode_topk = minimax_m3_index_decode(
                iq[:nd],
                kv,
                d.block_table,
                d.seq_lens,
                d.max_seq_len,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                self.num_kv_heads,
                self.scale,
                d.decode_query_len,
            )
        if index_md.num_prefills > 0:
            p = index_md.prefill
            assert p is not None
            score = minimax_m3_index_score(
                iq[nd:],
                kv,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                p.context_lens,
                p.max_query_len,
                p.max_seq_len,
                self.num_kv_heads,
                self.scale,
            )
            prefill_topk = minimax_m3_index_topk(
                score,
                p.cu_seqlens_q,
                p.context_lens,
                p.max_query_len,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
            )
        return decode_topk, prefill_topk


class AscendMiniMaxM3Indexer(nn.Module):
    def __init__(
        self,
        *,
        num_kv_heads: int,
        scale: float,
        topk_blocks: int,
        sparse_block_size: int,
        num_index_heads: int,
        index_head_dim: int,
        prefix: str,
        init_blocks: int = 0,
        local_blocks: int = 0,
        cache_config: CacheConfig | None = None,
    ) -> None:
        super().__init__()
        self.impl = AscendMiniMaxM3IndexerImpl(
            num_kv_heads=num_kv_heads,
            scale=scale,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
            num_index_heads=num_index_heads,
            index_head_dim=index_head_dim,
            prefix=prefix,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            cache_config=cache_config,
        )

    @property
    def index_cache(self) -> AscendMiniMaxM3IndexerCache:
        return self.impl.index_cache

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.impl(index_query)


class AscendMiniMaxM3SparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "ASCEND_MINIMAX_M3_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["AscendMiniMaxM3SparseImpl"]:
        return AscendMiniMaxM3SparseImpl

    @staticmethod
    def get_builder_cls() -> type["AscendMiniMaxM3SparseMetadataBuilder"]:
        return AscendMiniMaxM3SparseMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError
        return (0, 1, 2, 3, 4)


@dataclass
class AscendMiniMaxM3SparsePrefillMetadata:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int


@dataclass
class AscendMiniMaxM3SparseDecodeMetadata:
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    max_seq_len: int
    decode_query_len: int


@dataclass
class AscendMiniMaxM3SparseMetadata(AttentionMetadata):
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    prefill: AscendMiniMaxM3SparsePrefillMetadata | None = None
    decode: AscendMiniMaxM3SparseDecodeMetadata | None = None


class AscendMiniMaxM3SparseMetadataBuilder(
    AttentionMetadataBuilder[AscendMiniMaxM3SparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.context_len_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMiniMaxM3SparseMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        context_lens = self.context_len_buffer[:num_reqs]
        context_lens.copy_(
            common_attn_metadata.compute_num_computed_tokens(), non_blocking=True
        )

        prefill_metadata: AscendMiniMaxM3SparsePrefillMetadata | None = None
        if num_prefills > 0:
            prefill_kv_lens = seq_lens[num_decodes:]
            prefill_cu_seqlens_k = torch.empty(
                num_prefills + 1, dtype=torch.int32, device=seq_lens.device
            )
            prefill_cu_seqlens_k[0] = 0
            torch.cumsum(prefill_kv_lens, dim=0, out=prefill_cu_seqlens_k[1:])
            prefill_metadata = AscendMiniMaxM3SparsePrefillMetadata(
                cu_seqlens_q=(query_start_loc[num_decodes:] - num_decode_tokens).to(
                    torch.int32
                ),
                cu_seqlens_k=prefill_cu_seqlens_k,
                seq_lens=prefill_kv_lens,
                context_lens=context_lens[num_decodes:],
                block_table=block_table[num_decodes:],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
            )

        decode_metadata: AscendMiniMaxM3SparseDecodeMetadata | None = None
        if num_decodes > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            decode_metadata = AscendMiniMaxM3SparseDecodeMetadata(
                seq_lens=seq_lens[:num_decodes],
                block_table=block_table[:num_decodes],
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
            )

        return AscendMiniMaxM3SparseMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )


class AscendMiniMaxM3SparseImpl(AttentionImplBase[AscendMiniMaxM3SparseMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_blocks: int,
        sparse_block_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_idx: tuple[torch.Tensor | None, torch.Tensor | None],
        output: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output
        main_md = attn_metadata[layer.layer_name]
        assert isinstance(main_md, AscendMiniMaxM3SparseMetadata)
        decode_topk, prefill_topk = topk_idx

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None and decode_topk is not None
            minimax_m3_sparse_attn_decode(
                q[:nd],
                kv_cache,
                decode_topk,
                d.block_table,
                d.seq_lens,
                self.num_kv_heads,
                self.scale,
                out[:nd],
                d.decode_query_len,
            )

        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None and prefill_topk is not None
            minimax_m3_sparse_attn(
                q[nd:],
                kv_cache,
                prefill_topk,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                p.context_lens,
                p.max_query_len,
                self.num_kv_heads,
                self.scale,
                out[nd:],
                block_size=self.block_size,
            )
        return output


class AscendMiniMaxM3QKVParallelLinearWithIndexer(QKVParallelLinear):
    """Fused [q | k | v | index_q | index_k] column-parallel GEMM for M3 sparse layers."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        total_num_index_heads: int,
        index_head_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        assert total_num_index_heads == total_num_kv_heads, (
            "AscendMiniMaxM3QKVParallelLinearWithIndexer requires "
            "total_num_index_heads == total_num_kv_heads"
        )
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.total_num_index_heads = total_num_index_heads
        self.index_head_size = index_head_size

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.num_index_heads = self.num_kv_heads

        q = self.num_heads * self.head_size
        kv = self.num_kv_heads * self.head_size
        iq = self.num_index_heads * self.index_head_size
        ik = self.index_head_size
        self.output_sizes = [
            q * tp_size,
            kv * tp_size,
            kv * tp_size,
            iq * tp_size,
            ik * tp_size,
        ]

        self.custom_op, _, _ = get_parallel_op(False, prefix, self, "column")
        AscendColumnParallelLinear.__init__(
            self,
            input_size=self.hidden_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(self, input_):
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)

    def validate_shard_id(self, loaded_shard_id: str | None) -> None:
        if loaded_shard_id is None:
            return
        if loaded_shard_id not in ("q", "k", "v", "index_q", "index_k"):
            raise ValueError(
                "Shard id for AscendMiniMaxM3QKVParallelLinearWithIndexer must be "
                "one of 'q', 'k', 'v', 'index_q', 'index_k'; got "
                f"{loaded_shard_id}."
            )

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int | None:
        h = self.head_size
        nq, nkv, nidx = self.num_heads, self.num_kv_heads, self.num_index_heads
        return {
            "q": 0,
            "k": nq * h,
            "v": (nq + nkv) * h,
            "index_q": (nq + 2 * nkv) * h,
            "index_k": (nq + 2 * nkv + nidx) * h,
        }.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        h = self.head_size
        return {
            "q": self.num_heads * h,
            "k": self.num_kv_heads * h,
            "v": self.num_kv_heads * h,
            "index_q": self.num_index_heads * h,
            "index_k": self.index_head_size,
        }.get(loaded_shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight,
        loaded_shard_id: str | None = None,
    ) -> None:
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("q", "k", "v", "index_q", "index_k")

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        num_heads = (
            self.tp_size if loaded_shard_id == "index_k" else self.num_kv_head_replicas
        )
        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=num_heads,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight,
        loaded_shard_id: str | None = None,
    ) -> None:
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("q", "k", "v", "index_q", "index_k")
        output_dim = getattr(param, "output_dim", None)
        assert output_dim is not None

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        param_data = param.data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "q":
            shard_rank = self.tp_rank
        elif loaded_shard_id == "index_k":
            shard_rank = 0
        else:
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        loaded_weight = loaded_weight.narrow(
            output_dim, shard_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

class AscendMiniMaxM3IndexerLinear(AscendColumnParallelLinear):
    """Merged [index_q | index_k] projection for M3 sparse layers.

    index_q follows KV-head tensor parallel sharding. index_k is always
    replicated, so every TP rank loads the first index_k shard.
    """

    def __init__(
        self,
        hidden_size: int,
        total_num_index_heads: int,
        index_head_size: int,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.hidden_size = hidden_size
        self.total_num_index_heads = total_num_index_heads
        self.index_head_size = index_head_size

        tp_size = get_tensor_model_parallel_world_size()
        if tp_size >= self.total_num_index_heads:
            self.num_index_heads = 1
            self.num_index_head_replicas = divide(tp_size, self.total_num_index_heads)
        else:
            self.num_index_heads = divide(self.total_num_index_heads, tp_size)
            self.num_index_head_replicas = 1

        self.index_q_size = self.num_index_heads * self.index_head_size
        self.index_k_size = self.index_head_size
        self.output_sizes = [
            self.index_q_size * tp_size,
            self.index_k_size * tp_size,
        ]

        self.custom_op, _, _ = get_parallel_op(False, prefix, self, "column")
        AscendColumnParallelLinear.__init__(
            self,
            input_size=self.hidden_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(self, input_):
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)

    def validate_shard_id(self, loaded_shard_id: str | None) -> None:
        if loaded_shard_id is None:
            return
        if loaded_shard_id not in ("index_q", "index_k"):
            raise ValueError(
                "Shard id for AscendMinimaxM3Indexer must be one of "
                f"'index_q', 'index_k'; got {loaded_shard_id}."
            )

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int | None:
        return {
            "index_q": 0,
            "index_k": self.index_q_size,
        }.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        return {
            "index_q": self.index_q_size,
            "index_k": self.index_k_size,
        }.get(loaded_shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight,
        loaded_shard_id: str | None = None,
    ) -> None:
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("index_q", "index_k")

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        num_heads = (
            self.tp_size
            if loaded_shard_id == "index_k"
            else self.num_index_head_replicas
        )
        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=num_heads,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight,
        loaded_shard_id: str | None = None,
    ) -> None:
        self.validate_shard_id(loaded_shard_id)
        assert loaded_shard_id in ("index_q", "index_k")
        output_dim = getattr(param, "output_dim", None)
        assert output_dim is not None

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        param_data = param.data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "index_k":
            shard_rank = 0
        else:
            shard_rank = self.tp_rank // self.num_index_head_replicas
        loaded_weight = loaded_weight.narrow(
            output_dim, shard_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


def _quant_description_value(
    quant_config: QuantizationConfig | None,
    key: str,
) -> Any | None:
    quant_description = getattr(quant_config, "quant_description", None)
    if not isinstance(quant_description, dict):
        return None
    return quant_description.get(key)


def _sparse_proj_quant_type(
    quant_config: QuantizationConfig | None,
    prefix: str,
    proj_name: str,
) -> Any | None:
    candidates = [prefix]
    if not prefix.startswith("language_model."):
        candidates.append(f"language_model.{prefix}")
    if prefix.startswith("model."):
        candidates.append(f"language_model.{prefix}")

    for candidate in dict.fromkeys(candidates):
        value = _quant_description_value(
            quant_config,
            f"{candidate}.{proj_name}.weight",
        )
        if value is not None:
            return value
    return None


def _use_fused_qkv_indexer(
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> bool:
    if quant_config is None:
        return True

    qkv_types = [
        _sparse_proj_quant_type(quant_config, prefix, proj_name)
        for proj_name in ("q_proj", "k_proj", "v_proj")
    ]
    index_q_type = _sparse_proj_quant_type(quant_config, prefix, "index_q_proj")
    index_k_type = _sparse_proj_quant_type(quant_config, prefix, "index_k_proj")

    if any(value is None for value in (*qkv_types, index_q_type, index_k_type)):
        return True
    if len(set(qkv_types)) != 1:
        raise ValueError(
            f"MiniMax M3 q/k/v quantization types differ for {prefix}: {qkv_types}"
        )
    if index_q_type != index_k_type:
        raise ValueError(
            "MiniMax M3 index_q/index_k quantization types differ for "
            f"{prefix}: {index_q_type} vs {index_k_type}"
        )
    return qkv_types[0] == index_q_type


def _register_m3_sparse_packed_modules(
    quant_config: QuantizationConfig | None,
    fused_qkv_indexer: bool,
) -> None:
    if quant_config is None:
        return
    packed_modules_mapping = getattr(quant_config, "packed_modules_mapping", None)
    if not isinstance(packed_modules_mapping, dict):
        return
    packed_modules_mapping["qkv_proj"] = ["q_proj", "k_proj", "v_proj"]
    if not fused_qkv_indexer:
        packed_modules_mapping["indexer_proj"] = ["index_q_proj", "index_k_proj"]


class MiniMaxM3SparseAttention(nn.Module, AttentionLayerBase):
    """Block-sparse attention with lightning indexer on Ascend."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_dim: int,
        rope_parameters: dict[str, Any] | None = None,
        attn_window_size: int | None = None,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        sparse_cfg: dict[str, Any] | None = None,
        disable_index_value: bool = False,
    ) -> None:
        super().__init__()
        assert sparse_cfg is not None
        self.hidden_size = hidden_size
        self.disable_index_value = disable_index_value

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.total_idx_heads = sparse_cfg["sparse_num_index_heads"]
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        assert self.total_idx_heads == self.total_num_kv_heads, (
            "MiniMax M3 sparse attention requires "
            "sparse_num_index_heads == num_key_value_heads"
        )
        self.num_idx_heads = self.num_kv_heads
        self.index_q_size = self.num_idx_heads * self.idx_head_dim
        self._use_fused_qkv_indexer = _use_fused_qkv_indexer(quant_config, prefix)
        _register_m3_sparse_packed_modules(quant_config, self._use_fused_qkv_indexer)

        if self._use_fused_qkv_indexer:
            self.qkv_proj = AscendMiniMaxM3QKVParallelLinearWithIndexer(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                self.total_idx_heads,
                self.idx_head_dim,
                bias=qkv_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.indexer_proj = None
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=qkv_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.indexer_proj = AscendMiniMaxM3IndexerLinear(
                hidden_size,
                self.total_idx_heads,
                self.idx_head_dim,
                bias=qkv_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.indexer_proj",
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if rope_parameters is not None and "partial_rotary_factor" not in rope_parameters:
            rope_parameters["partial_rotary_factor"] = rotary_dim / self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        self.index_q_norm = GemmaRMSNorm(self.idx_head_dim, eps=rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.idx_head_dim, eps=rms_norm_eps)

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)

        vllm_config = get_current_vllm_config()
        self.layer_name = f"{prefix}.attn"
        self.kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        self.attn_backend = AscendMiniMaxM3SparseBackend
        self.impl = AscendMiniMaxM3SparseImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            topk_blocks=sparse_cfg["sparse_topk_blocks"],
            sparse_block_size=sparse_cfg["sparse_block_size"],
        )
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]
        self.sparse_block_size = sparse_cfg["sparse_block_size"]
        self.indexer = AscendMiniMaxM3Indexer(
            num_kv_heads=self.num_kv_heads,
            scale=self.scaling,
            topk_blocks=self.topk_blocks,
            sparse_block_size=self.sparse_block_size,
            num_index_heads=self.num_idx_heads,
            index_head_dim=self.idx_head_dim,
            prefix=self.layer_name,
            init_blocks=sparse_cfg.get("sparse_init_block", 0),
            local_blocks=sparse_cfg.get("sparse_local_block", 0),
            cache_config=cache_config,
        )

        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self
        self.kv_cache = torch.tensor([])
        global _SPARSE_ATTN_LOGGED
        if not _SPARSE_ATTN_LOGGED:
            logger.warning(
                "MiniMax M3 sparse attention enabled via npu_sparse_attention_score "
                "(topk_blocks=%d, block_size=%d)",
                sparse_cfg["sparse_topk_blocks"],
                sparse_cfg["sparse_block_size"],
            )
            _SPARSE_ATTN_LOGGED = True

    def get_attn_backend(self) -> type[AscendMiniMaxM3SparseBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _insert_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        index_key: torch.Tensor,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return
        main_meta = attn_metadata[self.layer_name]
        index_meta = attn_metadata[self.indexer.index_cache.prefix]
        assert isinstance(main_meta, AscendMiniMaxM3SparseMetadata)
        assert isinstance(index_meta, AscendMiniMaxM3IndexerMetadata)

        from vllm_ascend.device.device_op import DeviceOperator

        key_cache, value_cache = self.kv_cache
        num_tokens = main_meta.num_actual_tokens
        k_insert = key[:num_tokens].view(-1, self.num_kv_heads, self.head_dim)
        v_insert = value[:num_tokens].view(-1, self.num_kv_heads, self.head_dim)
        DeviceOperator.reshape_and_cache(
            k_insert,
            v_insert,
            key_cache,
            value_cache,
            main_meta.slot_mapping[:num_tokens],
        )

        idx_cache = self.indexer.index_cache.kv_cache
        if isinstance(idx_cache, (tuple, list)):
            if len(idx_cache) >= 2:
                idx_key_cache, idx_value_cache = idx_cache[0], idx_cache[1]
            else:
                idx_key_cache, idx_value_cache = idx_cache[0][0], idx_cache[0][1]
        else:
            idx_key_cache, idx_value_cache = idx_cache[0], idx_cache[1]
        idx_insert = index_key[:num_tokens].view(-1, 1, self.idx_head_dim)
        DeviceOperator.reshape_and_cache(
            idx_insert,
            idx_insert,
            idx_key_cache,
            idx_value_cache,
            index_meta.slot_mapping[:num_tokens],
        )

    def _sparse_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv, _ = self.qkv_proj(hidden_states)
        main_qkv_size = self.q_size + 2 * self.kv_size
        if self.indexer_proj is None:
            main_qkv = qkv.narrow(-1, 0, main_qkv_size)
            index_q = qkv.narrow(-1, main_qkv_size, self.index_q_size)
            index_k = qkv.narrow(
                -1,
                main_qkv_size + self.index_q_size,
                self.idx_head_dim,
            )
        else:
            main_qkv = qkv
            index_qk, _ = self.indexer_proj(hidden_states)
            index_q, index_k = index_qk.split(
                [self.index_q_size, self.idx_head_dim],
                dim=-1,
            )

        if (
            main_qkv.device.type != "npu"
            or main_qkv.dtype != torch.bfloat16
            or positions.ndim != 1
            or not getattr(self.rotary_emb, "is_neox_style", True)
        ):
            q, k, v = main_qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            v = v.contiguous()
            q, k = self._qk_norm(q, k)
            q, k = self.rotary_emb(positions, q, k)
        else:
            q, k, v = torch.ops.vllm.qkv_rmsnorm_rope(
                input=main_qkv.contiguous(),
                q_weight=self.q_norm.weight_plus_one,
                k_weight=self.k_norm.weight_plus_one,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                eps=self.q_norm.variance_epsilon,
                q_bias=None,
                k_bias=None,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                positions=positions,
            )

        index_q, index_k = self._index_qk_norm(index_q, index_k)
        index_q, index_k = self.rotary_emb(positions, index_q, index_k)

        return q, k, v, index_q, index_k

    def _run_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        index_query: torch.Tensor,
        index_key: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> None:
        """Insert KV, build sparse top-k indices, then run sparse attention."""
        self._insert_kv(key, value, index_key)
        topk_idx = self.indexer(index_query)
        self.impl.forward(self, query, self.kv_cache, topk_idx, attn_output)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v, index_q, index_k = self._sparse_prepare(positions, hidden_states)
        attn_out = torch.empty_like(q)
        torch.ops.vllm.minimax_m3_sparse_forward(
            q,
            k,
            v,
            index_q,
            index_k,
            attn_out,
            self.layer_name,
        )
        projected, _ = self.o_proj(attn_out)
        return projected

    def _qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_shape = q.shape
        k_shape = k.shape
        q = q.reshape(-1, self.head_dim).contiguous()
        k = k.reshape(-1, self.head_dim).contiguous()
        q = self.q_norm(q).reshape(q_shape)
        k = self.k_norm(k).reshape(k_shape)
        return q, k

    def _index_qk_norm(
        self, idx_q: torch.Tensor, idx_k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx_q_shape = idx_q.shape
        idx_k_shape = idx_k.shape
        idx_q = idx_q.reshape(-1, self.index_q_size)
        idx_k = idx_k.reshape(-1, self.idx_head_dim)
        idx_q = self.index_q_norm(idx_q).reshape(idx_q_shape)
        idx_k = self.index_k_norm(idx_k).reshape(idx_k_shape)
        return idx_q, idx_k


def minimax_m3_sparse_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    attn_output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()

    attn_metadata = forward_context.attn_metadata
    if not isinstance(attn_metadata, dict):
        attn_output.zero_()
        return

    layer = forward_context.no_compile_layers[layer_name]
    layer._run_sparse_attention(
        query, key, value, index_query, index_key, attn_output
    )


def minimax_m3_sparse_forward_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    attn_output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="minimax_m3_sparse_forward",
    op_func=minimax_m3_sparse_forward,
    mutates_args=["attn_output"],
    fake_impl=minimax_m3_sparse_forward_fake,
    dispatch_key="PrivateUse1",
)
