# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 sparse attention and indexer backends for Ascend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn.parameter import Parameter
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import divide, get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    adjust_block_scale_shard,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.parameter import BasevLLMParameter, BlockQuantScaleParameter
from vllm.utils.torch_utils import direct_register_custom_op
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
    KVCacheSpec,
)

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import (
    MiniMaxM3TPDecodeScoreMetadata,
    minimax_m3_index_tp_block_parallel_decode,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import (
    minimax_m3_index_decode as minimax_m3_index_decode_ascendc,
)
from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import (
    minimax_m3_index_prefill as minimax_m3_index_prefill_ascendc,
)
from vllm_ascend.ops.linear import AscendColumnParallelLinear
from vllm_ascend.ops.linear_op import get_parallel_op
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_USE_ASCENDC_INDEX_SCORE = get_ascend_device_type() != AscendDeviceType.A5

if not _USE_ASCENDC_INDEX_SCORE:
    from vllm_ascend.models.minimax_m3.ops.msa_m3_triton_a5 import (
        minimax_m3_index_decode,
        minimax_m3_index_score,
        minimax_m3_index_topk,
    )


def _should_use_tp_sharded_index_decode(tp_size: int, num_prefills: int) -> bool:
    # The A5 Triton decode kernel operates on the complete, replicated index-K
    # cache on every TP rank. Keep the mainline block-sharded optimization for
    # the other device families only.
    return get_ascend_device_type() != AscendDeviceType.A5 and tp_size > 1 and num_prefills == 0


def _active_decode_num_reqs(
    num_decodes: int,
    num_decode_tokens: int,
    decode_query_len: int,
) -> int:
    """Return the number of real decode requests, ignoring FIA/graph padding."""
    if decode_query_len <= 0:
        return 0
    return min(num_decodes, num_decode_tokens // decode_query_len)


def _active_prefill_num_reqs(
    num_prefills: int,
    num_prefill_tokens: int,
    query_start_loc_cpu: torch.Tensor,
    num_decodes: int,
) -> int:
    """Return real prefill requests, ignoring FIA/SP tail padding segments."""
    if num_prefills <= 0 or num_prefill_tokens <= 0:
        return 0
    qsl_cpu = query_start_loc_cpu.detach().cpu()
    num_reqs_fia = int(qsl_cpu.shape[0] - 1)
    active = 0
    tokens_accounted = 0
    for i in range(num_decodes, min(num_reqs_fia, num_decodes + num_prefills)):
        query_len = int(qsl_cpu[i + 1] - qsl_cpu[i])
        if query_len <= 0:
            continue
        if tokens_accounted + query_len > num_prefill_tokens:
            break
        tokens_accounted += query_len
        active += 1
    if active > 0:
        return active
    return min(1, num_prefills)


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
    def get_impl_cls() -> type[AscendMiniMaxM3IndexerImpl]:
        return AscendMiniMaxM3IndexerImpl

    @staticmethod
    def get_builder_cls() -> type[AscendMiniMaxM3IndexerMetadataBuilder]:
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
        del num_kv_heads, cache_dtype_str
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError
        return (0, 1, 2)


class AscendMiniMaxM3IndexerCache(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        prefix: str,
        cache_config: CacheConfig | None = None,
        kv_cache_dtype: str = "auto",
        kv_cache_torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.dtype = kv_cache_torch_dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.prefix = prefix
        self.cache_config = cache_config
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return AscendSFAIndexerCacheSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            cache_dtype_str=self.kv_cache_dtype,
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
    start_loc: torch.Tensor | None = None


@dataclass
class AscendMiniMaxM3IndexerDecodeMetadata:
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    max_seq_len: int
    decode_query_len: int
    cu_seqlens_q: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    start_loc: torch.Tensor | None = None
    tp_score: MiniMaxM3TPDecodeScoreMetadata | None = None


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
    causal_mask: torch.Tensor | None = None
    prefill: AscendMiniMaxM3IndexerPrefillMetadata | None = None
    decode: AscendMiniMaxM3IndexerDecodeMetadata | None = None


class AscendMiniMaxM3IndexerMetadataBuilder(AttentionMetadataBuilder[AscendMiniMaxM3IndexerMetadata]):
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
        self.block_size = kv_cache_spec.block_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.attn_mask_builder = AttentionMaskBuilder(device) if _USE_ASCENDC_INDEX_SCORE else None

    def _build_tp_score_metadata(
        self,
        block_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        context_lens: torch.Tensor,
        *,
        max_seq_len: int,
        decode_query_len: int,
    ) -> MiniMaxM3TPDecodeScoreMetadata:
        """Package graph-stable inputs; derive TP tensors in model forward."""
        tp_rank = get_tp_group().rank_in_group
        max_block_count = (max_seq_len + self.block_size - 1) // self.block_size
        blocks_per_tp = (max_block_count + self.tp_size - 1) // self.tp_size
        block_offset = tp_rank * blocks_per_tp
        block_count = max(0, min(blocks_per_tp, max_block_count - block_offset))
        return MiniMaxM3TPDecodeScoreMetadata(
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            context_lens=context_lens,
            max_block_count=max_block_count,
            block_size=self.block_size,
            block_offset=block_offset,
            block_count=block_count,
            decode_query_len=decode_query_len,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMiniMaxM3IndexerMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        qsl_cpu = common_attn_metadata.query_start_loc_cpu

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
            require_uniform=True,
        )

        prefill_metadata: AscendMiniMaxM3IndexerPrefillMetadata | None = None
        active_prefills = 0
        if num_prefills > 0:
            active_prefills = _active_prefill_num_reqs(num_prefills, num_prefill_tokens, qsl_cpu, num_decodes)
            prefill_end = num_decodes + active_prefills
            prefill_query_lens_cpu = qsl_cpu[num_decodes + 1 : prefill_end + 1] - qsl_cpu[num_decodes:prefill_end]
            prefill_context_lens = self.context_len_buffer[num_decodes:prefill_end]
            prefill_context_lens.copy_(
                (seq_lens[num_decodes:prefill_end].detach().cpu() - prefill_query_lens_cpu).to(
                    device=self.context_len_buffer.device,
                    dtype=torch.int32,
                    non_blocking=True,
                ),
                non_blocking=True,
            )
            cu_seqlens_q = (query_start_loc[num_decodes : prefill_end + 1] - num_decode_tokens).to(torch.int32)
            prefill_metadata = AscendMiniMaxM3IndexerPrefillMetadata(
                cu_seqlens_q=cu_seqlens_q,
                seq_lens=seq_lens[num_decodes:prefill_end],
                context_lens=prefill_context_lens,
                block_table=block_table[num_decodes:prefill_end],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
                start_loc=(
                    torch.div(
                        prefill_context_lens,
                        self.block_size,
                        rounding_mode="floor",
                    ).to(dtype=torch.int32)
                    if _USE_ASCENDC_INDEX_SCORE
                    else None
                ),
            )

        decode_metadata: AscendMiniMaxM3IndexerDecodeMetadata | None = None
        active_decodes = 0
        if num_decodes > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            active_decodes = _active_decode_num_reqs(num_decodes, num_decode_tokens, decode_query_len)
            decode_context_lens = None
            decode_cu_seqlens_q = None
            if _USE_ASCENDC_INDEX_SCORE:
                decode_context_lens = self.context_len_buffer[:active_decodes]
                decode_context_lens.copy_(
                    seq_lens[:active_decodes] - decode_query_len,
                    non_blocking=True,
                )
                decode_cu_seqlens_q = query_start_loc[: active_decodes + 1].to(torch.int32)
            decode_metadata = AscendMiniMaxM3IndexerDecodeMetadata(
                seq_lens=seq_lens[:active_decodes],
                block_table=block_table[:active_decodes],
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
                cu_seqlens_q=decode_cu_seqlens_q,
                context_lens=decode_context_lens,
            )
            if _USE_ASCENDC_INDEX_SCORE and self.tp_size > 1 and active_prefills == 0:
                decode_metadata.tp_score = self._build_tp_score_metadata(
                    decode_metadata.block_table,
                    decode_cu_seqlens_q,
                    decode_context_lens,
                    max_seq_len=decode_metadata.max_seq_len,
                    decode_query_len=decode_query_len,
                )

        return AscendMiniMaxM3IndexerMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=active_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=active_prefills,
            num_prefill_tokens=num_prefill_tokens,
            causal_mask=(
                self.attn_mask_builder.get_splitfuse_attn_mask() if self.attn_mask_builder is not None else None
            ),
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
        kv_cache_dtype: str = "auto",
        kv_cache_torch_dtype: torch.dtype = torch.bfloat16,
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
            kv_cache_dtype=kv_cache_dtype,
            kv_cache_torch_dtype=kv_cache_torch_dtype,
        )

    def _decode_topk_tp_sharded(
        self,
        idx_q: torch.Tensor,
        index_kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        decode_query_len: int,
        tp_group: Any,
        tp_size: int,
        tp_rank: int,
    ) -> torch.Tensor:
        full_idx_q = tp_group.all_gather(idx_q.contiguous(), dim=1)

        max_block_count = (max_seq_len + self.block_size - 1) // self.block_size
        blocks_per_tp = (max_block_count + tp_size - 1) // tp_size
        block_offset = tp_rank * blocks_per_tp
        block_count = max(0, min(blocks_per_tp, max_block_count - block_offset))
        local_block_table = block_table[:, block_offset : block_offset + block_count].contiguous()
        local_seq_lens = torch.clamp(
            seq_lens - block_offset * self.block_size,
            min=0,
            max=block_count * self.block_size,
        )

        local_topk, local_scores = minimax_m3_index_decode(
            full_idx_q,
            index_kv_cache,
            local_block_table,
            local_seq_lens,
            max_seq_len,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            full_idx_q.shape[1],
            decode_query_len,
            sm_scale=self.scale,
            block_offset=block_offset,
            block_count=block_count,
            global_seq_lens=seq_lens,
            return_scores=True,
        )
        gathered_scores = tp_group.all_gather(local_scores.contiguous(), dim=-1)
        gathered_topk = tp_group.all_gather(local_topk.contiguous(), dim=-1)

        local_head_count = idx_q.shape[1]
        local_head_start = tp_rank * local_head_count
        local_gathered_scores = gathered_scores.narrow(0, local_head_start, local_head_count)
        _, merged_pos = torch.topk(
            local_gathered_scores,
            k=self.topk_blocks,
            dim=-1,
        )
        local_gathered_topk = gathered_topk.narrow(0, local_head_start, local_head_count)
        return torch.gather(local_gathered_topk, dim=-1, index=merged_pos)

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None, None
        index_md = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, AscendMiniMaxM3IndexerMetadata)
        num_tokens = index_md.num_actual_tokens
        num_decode_tokens = index_md.num_decode_tokens
        iq = index_query[:num_tokens].view(-1, self.num_index_heads, self.index_head_dim)
        kv = self.index_cache.kv_cache

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None
        decode_select_num_idx: torch.Tensor | None = None
        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            tp_group = get_tp_group()
            decode_iq = iq[:num_decode_tokens]
            if _USE_ASCENDC_INDEX_SCORE:
                if tp_group.world_size > 1 and index_md.num_prefills == 0:
                    decode_topk = minimax_m3_index_tp_block_parallel_decode(
                        decode_iq,
                        kv,
                        d.tp_score,
                        index_md.causal_mask,
                        topk=self.topk_blocks,
                        init_blocks=self.init_blocks,
                        local_blocks=self.local_blocks,
                        tp_group=tp_group,
                    )
                else:
                    decode_start_loc = torch.div(
                        d.context_lens,
                        self.block_size,
                        rounding_mode="floor",
                    ).to(dtype=torch.int32)
                    decode_topk = minimax_m3_index_decode_ascendc(
                        decode_iq,
                        kv,
                        d.block_table,
                        d.cu_seqlens_q,
                        d.seq_lens,
                        d.context_lens,
                        decode_start_loc,
                        index_md.causal_mask,
                        topk=self.topk_blocks,
                        init_blocks=self.init_blocks,
                        local_blocks=self.local_blocks,
                        decode_query_len=d.decode_query_len,
                    )
            else:
                tp_size = tp_group.world_size
                tp_rank = tp_group.rank_in_group
                if _should_use_tp_sharded_index_decode(tp_size, index_md.num_prefills):
                    decode_topk = self._decode_topk_tp_sharded(
                        decode_iq,
                        kv,
                        d.block_table,
                        d.seq_lens,
                        d.max_seq_len,
                        d.decode_query_len,
                        tp_group,
                        tp_size,
                        tp_rank,
                    )
                else:
                    decode_topk, decode_select_num_idx = minimax_m3_index_decode(
                        decode_iq,
                        kv,
                        d.block_table,
                        d.seq_lens,
                        d.max_seq_len,
                        self.topk_blocks,
                        self.init_blocks,
                        self.local_blocks,
                        self.num_kv_heads,
                        d.decode_query_len,
                        sm_scale=self.scale,
                    )
        if index_md.num_prefills > 0:
            p = index_md.prefill
            assert p is not None
            if _USE_ASCENDC_INDEX_SCORE:
                prefill_topk = minimax_m3_index_prefill_ascendc(
                    iq[num_decode_tokens:],
                    kv,
                    p.block_table,
                    p.cu_seqlens_q,
                    p.seq_lens,
                    p.context_lens,
                    p.start_loc,
                    index_md.causal_mask,
                    max_query_len=p.max_query_len,
                    max_seq_len=p.max_seq_len,
                    topk=self.topk_blocks,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                )
            else:
                score = minimax_m3_index_score(
                    iq[num_decode_tokens:],
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
        return decode_topk, prefill_topk, decode_select_num_idx

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        """No-op: indexer replay parameters are owned by its metadata builder,
        not by the full-graph parameter-update channel.

        Selection note: _get_graph_update_backend picks the first executable
        backend, so correct updates for mixed-attention models rely on the
        full-attention (GQA) group being scanned first — M3's full-attention
        layers are the first layers registered, which holds today.
        """


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
        kv_cache_dtype: str = "auto",
        kv_cache_torch_dtype: torch.dtype = torch.bfloat16,
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
            kv_cache_dtype=kv_cache_dtype,
            kv_cache_torch_dtype=kv_cache_torch_dtype,
        )

    @property
    def index_cache(self) -> AscendMiniMaxM3IndexerCache:
        return self.impl.index_cache

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
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
    def get_impl_cls() -> type[AscendMiniMaxM3SparseImpl]:
        return AscendMiniMaxM3SparseImpl

    @staticmethod
    def get_builder_cls() -> type[AscendMiniMaxM3SparseMetadataBuilder]:
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


class AscendMiniMaxM3SparseMetadataBuilder(AttentionMetadataBuilder[AscendMiniMaxM3SparseMetadata]):
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
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        qsl_cpu = common_attn_metadata.query_start_loc_cpu

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
            require_uniform=True,
        )

        prefill_metadata: AscendMiniMaxM3SparsePrefillMetadata | None = None
        active_prefills = 0
        if num_prefills > 0:
            active_prefills = _active_prefill_num_reqs(num_prefills, num_prefill_tokens, qsl_cpu, num_decodes)
            prefill_end = num_decodes + active_prefills
            prefill_kv_lens = seq_lens[num_decodes:prefill_end]
            prefill_cu_seqlens_k = torch.empty(active_prefills + 1, dtype=torch.int32, device=seq_lens.device)
            prefill_cu_seqlens_k[0] = 0
            torch.cumsum(prefill_kv_lens, dim=0, out=prefill_cu_seqlens_k[1:])
            prefill_query_lens_cpu = qsl_cpu[num_decodes + 1 : prefill_end + 1] - qsl_cpu[num_decodes:prefill_end]
            prefill_context_lens = self.context_len_buffer[num_decodes:prefill_end]
            prefill_context_lens.copy_(
                (prefill_kv_lens.detach().cpu() - prefill_query_lens_cpu).to(
                    device=self.context_len_buffer.device,
                    dtype=torch.int32,
                    non_blocking=True,
                ),
                non_blocking=True,
            )
            cu_seqlens_q = (query_start_loc[num_decodes : prefill_end + 1] - num_decode_tokens).to(torch.int32)
            prefill_metadata = AscendMiniMaxM3SparsePrefillMetadata(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=prefill_cu_seqlens_k,
                seq_lens=prefill_kv_lens,
                context_lens=prefill_context_lens,
                block_table=block_table[num_decodes:prefill_end],
                max_query_len=common_attn_metadata.max_query_len,
                max_seq_len=common_attn_metadata.max_seq_len,
            )

        decode_metadata: AscendMiniMaxM3SparseDecodeMetadata | None = None
        active_decodes = 0
        if num_decodes > 0:
            qsl_cpu = common_attn_metadata.query_start_loc_cpu
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            active_decodes = _active_decode_num_reqs(num_decodes, num_decode_tokens, decode_query_len)
            decode_metadata = AscendMiniMaxM3SparseDecodeMetadata(
                seq_lens=seq_lens[:active_decodes],
                block_table=block_table[:active_decodes],
                max_seq_len=common_attn_metadata.max_seq_len,
                decode_query_len=decode_query_len,
            )

        return AscendMiniMaxM3SparseMetadata(
            seq_lens=seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_actual_tokens=num_tokens,
            num_decodes=active_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=active_prefills,
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
        self._dequant_scale: torch.Tensor | None = None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_idx: tuple[
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
        output: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output
        main_md = attn_metadata[layer.layer_name]
        assert isinstance(main_md, AscendMiniMaxM3SparseMetadata)
        decode_topk, prefill_topk, decode_select_num_idx = topk_idx

        num_decode_tokens = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)

        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None and decode_topk is not None
            if self.kv_cache_dtype.startswith("fp8") and self._dequant_scale is None:
                self._dequant_scale = torch.ones(
                    (1, 1, 1, 1),
                    dtype=torch.float32,
                    device=query.device,
                )
            minimax_m3_sparse_attn_decode(
                q[:num_decode_tokens],
                kv_cache,
                decode_topk,
                d.block_table,
                d.seq_lens,
                self.num_kv_heads,
                self.scale,
                out[:num_decode_tokens],
                d.decode_query_len,
                block_size=self.block_size,
                select_num_idx=decode_select_num_idx,
                dequant_scale=self._dequant_scale,
            )

        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None and prefill_topk is not None
            minimax_m3_sparse_attn(
                q[num_decode_tokens:],
                kv_cache,
                prefill_topk,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                p.context_lens,
                p.max_query_len,
                self.num_kv_heads,
                self.scale,
                out[num_decode_tokens:],
                block_size=self.block_size,
            )
        return output

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        """No-op: sparse-attention replay parameters are owned by its metadata
        builder, not by the full-graph parameter-update channel.

        Selection note: _get_graph_update_backend picks the first executable
        backend, so correct updates for mixed-attention models rely on the
        full-attention (GQA) group being scanned first — M3's full-attention
        layers are the first layers registered, which holds today.
        """


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
            "AscendMiniMaxM3QKVParallelLinearWithIndexer requires total_num_index_heads == total_num_kv_heads"
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
            shard_size, shard_offset = adjust_block_scale_shard(weight_block_size, shard_size, shard_offset)

        num_heads = self.tp_size if loaded_shard_id == "index_k" else self.num_kv_head_replicas
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
            shard_size, shard_offset = adjust_block_scale_shard(weight_block_size, shard_size, shard_offset)

        param_data = param.data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "q":
            shard_rank = self.tp_rank
        elif loaded_shard_id == "index_k":
            shard_rank = 0
        else:
            shard_rank = self.tp_rank // self.num_kv_head_replicas
        loaded_weight = loaded_weight.narrow(output_dim, shard_rank * shard_size, shard_size)
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
                f"Shard id for AscendMinimaxM3Indexer must be one of 'index_q', 'index_k'; got {loaded_shard_id}."
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
            shard_size, shard_offset = adjust_block_scale_shard(weight_block_size, shard_size, shard_offset)

        num_heads = self.tp_size if loaded_shard_id == "index_k" else self.num_index_head_replicas
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
            shard_size, shard_offset = adjust_block_scale_shard(weight_block_size, shard_size, shard_offset)
        assert shard_size is not None

        param_data = param.data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "index_k":
            shard_rank = 0
        else:
            shard_rank = self.tp_rank // self.num_index_head_replicas
        loaded_weight = loaded_weight.narrow(output_dim, shard_rank * shard_size, shard_size)
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
        _sparse_proj_quant_type(quant_config, prefix, proj_name) for proj_name in ("q_proj", "k_proj", "v_proj")
    ]
    index_q_type = _sparse_proj_quant_type(quant_config, prefix, "index_q_proj")
    index_k_type = _sparse_proj_quant_type(quant_config, prefix, "index_k_proj")

    if any(value is None for value in (*qkv_types, index_q_type, index_k_type)):
        return True
    if len(set(qkv_types)) != 1:
        raise ValueError(f"MiniMax M3 q/k/v quantization types differ for {prefix}: {qkv_types}")
    if index_q_type != index_k_type:
        raise ValueError(
            f"MiniMax M3 index_q/index_k quantization types differ for {prefix}: {index_q_type} vs {index_k_type}"
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
    layer._run_sparse_attention(query, key, value, index_query, index_key, attn_output)


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
