# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 FA3 absorbed MLA, sharing batch/graph orchestration with A5 FlashMLA.

The public FA3 contract uses q=RoPE, qv=NoPE and separate paged K/V tensors.
Persistent KV writes remain owned by MLA preprocessing. Only query-sized
buffers are packed; the history cache is never concatenated or materialized.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from vllm_ascend.attention.flashmla import (
    FLASHMLA_BLOCK_SIZE,
    FLASHMLA_QK_DIM,
    FLASHMLA_V_DIM,
    FlashMLAMetadataBuilder,
    _check_lengths,
    _check_tensor,
)
from vllm_ascend.attention.utils import get_flash_attn_npu_ops


@dataclass(frozen=True)
class FlashAttnNPUConfig:
    num_heads: int
    softmax_scale: float
    mask_mode: int = 3
    return_softmax_lse: bool = False

    def __post_init__(self):
        if self.num_heads <= 0 or self.mask_mode not in (0, 3):
            raise ValueError("A3 FlashAttention MLA requires positive Q heads and mask_mode 0 or 3")


@dataclass(frozen=True)
class FlashAttnNPUImpl:
    config: FlashAttnNPUConfig
    attention_op: Callable
    metadata_op: Callable
    qkv_dtype: torch.dtype
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    @classmethod
    def load(cls, config: FlashAttnNPUConfig, dtype: torch.dtype):
        ops = get_flash_attn_npu_ops()
        if ops is None:
            raise RuntimeError("A3 MLA requires flash_attn_npu_3.flash_attn_with_kvcache and get_scheduler_metadata")
        return cls(config, *ops, dtype)

    def build_metadata(self, cache_seqlens, cu_seqlens_q, seqused_q=None):
        batch = _check_lengths(cache_seqlens, cu_seqlens_q, seqused_q)
        if self.max_seqlen_q <= 0 or self.max_seqlen_k <= 0:
            raise ValueError("A3 FlashAttention metadata requires positive query and KV capacity bounds")
        return self.metadata_op(
            batch_size=batch,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            num_heads_q=self.config.num_heads,
            num_heads_kv=1,
            headdim=FLASHMLA_QK_DIM - FLASHMLA_V_DIM,
            headdim_v=FLASHMLA_V_DIM,
            cache_seqlens=cache_seqlens,
            qkv_dtype=self.qkv_dtype,
            cu_seqlens_q=cu_seqlens_q,
            page_size=FLASHMLA_BLOCK_SIZE,
            causal=self.config.mask_mode == 3,
        )

    def attention(
        self, q, k_cache, *, block_table, cache_seqlens, cu_seqlens_q, metadata, seqused_q=None, attn_mask=None
    ):
        batch = _check_lengths(cache_seqlens, cu_seqlens_q, seqused_q)
        if not isinstance(k_cache, tuple) or len(k_cache) != 2:
            raise ValueError("A3 FlashAttention requires separate (NoPE, RoPE) caches")
        nope, rope = k_cache
        if (
            q.ndim != 3
            or not 0 < q.shape[0] <= self.max_seqlen_q
            or q.shape[1:] != (self.config.num_heads, FLASHMLA_QK_DIM)
        ):
            raise ValueError("A3 FlashAttention query must be TND with head dimension 576")
        if q.dtype != self.qkv_dtype or q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("A3 FlashAttention requires matching BF16 or FP16 inputs")
        # FA3 accepts a padded leading page stride, but packed inner pages.
        # Reject interleaved token-fused views before FA3 could copy history.
        for name, cache, dim in (("NoPE", nope, FLASHMLA_V_DIM), ("RoPE", rope, FLASHMLA_QK_DIM - FLASHMLA_V_DIM)):
            if cache.ndim != 4 or cache.shape[0] == 0 or cache.shape[1:] != (FLASHMLA_BLOCK_SIZE, 1, dim):
                raise ValueError(f"A3 FlashAttention {name} cache must be (pages, 128, 1, {dim})")
            if cache.dtype != q.dtype or cache.stride(-1) != 1 or cache.stride(1) != dim or cache.stride(2) != dim:
                raise ValueError(f"A3 FlashAttention {name} cache requires packed component pages")
        if nope.shape[0] != rope.shape[0]:
            raise ValueError("A3 FlashAttention caches must have matching page counts")
        if (
            block_table.ndim != 2
            or block_table.shape[0] != batch
            or block_table.shape[1] * FLASHMLA_BLOCK_SIZE != self.max_seqlen_k
        ):
            raise ValueError("A3 FlashAttention block table must match the metadata KV capacity")
        if block_table.dtype != torch.int32:
            raise ValueError("A3 FlashAttention block table must be int32")
        if metadata.ndim != 1 or metadata.numel() == 0 or metadata.dtype != torch.uint8 or not metadata.is_contiguous():
            raise ValueError("A3 FlashAttention metadata must be a nonempty byte vector")
        for tensor in (nope, rope, block_table, cache_seqlens, metadata):
            if tensor.device != q.device:
                raise ValueError("A3 FlashAttention inputs must share a device")
        # The FA3 public API derives used query lengths from cu_seqlens_q.
        # Zero-KV rows own graph padding and must produce zero outputs.
        result = self.attention_op(
            q[..., FLASHMLA_V_DIM:],
            rope,
            nope,
            qv=q[..., :FLASHMLA_V_DIM],
            cache_seqlens=cache_seqlens,
            page_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=self.max_seqlen_q,
            softmax_scale=self.config.softmax_scale,
            causal=self.config.mask_mode == 3,
            scheduler_metadata=metadata,
            return_softmax_lse=self.config.return_softmax_lse,
        )
        if self.config.return_softmax_lse:
            output, lse, *_ = result
            _check_tensor("softmax_lse", lse, (self.config.num_heads, q.shape[0]), torch.float32)
        else:
            output, lse = result, None
        _check_tensor("output", output, (q.shape[0], self.config.num_heads, FLASHMLA_V_DIM), q.dtype)
        return output.transpose(0, 1), lse


class FlashAttnNPUMetadataBuilder(FlashMLAMetadataBuilder):
    @staticmethod
    def _load_adapter(impl):
        return FlashAttnNPUImpl.load(
            FlashAttnNPUConfig(impl.num_heads, impl.scale, return_softmax_lse=getattr(impl, "dcp_size", 1) > 1),
            impl.dtype,
        )

    def _get_adapter(self, causal, tokens, columns, *, current=False):
        adapter = super()._get_adapter(causal, tokens, columns, current=current)
        # Bounds come from allocated shapes, never a device max().item().
        return replace(adapter, max_seqlen_q=tokens, max_seqlen_k=columns * FLASHMLA_BLOCK_SIZE)

    def _allocate(self, tokens, rows, columns, causal):
        flash = super()._allocate(tokens, rows, columns, causal)
        if flash.current is not None:
            cache = flash.current.cache
            pages = cache.shape[0]
            flat = cache.view(-1)
            split = pages * FLASHMLA_BLOCK_SIZE * FLASHMLA_V_DIM
            flash.current.cache = (
                flat[:split].view(pages, FLASHMLA_BLOCK_SIZE, 1, FLASHMLA_V_DIM),
                flat[split:].view(pages, FLASHMLA_BLOCK_SIZE, 1, FLASHMLA_QK_DIM - FLASHMLA_V_DIM),
            )
        return flash
