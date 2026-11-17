# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-quant FlashMLA adapter for the CANN ops-transformer interface."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm_ascend.attention.flashmla import (
    FLASHMLA_BLOCK_SIZE,
    FLASHMLA_MASK_SIZE,
    FLASHMLA_QK_DIM,
    FLASHMLA_V_DIM,
    _check_lengths,
    _check_tensor,
)
from vllm_ascend.attention.utils import get_quant_flashmla_ops

QUANT_FLASHMLA_FP8_MODE = 1
QUANT_FLASHMLA_SUPPORTED_Q_HEADS = frozenset((1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128))


@dataclass(frozen=True)
class QuantFlashMLAConfig:
    num_heads: int
    softmax_scale: float
    mask_mode: int = 3
    layout_kv: str = "PA_BBND"
    return_softmax_lse: bool = False
    quant_mode: int = QUANT_FLASHMLA_FP8_MODE

    def __post_init__(self) -> None:
        if self.num_heads not in QUANT_FLASHMLA_SUPPORTED_Q_HEADS:
            raise ValueError(
                f"QuantFlashMLA requires local Q heads in {QUANT_FLASHMLA_SUPPORTED_Q_HEADS}, got {self.num_heads}"
            )
        if self.quant_mode != QUANT_FLASHMLA_FP8_MODE:
            raise ValueError("The vLLM Ascend QuantFlashMLA adapter currently supports FP8 E4M3 quant_mode=1 only")
        if self.mask_mode not in (0, 3):
            raise ValueError("QuantFlashMLA mask_mode must be 0 or 3")
        if self.layout_kv != "PA_BBND":
            raise ValueError("QuantFlashMLA layout_kv must be PA_BBND")


@dataclass(frozen=True)
class QuantFlashMLAImpl:
    """Wrap the FP8 MLA attention and device-side metadata operators."""

    config: QuantFlashMLAConfig
    attention_op: Callable
    metadata_op: Callable

    @classmethod
    def load(cls, config: QuantFlashMLAConfig) -> "QuantFlashMLAImpl":
        available_ops = get_quant_flashmla_ops()
        if available_ops is None:
            raise RuntimeError(
                "QuantFlashMLA requires cann_ops_transformer.ops.quant_flash_mla "
                "and quant_flash_mla_metadata. Install an ops-transformer package "
                "matching the worker's CANN and torch_npu versions."
            )
        attention_op, metadata_op = available_ops
        return cls(config, attention_op, metadata_op)

    def build_metadata(
        self,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the AICPU/device tiling schedule for the current batch."""
        _check_lengths(cache_seqlens, cu_seqlens_q, seqused_q)
        return self.metadata_op(
            cache_seqlens,
            self.config.num_heads,
            1,
            self.config.quant_mode,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            head_dim_qk=FLASHMLA_QK_DIM,
            head_dim_v=FLASHMLA_V_DIM,
            mask_mode=self.config.mask_mode,
            layout_q="TND",
        )

    def attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        *,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        metadata: torch.Tensor,
        seqused_q: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run full-quant MLA over one shared 576-wide paged KV cache."""
        self._check_inputs(
            q,
            k_cache,
            q_descale,
            k_descale,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            metadata,
            seqused_q,
            attn_mask,
        )
        output, lse = self.attention_op(
            q,
            k_cache,
            q_descale,
            k_descale,
            block_table,
            cache_seqlens,
            self.config.quant_mode,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            attn_mask=attn_mask,
            metadata=metadata,
            head_dim_v=FLASHMLA_V_DIM,
            softmax_scale=self.config.softmax_scale,
            mask_mode=self.config.mask_mode,
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            layout_q="TND",
            layout_kv=self.config.layout_kv,
            layout_out="NTD",
            return_softmax_lse=self.config.return_softmax_lse,
        )
        _check_tensor("output", output, (self.config.num_heads, q.shape[0], FLASHMLA_V_DIM), torch.bfloat16)
        if self.config.return_softmax_lse:
            _check_tensor("softmax_lse", lse, (self.config.num_heads, q.shape[0]), torch.float32)
        return output, lse

    def _check_inputs(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        metadata: torch.Tensor,
        seqused_q: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> None:
        batch_size = _check_lengths(cache_seqlens, cu_seqlens_q, seqused_q)
        tokens = q.shape[0] if q.ndim else 0
        if q.ndim != 3 or tokens == 0 or q.shape[1:] != (self.config.num_heads, FLASHMLA_QK_DIM):
            raise ValueError("QuantFlashMLA q must be nonempty TND with configured local heads and dimension 576")
        if q.dtype != torch.float8_e4m3fn:
            raise ValueError("QuantFlashMLA FP8 mode requires q dtype float8_e4m3fn")
        cache_tail = (FLASHMLA_BLOCK_SIZE, 1, FLASHMLA_QK_DIM)
        if k_cache.ndim != 4 or k_cache.shape[0] == 0 or k_cache.shape[1:] != cache_tail:
            raise ValueError(f"QuantFlashMLA PA_BBND cache must have shape (pages, {cache_tail})")
        if k_cache.dtype != q.dtype:
            raise ValueError("QuantFlashMLA q and k_cache must have the same FP8 dtype")
        _check_tensor("q_descale", q_descale, (tokens, self.config.num_heads), torch.float32)
        _check_tensor("k_descale", k_descale, (1,), torch.float32)
        if block_table.ndim != 2 or block_table.shape[0] != batch_size or block_table.shape[1] == 0:
            raise ValueError("QuantFlashMLA block_table must have shape (batch_size, positive_table_capacity)")
        if block_table.dtype != torch.int32:
            raise ValueError("QuantFlashMLA block_table must be int32")
        if metadata.ndim != 2 or metadata.shape[0] != 2 or metadata.shape[1] == 0 or metadata.dtype != torch.int32:
            raise ValueError("QuantFlashMLA metadata must be a nonempty (2, schedule_size) int32 tensor")
        if self.config.mask_mode == 0 and attn_mask is not None:
            raise ValueError("QuantFlashMLA mask_mode=0 requires attn_mask=None")
        if self.config.mask_mode == 3:
            if attn_mask is None:
                raise ValueError("QuantFlashMLA mask_mode=3 requires the upper-triangular int8 mask")
            _check_tensor("attn_mask", attn_mask, (FLASHMLA_MASK_SIZE, FLASHMLA_MASK_SIZE), torch.int8)
        for tensor in (k_cache, q_descale, k_descale, block_table, cache_seqlens, metadata, attn_mask):
            if tensor is not None and tensor.device != q.device:
                raise ValueError("QuantFlashMLA inputs must be on the same device")
