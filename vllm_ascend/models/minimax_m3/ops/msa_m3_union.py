# SPDX-License-Identifier: Apache-2.0
"""Union prefill using a matched, separately installed pair of custom CANN ops."""

from importlib import import_module

import torch

from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _split_main_kv_cache
from vllm_ascend.models.minimax_m3.union_policy import UNION_MAX_Q_HEADS, UNION_PAGE_SIZE, UNION_SIZE


class UnionSparseAttention:
    def __init__(self) -> None:
        # Only enabled workers load the optional extensions. The caller must
        # source the matched vendor packages before starting the engine; never
        # rewrite process-wide OPP search paths from the attention hot path.
        try:
            self.nn_ops = import_module("cann_ops_nn_custom")
            self.attention_ops = import_module("cann_ops_transformer_custom")
        except ImportError as exc:
            raise RuntimeError("MiniMax M3 Union requires the custom NN and transformer Union packages") from exc
        for module, functions in (
            (self.nn_ops, ("union_sparse_meta_launch",)),
            (self.attention_ops, ("generic_block_sparse_attention_metadata", "generic_block_sparse_attention")),
        ):
            if any(not callable(getattr(module, name, None)) for name in functions):
                raise RuntimeError("The installed custom CANN packages do not expose the Union operator API")

    @torch.no_grad()
    def __call__(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
        topk_idx: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        seq_lens: torch.Tensor,
        num_union_groups: int,
        max_kv_blocks: int,
        sm_scale: float,
        output: torch.Tensor,
    ) -> None:
        if q.shape[0] == 0:
            return
        key, value = _split_main_kv_cache(kv_cache)
        if q.dtype not in (torch.bfloat16, torch.float16) or key.dtype != q.dtype or value.dtype != q.dtype:
            raise ValueError("Union requires matching BF16/FP16 Q/K/V; FP8 KV cache is unsupported")
        if value.shape != key.shape or output.shape != q.shape or output.dtype != q.dtype:
            raise ValueError("Union requires matching K/V shapes and an output matching Q")
        if key.shape[1:3] != (UNION_PAGE_SIZE, 1) or not 1 <= q.shape[1] <= UNION_MAX_Q_HEADS:
            raise ValueError("Union requires page128, one local KV head and at most 16 local Q heads")
        if not key.is_contiguous() or not value.is_contiguous():
            raise ValueError("Union requires contiguous PA_BBND K/V caches")
        if topk_idx.ndim != 3 or topk_idx.shape[:2] != (1, q.shape[0]) or topk_idx.shape[-1] <= 0:
            raise ValueError("Union requires token-wise sparse indices [1,totalQ,K] with K > 0")
        if num_union_groups <= 0 or max_kv_blocks <= 0:
            raise ValueError("Union requires positive request-local group and KV block counts")

        # Preserve the selector's sets: normalize score-ordered IDs into the
        # sorted logical IDs with trailing -1 required by UnionSparseMeta.
        # All sorting, counts, union tables and exact masks stay on the NPU.
        sentinel = torch.iinfo(torch.int32).max
        indices = topk_idx.to(dtype=torch.int32)
        valid = indices >= 0
        count = valid.sum(dim=-1).to(dtype=torch.int32)
        indices = torch.sort(torch.where(valid, indices, sentinel), dim=-1).values
        indices = torch.where(indices == sentinel, -1, indices).contiguous()
        capacity = min(UNION_SIZE * indices.shape[-1], max_kv_blocks)
        union_idx, union_count, exact_mask = self.nn_ops.union_sparse_meta_launch(
            indices,
            count,
            cu_seqlens_q,
            cu_seqlens_kv,
            None,
            seq_lens,
            1,
            UNION_PAGE_SIZE,
            UNION_SIZE,
            True,
            capacity,
            num_union_groups,
            max_kv_blocks,
        )
        common = dict(
            # The NN op takes int32 lengths; GBSA takes int64 cuQ.
            cu_seqlens_q=cu_seqlens_q.to(dtype=torch.int64),
            cu_seqlens_kv=None,
            seqused_kv=seq_lens,
            layout_q="TND",
            layout_kv="PA_BBND",
            layout_sparse_pattern=4,
            mask_mode=3,
            quant_mode=0,
            softmax_precision=0,
            win_left=-1,
            win_right=-1,
            residual_block_mode=0,
            is_consistent_topk=False,
        )
        metadata = self.attention_ops.generic_block_sparse_attention_metadata(
            union_idx,
            union_count,
            q.shape[1],
            1,
            q.shape[2],
            [1, UNION_PAGE_SIZE],
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=key.shape[0] * UNION_PAGE_SIZE,
            **common,
        )
        result, _ = self.attention_ops.generic_block_sparse_attention(
            q,
            key,
            value,
            union_idx,
            union_count,
            [1, UNION_PAGE_SIZE],
            metadata=metadata,
            block_table=block_table,
            attn_mask=exact_mask,
            softmax_scale=sm_scale,
            return_softmax_lse=False,
            **common,
        )
        output.copy_(result)
