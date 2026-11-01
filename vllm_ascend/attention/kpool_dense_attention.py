# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch
import torch_npu
from vllm.config import VllmConfig

from vllm_ascend.attention.kpool_query_plan import (
    DENSE_KERNEL_BLOCK_SIZE,
    DENSE_SELECTION_BUDGET,
    KPoolMHAPlan,
    KPoolQueryPlan,
)
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

DENSE_LATENT_DIM = 512
DENSE_LOCAL_HEADS = 8
DENSE_TP_SIZE = 8
C8_PACKED_DIM = 528
C8_SCALE_TILE = 128
FIA_COMPAT_ROPE_DIM = 64
MHA_HEAD_DIM = 256
KPOOL_EXPANSION_TAIL = 3


def supports_projected_kpool_attention(config: VllmConfig) -> bool:
    model = getattr(config.model_config, "hf_text_config", None)
    return bool(
        _supports_c8_kpool_attention(config, HardwareCapability.KPOOL_C8_PROJECTED_MHA)
        and getattr(model, "qk_nope_head_dim", None) == MHA_HEAD_DIM
        and getattr(model, "v_head_dim", None) == MHA_HEAD_DIM
        and getattr(model, "index_kpool", None) == KPOOL_EXPANSION_TAIL + 1
    )


def can_execute_projected_kpool_attention(cache: torch.Tensor, metadata: Any, plan: KPoolMHAPlan) -> bool:
    return bool(
        cache.dtype == torch.int8
        and cache.ndim == 4
        and cache.shape[2:] == (1, C8_PACKED_DIM)
        and cache.is_contiguous()
        and metadata.block_size == DENSE_KERNEL_BLOCK_SIZE
        and metadata.block_table.shape[0] == len(plan.key_lengths)
        and metadata.block_table.shape[1] >= (plan.max_key_len + DENSE_KERNEL_BLOCK_SIZE - 1) // DENSE_KERNEL_BLOCK_SIZE
    )


def execute_projected_kpool_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    metadata: Any,
    plan: KPoolMHAPlan,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    cache = cache.view(-1, DENSE_KERNEL_BLOCK_SIZE, 1, C8_PACKED_DIM)
    gathered = []
    for request, length in enumerate(plan.key_lengths):
        pages = (length + DENSE_KERNEL_BLOCK_SIZE - 1) // DENSE_KERNEL_BLOCK_SIZE
        packed = torch.index_select(cache, 0, metadata.block_table[request, :pages].long())
        scales = packed[..., DENSE_LATENT_DIM:].contiguous().view(torch.float32).repeat_interleave(C8_SCALE_TILE, -1)
        latent = (packed[..., :DENSE_LATENT_DIM].float() * scales).bfloat16().view(-1, DENSE_LATENT_DIM)
        gathered.append(latent[:length])
    latent = gathered[0] if len(gathered) == 1 else torch.cat(gathered)
    expanded = latent[None].expand(DENSE_LOCAL_HEADS, -1, -1)
    key = torch.bmm(expanded, key_weight.transpose(1, 2)).transpose(0, 1).contiguous()
    value = torch.bmm(expanded, value_weight).transpose(0, 1).contiguous()
    outputs: list[torch.Tensor] = []
    query_start, key_start = 0, 0
    # Request-local TND masks avoid padding incompatible ragged mask shapes.
    for query_end, key_end, length in zip(plan.query_ends, plan.key_ends, plan.key_lengths):
        slots = indices[query_start:query_end, 0]
        visible = plan.visible_key_lengths[query_start:query_end, None]
        slots = torch.where((slots >= 0) & (slots < visible), slots, length).long()
        mask = torch.ones((slots.shape[0], length + 1), dtype=torch.int32, device=query.device)
        mask.scatter_(1, slots, 0)
        mask = mask[:, :length].bool().contiguous()
        outputs.append(
            torch_npu.npu_fusion_attention(
                query[query_start:query_end],
                key[key_start:key_end],
                value[key_start:key_end],
                DENSE_LOCAL_HEADS,
                input_layout="TND",
                atten_mask=mask,
                scale=scale,
                keep_prob=1.0,
                sparse_mode=0,
                actual_seq_qlen=[query_end - query_start],
                actual_seq_kvlen=[length],
            )[0]
        )
        query_start, key_start = query_end, key_end
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


def supports_dense_kpool_attention(config: VllmConfig) -> bool:
    return _supports_c8_kpool_attention(config, HardwareCapability.KPOOL_C8_DENSE_PREFIX)


def _supports_c8_kpool_attention(config: VllmConfig, capability: HardwareCapability) -> bool:
    model = getattr(config.model_config, "hf_text_config", None)
    parallel = config.parallel_config
    return bool(
        getattr(config.cache_config, "cache_dtype", None) == "int8"
        and getattr(config.model_config, "dtype", None) == torch.bfloat16
        and getattr(parallel, "tensor_parallel_size", None) == DENSE_TP_SIZE
        and parallel.prefill_context_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and getattr(model, "num_attention_heads", None) == DENSE_LOCAL_HEADS * DENSE_TP_SIZE
        and getattr(model, "kv_lora_rank", None) == DENSE_LATENT_DIM
        and getattr(model, "qk_rope_head_dim", None) == 0
        and getattr(model, "index_topk", None) == DENSE_SELECTION_BUDGET
        and get_current_hardware_profile().supports(capability)
    )


def execute_kpool_hybrid_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    metadata: Any,
    key_lengths: torch.Tensor,
    query_plan: KPoolQueryPlan | None,
    scale: float,
    sparse_attention: Callable,
) -> torch.Tensor | None:
    plan = None if query_plan is None else query_plan.dense
    if (
        query_plan is None
        or plan is None
        or query_plan.num_tokens != query.shape[0]
        or query.dtype != torch.bfloat16
        or query.shape[1:] != (DENSE_LOCAL_HEADS, DENSE_LATENT_DIM)
        or cache.dtype != torch.int8
        or cache.ndim != 4
        or cache.shape[2] != 1
        or cache.shape[-1] != C8_PACKED_DIM
        or not cache.is_contiguous()
        or metadata.block_size != DENSE_KERNEL_BLOCK_SIZE
        or metadata.attn_mask is None
        or metadata.attn_mask.shape != (DENSE_SELECTION_BUDGET, DENSE_SELECTION_BUDGET)
        or metadata.attn_mask.dtype not in (torch.bool, torch.int8)
        or metadata.block_table.shape[0] <= plan.request_index
        or metadata.block_table.shape[1] < plan.block_table.shape[1]
    ):
        return None
    cache = cache.view(-1, DENSE_KERNEL_BLOCK_SIZE, 1, C8_PACKED_DIM)
    pages = metadata.block_table[plan.request_index, : plan.block_table.shape[1]].long()
    packed = torch.index_select(cache, 0, pages)
    scales = packed[..., DENSE_LATENT_DIM:].contiguous().view(torch.float32).repeat_interleave(C8_SCALE_TILE, -1)
    latent = (packed[..., :DENSE_LATENT_DIM].float() * scales).bfloat16()
    latent = latent.view(-1, 1, DENSE_KERNEL_BLOCK_SIZE, DENSE_LATENT_DIM)
    dense_query = torch.index_select(query, 0, plan.query_rows)
    # FIA's floating MLA ABI requires a RoPE input on this stack. Zero
    # channels preserve NoPE logits, including the model's original scale.
    query_rope = query.new_zeros((plan.query_rows.numel(), DENSE_LOCAL_HEADS, FIA_COMPAT_ROPE_DIM))
    key_rope = query.new_zeros((*latent.shape[:-1], FIA_COMPAT_ROPE_DIM))
    dense_output = torch_npu.npu_fused_infer_attention_score(
        dense_query,
        latent,
        latent,
        num_heads=DENSE_LOCAL_HEADS,
        num_key_value_heads=1,
        input_layout="TND",
        atten_mask=metadata.attn_mask,
        sparse_mode=3,
        scale=scale,
        block_table=plan.block_table,
        block_size=DENSE_KERNEL_BLOCK_SIZE,
        actual_seq_lengths=[plan.query_rows.numel()],
        actual_seq_lengths_kv=[plan.seq_len],
        query_rope=query_rope,
        key_rope=key_rope,
    )[0]
    output = torch.empty_like(query)
    output.index_copy_(0, plan.query_rows, dense_output)
    if plan.sparse_rows.numel():
        sparse_output = sparse_attention(
            torch.index_select(query, 0, plan.sparse_rows),
            None,
            (cache,),
            torch.index_select(indices, 0, plan.sparse_rows),
            metadata,
            plan.sparse_cum_query_lens,
            key_lengths,
        )
        output.index_copy_(0, plan.sparse_rows, sparse_output)
    return output
