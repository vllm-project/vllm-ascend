# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.triton_utils import tl, triton

_MIN_KV_PARTITION = 128
_MAX_KV_SPLITS = 16
_TARGET_QUERY_TASKS = 32
_MERGE_BLOCK_D = 128


def choose_sfa_split_count(query_count: int, topk_count: int) -> int:
    """Choose a power-of-two split while keeping at least 128 KV per task."""
    if query_count <= 0 or topk_count < 2 * _MIN_KV_PARTITION:
        return 1
    max_splits = min(_MAX_KV_SPLITS, topk_count // _MIN_KV_PARTITION)
    split = 1
    while split * 2 <= max_splits:
        current_distance = abs(query_count * split - _TARGET_QUERY_TASKS)
        next_distance = abs(query_count * split * 2 - _TARGET_QUERY_TASKS)
        if next_distance > current_distance:
            break
        split *= 2
    return split if topk_count % split == 0 else 1


def can_use_sfa_split_kv(
    query: torch.Tensor,
    query_rope: torch.Tensor | None,
    key: torch.Tensor,
    sparse_indices: torch.Tensor,
    available_kv_tokens: int,
) -> bool:
    if not getattr(query, "is_npu", False) or query_rope is None:
        return False
    if query.ndim != 3 or query_rope.ndim != 3 or sparse_indices.ndim != 3:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype:
        return False
    if query_rope.dtype != query.dtype:
        return False
    if sparse_indices.dtype != torch.int32 or sparse_indices.shape[1] != 1:
        return False
    if query.shape[:2] != query_rope.shape[:2]:
        return False
    # The A5 SFA fast path assumes that every row in a short sparse-index
    # tensor is full. Keep early-context padding on the original operator.
    if available_kv_tokens < sparse_indices.shape[-1]:
        return False
    return choose_sfa_split_count(query.shape[0], sparse_indices.shape[-1]) > 1


@triton.jit
def _expand_sfa_query_kernel(
    query,
    query_rope,
    query_parts,
    query_rope_parts,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_rt,
    stride_rh,
    stride_rd,
    head_count: tl.constexpr,
    query_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    split_count: tl.constexpr,
    BLOCK_QD: tl.constexpr,
    BLOCK_RD: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    query_idx = tl.program_id(axis=0)
    head_idx = tl.program_id(axis=1)
    split_idx = tl.arange(0, BLOCK_P)
    query_d_idx = tl.arange(0, BLOCK_QD)
    rope_d_idx = tl.arange(0, BLOCK_RD)
    split_mask = split_idx < split_count
    query_d_mask = query_d_idx < query_dim
    rope_d_mask = rope_d_idx < rope_dim

    query_row = tl.load(
        query + query_idx * stride_qt + head_idx * stride_qh + query_d_idx * stride_qd,
        mask=query_d_mask,
        other=0.0,
    )
    rope_row = tl.load(
        query_rope + query_idx * stride_rt + head_idx * stride_rh + rope_d_idx * stride_rd,
        mask=rope_d_mask,
        other=0.0,
    )
    output_row = (query_idx * split_count + split_idx[:, None]) * head_count + head_idx
    tl.store(
        query_parts + output_row * query_dim + query_d_idx[None, :],
        query_row[None, :],
        mask=split_mask[:, None] & query_d_mask[None, :],
    )
    tl.store(
        query_rope_parts + output_row * rope_dim + rope_d_idx[None, :],
        rope_row[None, :],
        mask=split_mask[:, None] & rope_d_mask[None, :],
    )


@triton.jit
def _partition_sfa_indices_kernel(
    sparse_indices,
    index_parts,
    query_lengths,
    query_lengths_parts,
    stride_it,
    stride_ih,
    stride_ik,
    query_count: tl.constexpr,
    batch_count: tl.constexpr,
    kv_head_count: tl.constexpr,
    topk_count: tl.constexpr,
    part_capacity: tl.constexpr,
    split_count: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    kv_head_idx = tl.program_id(axis=1)
    offsets = tl.arange(0, BLOCK_K)
    input_mask = (row_idx < query_count) & (offsets < topk_count)
    input_row = tl.load(
        sparse_indices + row_idx * stride_it + kv_head_idx * stride_ih + offsets * stride_ik,
        mask=input_mask,
        other=-1,
    )
    valid_count = tl.sum((input_row >= 0).to(tl.int32), axis=0)
    valid_per_part = (valid_count + split_count - 1) // split_count

    split_idx = offsets // part_capacity
    index_in_part = offsets % part_capacity
    source_idx = split_idx * valid_per_part + index_in_part
    source_mask = (
        (row_idx < query_count) & (offsets < topk_count) & (index_in_part < valid_per_part) & (source_idx < valid_count)
    )
    selected = tl.load(
        sparse_indices + row_idx * stride_it + kv_head_idx * stride_ih + source_idx * stride_ik,
        mask=source_mask,
        other=-1,
    )
    output_offsets = ((row_idx * split_count + split_idx) * kv_head_count + kv_head_idx) * part_capacity + index_in_part
    tl.store(index_parts + output_offsets, selected, mask=(row_idx < query_count) & (offsets < topk_count))

    length_mask = (kv_head_idx == 0) & (row_idx < batch_count)
    query_end = tl.load(query_lengths + row_idx, mask=length_mask, other=0)
    tl.store(query_lengths_parts + row_idx, query_end * split_count, mask=length_mask)


@triton.jit
def _merge_sfa_partials_kernel(
    partial_output,
    partial_max,
    partial_sum,
    output,
    output_max,
    output_sum,
    head_count: tl.constexpr,
    head_dim: tl.constexpr,
    split_count: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    query_idx = tl.program_id(axis=0)
    head_idx = tl.program_id(axis=1)
    d_block_idx = tl.program_id(axis=2)
    split_idx = tl.arange(0, BLOCK_P)
    d_idx = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    split_mask = split_idx < split_count
    d_mask = d_idx < head_dim

    stat_offsets = (query_idx * split_count + split_idx) * head_count + head_idx
    local_max = tl.load(partial_max + stat_offsets, mask=split_mask, other=-float("inf"))
    local_sum = tl.load(partial_sum + stat_offsets, mask=split_mask, other=0.0)
    valid = split_mask & (local_sum > 0.0)
    global_max = tl.max(tl.where(valid, local_max, -float("inf")), axis=0)
    max_delta = tl.where(valid, local_max - global_max, 0.0)
    weights = tl.where(valid, local_sum * tl.exp(max_delta), 0.0)
    global_sum = tl.sum(weights, axis=0)

    partial_offsets = ((query_idx * split_count + split_idx[:, None]) * head_count + head_idx) * head_dim + d_idx[
        None, :
    ]
    local_output = tl.load(
        partial_output + partial_offsets,
        mask=split_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    numerator = tl.sum(local_output * weights[:, None], axis=0)
    denominator = tl.where(global_sum > 0.0, global_sum, 1.0)
    merged = tl.where(global_sum > 0.0, numerator / denominator, 0.0)
    output_offsets = (query_idx * head_count + head_idx) * head_dim + d_idx
    tl.store(output + output_offsets, merged, mask=d_mask)

    if d_block_idx == 0:
        stat_out_offset = query_idx * head_count + head_idx
        tl.store(output_max + stat_out_offset, global_max)
        tl.store(output_sum + stat_out_offset, global_sum)


def _prepare_sfa_split_inputs(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    sparse_indices: torch.Tensor,
    query_lengths: torch.Tensor,
    split_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_count, head_count, query_dim = query.shape
    rope_dim = query_rope.shape[-1]
    kv_head_count = sparse_indices.shape[1]
    topk_count = sparse_indices.shape[-1]
    part_capacity = topk_count // split_count
    query_parts = torch.empty(
        (query_count * split_count, head_count, query_dim),
        dtype=query.dtype,
        device=query.device,
    )
    query_rope_parts = torch.empty(
        (query_count * split_count, head_count, rope_dim),
        dtype=query_rope.dtype,
        device=query_rope.device,
    )
    index_parts = torch.empty(
        (query_count * split_count, kv_head_count, part_capacity),
        dtype=sparse_indices.dtype,
        device=sparse_indices.device,
    )
    query_lengths_parts = torch.empty_like(query_lengths)

    _expand_sfa_query_kernel[(query_count, head_count)](
        query,
        query_rope,
        query_parts,
        query_rope_parts,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query_rope.stride(0),
        query_rope.stride(1),
        query_rope.stride(2),
        head_count=head_count,
        query_dim=query_dim,
        rope_dim=rope_dim,
        split_count=split_count,
        BLOCK_QD=triton.next_power_of_2(query_dim),
        BLOCK_RD=triton.next_power_of_2(rope_dim),
        BLOCK_P=triton.next_power_of_2(split_count),
        num_warps=4,
        num_stages=1,
    )
    _partition_sfa_indices_kernel[(max(query_count, query_lengths.numel()), kv_head_count)](
        sparse_indices,
        index_parts,
        query_lengths,
        query_lengths_parts,
        sparse_indices.stride(0),
        sparse_indices.stride(1),
        sparse_indices.stride(2),
        query_count=query_count,
        batch_count=query_lengths.numel(),
        kv_head_count=kv_head_count,
        topk_count=topk_count,
        part_capacity=part_capacity,
        split_count=split_count,
        BLOCK_K=triton.next_power_of_2(topk_count),
        num_warps=4,
        num_stages=1,
    )
    return query_parts, query_rope_parts, index_parts, query_lengths_parts


def _merge_sfa_partials(
    partial_output: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    query_count: int,
    split_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    head_count = partial_output.shape[1]
    head_dim = partial_output.shape[2]
    output = torch.empty(
        (query_count, head_count, head_dim),
        dtype=partial_output.dtype,
        device=partial_output.device,
    )
    output_max = torch.empty((1, query_count, head_count), dtype=torch.float32, device=partial_output.device)
    output_sum = torch.empty_like(output_max)
    _merge_sfa_partials_kernel[(query_count, head_count, triton.cdiv(head_dim, _MERGE_BLOCK_D))](
        partial_output,
        partial_max,
        partial_sum,
        output,
        output_max,
        output_sum,
        head_count=head_count,
        head_dim=head_dim,
        split_count=split_count,
        BLOCK_D=_MERGE_BLOCK_D,
        BLOCK_P=triton.next_power_of_2(split_count),
        num_warps=4,
        num_stages=1,
    )
    return output, output_max, output_sum


def sparse_flash_attention_split_kv(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale_value: float,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    query_rope: torch.Tensor,
    key_rope: torch.Tensor,
    return_softmax_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_count = query.shape[0]
    split_count = choose_sfa_split_count(query_count, sparse_indices.shape[-1])
    if split_count <= 1:
        raise ValueError("sparse_flash_attention_split_kv requires at least two KV partitions")
    query_parts, query_rope_parts, index_parts, query_lengths_parts = _prepare_sfa_split_inputs(
        query,
        query_rope,
        sparse_indices,
        actual_seq_lengths_query,
        split_count,
    )
    partial_output, partial_max, partial_sum = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query_parts,
        key=key,
        value=value,
        sparse_indices=index_parts,
        scale_value=scale_value,
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=query_lengths_parts,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        query_rope=query_rope_parts,
        key_rope=key_rope,
        layout_query="TND",
        layout_kv="PA_BSND",
        # LightningIndexer already emits causally valid indices. Multiplying
        # S1 by split_count changes query coordinates, so masking is disabled
        # for the partial calls and the selected set remains the sole mask.
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    result = _merge_sfa_partials(
        partial_output,
        partial_max,
        partial_sum,
        query_count,
        split_count,
    )
    return result if return_softmax_lse else result[0]
