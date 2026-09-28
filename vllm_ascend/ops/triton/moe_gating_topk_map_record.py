# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Small-expert MoE routing with grid-owned EPLB records.

Each routing program owns a contiguous token range and writes one record
for the EP-local physical experts. A second, single-writer program reduces
the records into the cumulative load.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit
def _moe_gating_topk_map_record_kernel(
    logits_ptr,
    bias_ptr,
    table_ptr,
    record_enabled_ptr,
    valid_tokens_ptr,
    weights_ptr,
    physical_ids_ptr,
    grid_records_ptr,
    tokens,
    experts,
    table_rows,
    local_expert_start,
    local_expert_count,
    scaling,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SOFTMAX: tl.constexpr,
    VALID_IS_TENSOR: tl.constexpr,
):
    grid_id = tl.program_id(0)
    num_grids = tl.num_programs(0)
    tokens_per_grid = tokens // num_grids
    extra_tokens = tokens % num_grids
    token_start = grid_id * tokens_per_grid + tl.minimum(grid_id, extra_tokens)
    token_end = token_start + tokens_per_grid + (grid_id < extra_tokens).to(tl.int32)
    expert = tl.arange(0, BLOCK_E)
    physical = tl.arange(0, BLOCK_P)
    slot = tl.arange(0, BLOCK_K)
    expert_mask = expert < experts
    grid_record = tl.full((BLOCK_P,), 0, tl.int32)
    recording = tl.load(record_enabled_ptr) != 0
    if VALID_IS_TENSOR:
        valid_tokens = tl.load(valid_tokens_ptr)
    else:
        valid_tokens = valid_tokens_ptr
    if HAS_BIAS:
        bias = tl.load(bias_ptr + expert, mask=expert_mask, other=0).to(tl.float32)

    for tile_start in tl.range(token_start, token_end, BLOCK_T):
        row = tile_start + tl.arange(0, BLOCK_T)
        row_mask = row < token_end
        valid = row_mask[:, None] & expert_mask[None, :]
        logits = tl.load(
            logits_ptr + row[:, None] * experts + expert[None, :],
            mask=valid,
            other=0,
        ).to(tl.float32)
        if SOFTMAX:
            safe_logits = tl.where(valid, logits, float("-inf"))
            max_logits = tl.max(safe_logits, 1)
            safe_max = tl.where(max_logits == float("-inf"), 0.0, max_logits)
            exponent = tl.exp(safe_logits - safe_max[:, None])
            score = exponent / (tl.sum(exponent, 1)[:, None] + 1e-20)
        else:
            score = tl.sigmoid(logits)

        if HAS_BIAS:
            ranking = score + bias[None, :]
        else:
            ranking = score
        ranking = tl.where(valid, ranking, float("-inf"))

        tile_hits = tl.full((BLOCK_T, BLOCK_P), 0, tl.int32)
        selected_scores = tl.full((BLOCK_T, BLOCK_K), 0, tl.float32)
        record_row = row_mask & (row < valid_tokens) & recording

        for rank in tl.static_range(K):
            best = tl.max(ranking, 1)
            index = tl.min(tl.where(ranking == best[:, None], expert[None, :], BLOCK_E), 1)
            picked = tl.sum(tl.where(expert[None, :] == index[:, None], score, 0), 1)
            mapped = tl.load(
                table_ptr + (row % table_rows) * experts + index,
                mask=row_mask,
                other=-1,
            )
            tl.store(physical_ids_ptr + row * K + rank, mapped, mask=row_mask)
            selected_scores = tl.where(slot[None, :] == rank, picked[:, None], selected_scores)
            local_id = mapped - local_expert_start
            tile_hits += ((physical[None, :] == local_id[:, None]) & record_row[:, None]).to(tl.int32)
            ranking = tl.where(expert[None, :] == index[:, None], float("-inf"), ranking)

        denominator = tl.sum(selected_scores, 1) + 1e-20
        tl.store(
            weights_ptr + row[:, None] * K + slot[None, :],
            selected_scores / denominator[:, None] * scaling,
            mask=row_mask[:, None] & (slot[None, :] < K),
        )
        grid_record += tl.sum(tile_hits, 0)

    # One ordinary store per grid-owned EP-local physical expert.
    tl.store(
        grid_records_ptr + grid_id * local_expert_count + physical,
        grid_record,
        mask=physical < local_expert_count,
    )


@triton.jit
def _reduce_grid_records_kernel(
    grid_records_ptr,
    load_ptr,
    record_enabled_ptr,
    num_grids,
    local_expert_start,
    local_expert_count,
    BLOCK_GRID: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    grid = tl.arange(0, BLOCK_GRID)
    physical = tl.arange(0, BLOCK_P)
    mask = (grid[:, None] < num_grids) & (physical[None, :] < local_expert_count)
    partial = tl.load(
        grid_records_ptr + grid[:, None] * local_expert_count + physical[None, :],
        mask=mask,
        other=0,
    )
    record = tl.sum(partial, 0)
    if tl.load(record_enabled_ptr) != 0:
        load_ptrs = load_ptr + local_expert_start + physical
        old_load = tl.load(load_ptrs, mask=physical < local_expert_count, other=0)
        tl.store(load_ptrs, old_load + record, mask=physical < local_expert_count)


def moe_gating_topk_map_record(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    routing_table: torch.Tensor,
    expert_load: torch.Tensor,
    record_enabled: torch.Tensor,
    valid_tokens: torch.Tensor | int,
    *,
    k: int,
    scoring: str,
    routed_scaling_factor: float = 1.0,
    local_expert_start: int = 0,
    local_expert_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return routing weights and physical IDs, accumulating valid-token load.

    This initial implementation supports the ungrouped, renormalized route.
    The routing table is periodic over token rows, as in the EPLB path.
    """
    if logits.ndim != 2 or not logits.is_contiguous():
        raise ValueError("logits must be contiguous [tokens, experts]")
    tokens, experts = logits.shape
    if not 1 <= k <= experts:
        raise ValueError("k must be between 1 and the expert count")
    if scoring not in ("softmax", "sigmoid"):
        raise ValueError("scoring must be softmax or sigmoid")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("logits must be float16, bfloat16 or float32")
    if bias is not None and (bias.shape != (experts,) or not bias.is_contiguous() or bias.dtype != logits.dtype):
        raise ValueError("bias must be contiguous [experts] with logits dtype")
    if routing_table.ndim != 2 or routing_table.shape[1] != experts or not routing_table.is_contiguous():
        raise ValueError("routing_table must be contiguous [rows, experts]")
    if routing_table.dtype != torch.int32 or expert_load.dtype != torch.int32:
        raise ValueError("routing_table and expert_load must be int32")
    if expert_load.ndim != 1 or expert_load.stride(0) != 1:
        raise ValueError("expert_load must be a contiguous one-dimensional view")
    if record_enabled.numel() != 1:
        raise ValueError("record_enabled must contain one device value")
    if isinstance(valid_tokens, torch.Tensor) and valid_tokens.numel() != 1:
        raise ValueError("valid_tokens must contain one device value")
    tensors = (
        routing_table,
        expert_load,
        record_enabled,
        bias,
        valid_tokens if isinstance(valid_tokens, torch.Tensor) else None,
    )
    if any(tensor is not None and tensor.device != logits.device for tensor in tensors):
        raise ValueError("all tensors must be on the same device")
    if routing_table.shape[0] == 0:
        raise ValueError("routing_table must have at least one row")
    if isinstance(valid_tokens, int) and not 0 <= valid_tokens <= tokens:
        raise ValueError("valid_tokens must be between zero and tokens")
    if local_expert_count is None:
        local_expert_count = expert_load.numel() - local_expert_start
    if (
        local_expert_start < 0
        or local_expert_count < 0
        or local_expert_start + local_expert_count > expert_load.numel()
    ):
        raise ValueError("local expert range exceeds expert_load")
    if local_expert_count != experts:
        raise ValueError("grid-record fast path requires matching logical and local physical expert counts")
    weights = torch.empty((tokens, k), dtype=logits.dtype, device=logits.device)
    physical_ids = torch.empty((tokens, k), dtype=torch.int32, device=logits.device)
    if tokens == 0:
        return weights, physical_ids
    init_device_properties_triton()
    if tokens <= 64:
        block_t = 2
        num_grids = triton.cdiv(tokens, block_t)
    else:
        block_t = 8 if tokens <= 512 else 32
        num_grids = min(tokens, get_vectorcore_num())
    grid_records = torch.empty((num_grids, local_expert_count), dtype=torch.int32, device=logits.device)
    _moe_gating_topk_map_record_kernel[(num_grids,)](
        logits,
        bias if bias is not None else logits,
        routing_table,
        record_enabled,
        valid_tokens,
        weights,
        physical_ids,
        grid_records,
        tokens,
        experts,
        routing_table.shape[0],
        local_expert_start,
        local_expert_count,
        routed_scaling_factor,
        K=k,
        BLOCK_T=block_t,
        BLOCK_E=triton.next_power_of_2(experts),
        BLOCK_K=triton.next_power_of_2(k),
        BLOCK_P=triton.next_power_of_2(max(local_expert_count, 1)),
        HAS_BIAS=bias is not None,
        SOFTMAX=scoring == "softmax",
        VALID_IS_TENSOR=isinstance(valid_tokens, torch.Tensor),
        num_warps=4,
    )
    _reduce_grid_records_kernel[(1,)](
        grid_records,
        expert_load,
        record_enabled,
        num_grids,
        local_expert_start,
        local_expert_count,
        BLOCK_GRID=triton.next_power_of_2(num_grids),
        BLOCK_P=triton.next_power_of_2(local_expert_count),
        num_warps=4,
    )
    return weights, physical_ids
