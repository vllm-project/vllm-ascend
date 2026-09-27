# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Small-expert MoE routing with EPLB mapping and load recording.

The selection for each token is independent, while the load histogram is
shared. One program handles several tokens, reduces their physical-expert
counts locally, and performs at most one atomic addition per physical expert.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _moe_gating_topk_map_record_kernel(
    logits_ptr,
    bias_ptr,
    table_ptr,
    load_ptr,
    record_enabled_ptr,
    valid_tokens_ptr,
    weights_ptr,
    physical_ids_ptr,
    tokens,
    experts,
    table_rows,
    physical_experts,
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
    row = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    expert = tl.arange(0, BLOCK_E)
    physical = tl.arange(0, BLOCK_P)
    slot = tl.arange(0, BLOCK_K)
    row_mask = row < tokens
    expert_mask = expert < experts
    valid = row_mask[:, None] & expert_mask[None, :]

    logits = tl.load(
        logits_ptr + row[:, None] * experts + expert[None, :],
        mask=valid,
        other=0,
    ).to(tl.float32)
    if SOFTMAX:
        safe_logits = tl.where(valid, logits, float("-inf"))
        exponent = tl.exp(safe_logits - tl.max(safe_logits, 1)[:, None])
        score = exponent / tl.sum(exponent, 1)[:, None]
    else:
        score = tl.sigmoid(logits)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + expert, mask=expert_mask, other=0).to(tl.float32)
        ranking = score + bias[None, :]
    else:
        ranking = score
    ranking = tl.where(valid, ranking, float("-inf"))

    counts = tl.full((BLOCK_T, BLOCK_P), 0, tl.int32)
    selected_scores = tl.full((BLOCK_T, BLOCK_K), 0, tl.float32)
    recording = tl.load(record_enabled_ptr) != 0
    if VALID_IS_TENSOR:
        valid_tokens = tl.load(valid_tokens_ptr)
    else:
        valid_tokens = valid_tokens_ptr
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
        counts += ((physical[None, :] == local_id[:, None]) & record_row[:, None]).to(tl.int32)
        ranking = tl.where(expert[None, :] == index[:, None], float("-inf"), ranking)

    denominator = tl.sum(selected_scores, 1) + 1e-20
    tl.store(
        weights_ptr + row[:, None] * K + slot[None, :],
        selected_scores / denominator[:, None] * scaling,
        mask=row_mask[:, None] & (slot[None, :] < K),
    )
    program_counts = tl.sum(counts, 0)
    tl.atomic_add(
        load_ptr + local_expert_start + physical,
        program_counts,
        mask=(physical < local_expert_count)
        & (local_expert_start + physical < physical_experts)
        & (program_counts != 0),
    )


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
    tensors = (routing_table, expert_load, record_enabled, bias)
    if isinstance(valid_tokens, torch.Tensor):
        tensors += (valid_tokens,)
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
    weights = torch.empty((tokens, k), dtype=logits.dtype, device=logits.device)
    physical_ids = torch.empty((tokens, k), dtype=torch.int32, device=logits.device)
    if tokens == 0:
        return weights, physical_ids
    block_t = 2 if tokens <= 512 else 32
    _moe_gating_topk_map_record_kernel[(triton.cdiv(tokens, block_t),)](
        logits,
        bias if bias is not None else logits,
        routing_table,
        expert_load,
        record_enabled,
        valid_tokens,
        weights,
        physical_ids,
        tokens,
        experts,
        routing_table.shape[0],
        expert_load.numel(),
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
    return weights, physical_ids
