"""Fused candidate-pool construction for Qwen DFlash and DeepSeek DSpark.

The implementation keeps the original two-kernel contract: one kernel builds
per-row candidate flags, a second kernel reduces the flags into a batch union
and masks suffix rows. This is not a Top-M implementation.

Only the DSpark path (`dspark_route`) is wired into activation routing. The
Top-8 BF16 path (`topm_route`) is an unwired prototype reserved for a
follow-up PR: it targets a `DeviceOperator.moe_gating_top_k` adaptor API that
is not yet available in this tree.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as ascend

from vllm_ascend.device.device_op import DeviceOperator

USES_TOPM = False

BF16_NEG = -3.3895313892515355e38
FP32_NEG = -3.4028234663852886e38
BLOCK_EXPERTS = 32


@triton.jit
def _pool_candidates_bf16(
    logits_ptr,
    flags_ptr,
    roles_ptr,
    stride_row,
    stride_expert,
    NUM_EXPERTS: tl.constexpr,
    SUFFIX_TOP_K: tl.constexpr,
):
    """Published BF16 Top8/Top2 path with a stable lower-ID tie break."""
    with ascend.scope(core_mode="vector"):
        row = tl.program_id(0)
        expert_ids = tl.arange(0, NUM_EXPERTS)
        x = tl.load(logits_ptr + row * stride_row + expert_ids * stride_expert)
        bits = (x.to(tl.float32).to(tl.int32, bitcast=True) >> 16) & 65535
        bits = tl.where(x == 0, 0, bits)
        ordered = tl.where(bits >= 32768, 65535 - bits, bits + 32768)
        # NUM_EXPERTS <= 256 keeps the largest integer key exactly
        # representable in float32 (65535 * 256 + 255 == 2**24 - 1).
        keys = (ordered * NUM_EXPERTS + NUM_EXPERTS - 1 - expert_ids).to(tl.float32)
        sorted_keys = ascend.sort(keys, descending=True)
        role = tl.load(roles_ptr + row)
        rank = tl.where(role == 1, 7, SUFFIX_TOP_K - 1)
        threshold = tl.sum(tl.where(expert_ids == rank, sorted_keys, 0), 0)
        tl.store(
            flags_ptr + row * NUM_EXPERTS + expert_ids,
            (keys >= threshold) & (role != 0),
        )


@triton.jit
def _pool_candidates_dspark(
    logits_ptr,
    bias_ptr,
    flags_ptr,
    raw_scores_ptr,
    roles_ptr,
    stride_row,
    stride_expert,
    NUM_EXPERTS: tl.constexpr,
    ROUTE_TOP_K: tl.constexpr,
    SUFFIX_TOP_K: tl.constexpr,
):
    """K1: sqrt-softplus + bias + stable Top6/Top2 candidate flags."""
    with ascend.scope(core_mode="vector"):
        row = tl.program_id(0)
        expert_ids = tl.arange(0, NUM_EXPERTS)
        x = tl.load(logits_ptr + row * stride_row + expert_ids * stride_expert).to(tl.float32)
        # Match moe_gating_top_k_hash's AscendC operation order exactly:
        # Exp -> Adds(1) -> Ln -> Sqrt. RawScores excludes correction bias.
        softplus = tl.log(tl.exp(x) + 1.0)
        raw = tl.sqrt(softplus)
        score = raw + tl.load(bias_ptr + expert_ids).to(tl.float32)
        tl.store(raw_scores_ptr + row * NUM_EXPERTS + expert_ids, raw)

        sorted_scores = ascend.sort(score, descending=True)
        role = tl.load(roles_ptr + row)
        limit = tl.where(
            role == 1,
            ROUTE_TOP_K,
            tl.where(role == 2, SUFFIX_TOP_K, 0),
        )
        rank = tl.maximum(limit - 1, 0)
        threshold = tl.sum(tl.where(expert_ids == rank, sorted_scores, 0.0), 0)
        greater = score > threshold
        equal = score == threshold
        num_greater = tl.sum(greater.to(tl.int32), 0)
        ties_needed = limit - num_greater
        # Expert IDs are the lane order. Prefix-summing equal scores selects
        # lower IDs first and guarantees exactly `limit` flags on ties.
        tie_rank = tl.cumsum(equal.to(tl.int32), axis=0)
        selected = greater | (equal & (tie_rank <= ties_needed))
        tl.store(
            flags_ptr + row * NUM_EXPERTS + expert_ids,
            selected & (role != 0),
        )


@triton.jit
def _batch_union_apply_bf16(
    logits_ptr,
    flags_ptr,
    roles_ptr,
    masked_logits_ptr,
    NUM_ROWS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    stride_row,
    stride_expert,
    NUM_EXPERTS: tl.constexpr,
):
    with ascend.scope(core_mode="vector"):
        row_ids = tl.arange(0, ROW_BLOCK)
        expert_ids = tl.arange(0, NUM_EXPERTS)
        flags = tl.load(
            flags_ptr + row_ids[:, None] * NUM_EXPERTS + expert_ids[None, :],
            row_ids[:, None] < NUM_ROWS,
            0,
        )
        pool = tl.max(flags, 0) != 0
        row_roles = tl.load(roles_ptr + row_ids, row_ids < NUM_ROWS, 0)
        raw = tl.load(
            logits_ptr + row_ids[:, None] * stride_row + expert_ids[None, :] * stride_expert,
            row_ids[:, None] < NUM_ROWS,
            0,
        )
        masked = tl.where(
            (row_roles[:, None] == 2) & ~pool[None, :],
            BF16_NEG,
            raw.to(tl.float32),
        )
        tl.store(
            masked_logits_ptr + row_ids[:, None] * NUM_EXPERTS + expert_ids[None, :],
            masked,
            row_ids[:, None] < NUM_ROWS,
        )


@triton.jit
def _batch_union_apply_dspark(
    raw_scores_ptr,
    bias_ptr,
    flags_ptr,
    roles_ptr,
    masked_scores_ptr,
    NUM_ROWS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    """K2: reduce the full padded batch union and mask suffix scores."""
    with ascend.scope(core_mode="vector"):
        expert_ids = tl.program_id(0) * BLOCK_EXPERTS + tl.arange(0, BLOCK_EXPERTS)
        row_ids = tl.arange(0, ROW_BLOCK)
        expert_mask = expert_ids < NUM_EXPERTS
        row_mask = row_ids < NUM_ROWS
        flags = tl.load(
            flags_ptr + row_ids[:, None] * NUM_EXPERTS + expert_ids[None, :],
            row_mask[:, None] & expert_mask[None, :],
            0,
        )
        pool = tl.max(flags, 0) != 0
        row_roles = tl.load(roles_ptr + row_ids, row_mask, 0)
        raw = tl.load(
            raw_scores_ptr + row_ids[:, None] * NUM_EXPERTS + expert_ids[None, :],
            row_mask[:, None] & expert_mask[None, :],
            0.0,
        )
        bias = tl.load(bias_ptr + expert_ids, expert_mask, 0.0)
        selection = raw + bias[None, :]
        masked = tl.where(
            (row_roles[:, None] == 2) & ~pool[None, :],
            FP32_NEG,
            selection,
        )
        tl.store(
            masked_scores_ptr + row_ids[:, None] * NUM_EXPERTS + expert_ids[None, :],
            masked,
            row_mask[:, None] & expert_mask[None, :],
        )


def allocate_workspace(logits, m=None):
    """Allocate the published BF16 path workspace.

    `m` is a deprecated compatibility argument. It is deliberately unused.
    """
    del m
    n = logits.shape[0]
    return (
        torch.empty(n, logits.shape[1], device=logits.device, dtype=torch.int8),
        torch.empty_like(logits, memory_format=torch.contiguous_format),
    )


def allocate_dspark_workspace(logits):
    n, num_experts = logits.shape
    return (
        torch.empty(n, num_experts, device=logits.device, dtype=torch.int8),
        torch.empty(n, num_experts, device=logits.device, dtype=torch.float32),
        torch.empty(n, num_experts, device=logits.device, dtype=torch.float32),
    )


def _validate_fused_route_roles(logits, roles):
    if roles.ndim != 1 or roles.numel() != logits.shape[0]:
        raise ValueError("Router roles must be a 1D tensor matching logits rows")
    if roles.device != logits.device:
        raise ValueError("Router roles and logits must be on the same device")
    if roles.dtype not in (torch.int32, torch.int64):
        raise ValueError("Router roles must use int32 or int64")
    if not roles.is_contiguous():
        raise ValueError("Router roles must be contiguous")


def _validate_workspace(workspace, expected, logits, name):
    if not isinstance(workspace, (tuple, list)) or len(workspace) != len(expected):
        raise ValueError(f"{name} workspace must contain {len(expected)} tensors")
    for index, (tensor, shape, dtype) in enumerate(zip(workspace, expected)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} workspace entry {index} must be a tensor")
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != logits.device:
            raise ValueError(
                f"{name} workspace entry {index} must have shape={shape}, "
                f"dtype={dtype}, device={logits.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} workspace entry {index} must be contiguous")


def topm_route(
    logits,
    roles,
    m=None,
    workspace=None,
    suffix_pool_top_k=2,
):
    """Published BF16 Top8 path; `m` is deprecated and ignored."""
    if logits.dtype != torch.bfloat16 or logits.ndim != 2 or logits.shape[1] not in (128, 256):
        raise ValueError("Fused pool requires BF16 [N,128] or [N,256]")
    if not 1 <= suffix_pool_top_k <= 8:
        raise ValueError("suffix_pool_top_k must be in [1, 8]")
    _validate_fused_route_roles(logits, roles)
    # the fused kernel does not consume non-unit strides correctly
    # (validated on A3). Production router logits are always contiguous;
    # guard here so strided inputs are copied once instead of mis-read.
    if not logits.is_contiguous():
        logits = logits.contiguous()
    n, num_experts = logits.shape
    if workspace is None:
        workspace = allocate_workspace(logits)
    else:
        _validate_workspace(
            workspace,
            ((logits.shape[0], num_experts), (logits.shape[0], num_experts)),
            logits,
            "BF16",
        )
    flags, masked = workspace
    _pool_candidates_bf16[(n,)](
        logits,
        flags,
        roles,
        logits.stride(0),
        logits.stride(1),
        NUM_EXPERTS=num_experts,
        SUFFIX_TOP_K=suffix_pool_top_k,
        multibuffer=False,
    )
    _batch_union_apply_bf16[(1,)](
        logits,
        flags,
        roles,
        masked,
        n,
        triton.next_power_of_2(n),
        logits.stride(0),
        logits.stride(1),
        NUM_EXPERTS=num_experts,
        multibuffer=False,
    )
    weights, ids, _ = DeviceOperator.moe_gating_top_k(
        masked,
        k=8,
        k_group=1,
        group_count=1,
        group_select_mode=1,
        renorm=1,
        norm_type=0,
        out_flag=False,
        routed_scaling_factor=1.0,
        eps=1e-20,
        bias_opt=None,
    )
    return weights, ids


def dspark_route(
    logits,
    roles,
    e_score_correction_bias,
    workspace=None,
    route_top_k=6,
    suffix_pool_top_k=2,
    renormalize=True,
    routed_scaling_factor=1.5,
):
    """DeepSeek-V4 Top6 sqrt-softplus fused path."""
    if logits.dtype != torch.float32 or logits.ndim != 2 or logits.shape[1] != 256:
        raise ValueError("DSpark fused pool requires FP32 [N,256]")
    if e_score_correction_bias is None or e_score_correction_bias.shape != (256,):
        raise ValueError("DSpark fused pool requires a [256] correction bias")
    if e_score_correction_bias.device != logits.device:
        raise ValueError("DSpark correction bias must be on the logits device")
    if e_score_correction_bias.dtype != torch.float32:
        raise ValueError("DSpark correction bias must use float32")
    if route_top_k != 6 or suffix_pool_top_k != 2:
        raise ValueError("DSpark fused pool requires route_top_k=6 and suffix_pool_top_k=2")
    if not renormalize:
        raise ValueError("DSpark fused pool requires renormalize=True")
    _validate_fused_route_roles(logits, roles)
    # the fused kernel does not consume non-unit strides correctly
    # (validated on A3). Production router logits are always contiguous;
    # guard here so strided inputs are copied once instead of mis-read.
    if not logits.is_contiguous():
        logits = logits.contiguous()
    n, num_experts = logits.shape
    if not e_score_correction_bias.is_contiguous():
        e_score_correction_bias = e_score_correction_bias.contiguous()
    if workspace is None:
        workspace = allocate_dspark_workspace(logits)
    else:
        _validate_workspace(
            workspace,
            (
                (logits.shape[0], num_experts),
                (logits.shape[0], num_experts),
                (logits.shape[0], num_experts),
            ),
            logits,
            "DSpark",
        )
    flags, raw_scores, masked_scores = workspace
    _pool_candidates_dspark[(n,)](
        logits,
        e_score_correction_bias,
        flags,
        raw_scores,
        roles,
        logits.stride(0),
        logits.stride(1),
        NUM_EXPERTS=num_experts,
        ROUTE_TOP_K=route_top_k,
        SUFFIX_TOP_K=suffix_pool_top_k,
        multibuffer=False,
    )
    _batch_union_apply_dspark[(triton.cdiv(num_experts, BLOCK_EXPERTS),)](
        raw_scores,
        e_score_correction_bias,
        flags,
        roles,
        masked_scores,
        n,
        triton.next_power_of_2(n),
        NUM_EXPERTS=num_experts,
        BLOCK_EXPERTS=BLOCK_EXPERTS,
        multibuffer=False,
    )
    _, ids = torch.topk(masked_scores, route_top_k, dim=-1)
    weights = raw_scores.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * routed_scaling_factor
    return weights, ids.to(torch.int32)
