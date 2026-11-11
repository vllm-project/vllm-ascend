"""Fused candidate-pool construction for Qwen DFlash and DeepSeek DSpark.

The implementation keeps the original two-kernel contract: one kernel builds
per-row candidate flags, a second kernel reduces the flags into a batch union
and masks suffix rows. This is not a Top-M implementation.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as ascend

from vllm_ascend.device.device_op import DeviceOperator

USES_TOPM = False


@triton.jit
def _pool_candidates_bf16(
    X,
    Flags,
    Roles,
    S0: tl.constexpr,
    S1: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    SUFFIX_TOP_K: tl.constexpr,
):
    """Published BF16 Top8/Top2 path with a stable lower-ID tie break."""
    with ascend.scope(core_mode="vector"):
        row = tl.program_id(0)
        e = tl.arange(0, NUM_EXPERTS)
        x = tl.load(X + row * S0 + e * S1)
        bits = (x.to(tl.float32).to(tl.int32, bitcast=True) >> 16) & 65535
        bits = tl.where(x == 0, 0, bits)
        ordered = tl.where(bits >= 32768, 65535 - bits, bits + 32768)
        # NUM_EXPERTS <= 256 keeps the largest integer key exactly
        # representable in float32 (65535 * 256 + 255 == 2**24 - 1).
        keys = (ordered * NUM_EXPERTS + NUM_EXPERTS - 1 - e).to(tl.float32)
        sorted_keys = ascend.sort(keys, descending=True)
        role = tl.load(Roles + row)
        rank = tl.where(role == 1, 7, SUFFIX_TOP_K - 1)
        threshold = tl.sum(tl.where(e == rank, sorted_keys, 0), 0)
        tl.store(
            Flags + row * NUM_EXPERTS + e,
            (keys >= threshold) & (role != 0),
        )


@triton.jit
def _pool_candidates_dspark(
    X,
    Bias,
    Flags,
    RawScores,
    Roles,
    S0: tl.constexpr,
    S1: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    ROUTE_TOP_K: tl.constexpr,
    SUFFIX_TOP_K: tl.constexpr,
):
    """K1: sqrt-softplus + bias + stable Top6/Top2 candidate flags."""
    with ascend.scope(core_mode="vector"):
        row = tl.program_id(0)
        e = tl.arange(0, NUM_EXPERTS)
        x = tl.load(X + row * S0 + e * S1).to(tl.float32)
        # Match moe_gating_top_k_hash's AscendC operation order exactly:
        # Exp -> Adds(1) -> Ln -> Sqrt. RawScores excludes correction bias.
        softplus = tl.log(tl.exp(x) + 1.0)
        raw = tl.sqrt(softplus)
        score = raw + tl.load(Bias + e).to(tl.float32)
        tl.store(RawScores + row * NUM_EXPERTS + e, raw)

        sorted_scores = ascend.sort(score, descending=True)
        role = tl.load(Roles + row)
        limit = tl.where(
            role == 1,
            ROUTE_TOP_K,
            tl.where(role == 2, SUFFIX_TOP_K, 0),
        )
        rank = tl.maximum(limit - 1, 0)
        threshold = tl.sum(tl.where(e == rank, sorted_scores, 0.0), 0)
        greater = score > threshold
        equal = score == threshold
        num_greater = tl.sum(greater.to(tl.int32), 0)
        ties_needed = limit - num_greater
        # Expert IDs are the lane order. Prefix-summing equal scores selects
        # lower IDs first and guarantees exactly `limit` flags on ties.
        tie_rank = tl.cumsum(equal.to(tl.int32), axis=0)
        selected = greater | (equal & (tie_rank <= ties_needed))
        tl.store(
            Flags + row * NUM_EXPERTS + e,
            selected & (role != 0),
        )


@triton.jit
def _batch_union_apply_bf16(
    X,
    Flags,
    Roles,
    Y,
    N: tl.constexpr,
    NR: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
):
    with ascend.scope(core_mode="vector"):
        r = tl.arange(0, NR)
        e = tl.arange(0, NUM_EXPERTS)
        flags = tl.load(
            Flags + r[:, None] * NUM_EXPERTS + e[None, :],
            r[:, None] < N,
            0,
        )
        pool = tl.max(flags, 0) != 0
        roles = tl.load(Roles + r, r < N, 0)
        raw = tl.load(X + r[:, None] * S0 + e[None, :] * S1, r[:, None] < N, 0)
        masked = tl.where(
            (roles[:, None] == 2) & ~pool[None, :],
            -3.3895313892515355e38,
            raw.to(tl.float32),
        )
        tl.store(
            Y + r[:, None] * NUM_EXPERTS + e[None, :],
            masked,
            r[:, None] < N,
        )


@triton.jit
def _batch_union_apply_dspark(
    RawScores,
    Bias,
    Flags,
    Roles,
    MaskedScores,
    N: tl.constexpr,
    NR: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """K2: reduce the full padded batch union and mask suffix scores."""
    with ascend.scope(core_mode="vector"):
        e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        r = tl.arange(0, NR)
        expert_mask = e < NUM_EXPERTS
        row_mask = r < N
        flags = tl.load(
            Flags + r[:, None] * NUM_EXPERTS + e[None, :],
            row_mask[:, None] & expert_mask[None, :],
            0,
        )
        pool = tl.max(flags, 0) != 0
        roles = tl.load(Roles + r, row_mask, 0)
        raw = tl.load(
            RawScores + r[:, None] * NUM_EXPERTS + e[None, :],
            row_mask[:, None] & expert_mask[None, :],
            0.0,
        )
        bias = tl.load(Bias + e, expert_mask, 0.0)
        selection = raw + bias[None, :]
        masked = tl.where(
            (roles[:, None] == 2) & ~pool[None, :],
            -3.4028234663852886e38,
            selection,
        )
        tl.store(
            MaskedScores + r[:, None] * NUM_EXPERTS + e[None, :],
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
    # the fused kernel does not consume non-unit strides correctly
    # (validated on A3). Production router logits are always contiguous;
    # guard here so strided inputs are copied once instead of mis-read.
    if not logits.is_contiguous():
        logits = logits.contiguous()
    n, num_experts = logits.shape
    flags, masked = workspace or allocate_workspace(logits)
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
    if route_top_k != 6 or suffix_pool_top_k != 2:
        raise ValueError("DSpark fused pool requires route_top_k=6 and suffix_pool_top_k=2")
    if not renormalize:
        raise ValueError("DSpark fused pool requires renormalize=True")
    # the fused kernel does not consume non-unit strides correctly
    # (validated on A3). Production router logits are always contiguous;
    # guard here so strided inputs are copied once instead of mis-read.
    if not logits.is_contiguous():
        logits = logits.contiguous()
    n, num_experts = logits.shape
    flags, raw_scores, masked_scores = workspace or allocate_dspark_workspace(logits)
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
    block_e = 32
    _batch_union_apply_dspark[(triton.cdiv(num_experts, block_e),)](
        raw_scores,
        e_score_correction_bias,
        flags,
        roles,
        masked_scores,
        n,
        triton.next_power_of_2(n),
        NUM_EXPERTS=num_experts,
        BLOCK_E=block_e,
        multibuffer=False,
    )
    _, ids = torch.topk(masked_scores, route_top_k, dim=-1)
    weights = raw_scores.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * routed_scaling_factor
    return weights, ids.to(torch.int32)
