"""Existing TopK/zero/fill/apply/native route structure with anchors and roles."""

import torch
import triton
import triton.language as tl

FIXED_GRAPH_ROWS = (8, 16, 24, 32, 48, 64)


@triton.jit
def _anchor_fill(
    Top,
    Roles,
    Mask,
    N: tl.constexpr,
    TOP_K: tl.constexpr,
    SUFFIX_TOP_K: tl.constexpr,
):
    p = tl.program_id(0)
    for row in tl.range(p, N, tl.num_programs(0)):
        role = tl.load(Roles + row)
        limit = tl.where(role == 1, TOP_K, tl.where(role == 2, SUFFIX_TOP_K, 0))
        for rank in range(TOP_K):
            eid = tl.load(Top + row * TOP_K + rank, rank < limit, 0)
            tl.store(Mask + eid, True, rank < limit)


@triton.jit
def _anchor_apply(
    X,
    Roles,
    Mask,
    N: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    NEG: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
):
    p = tl.program_id(0)
    e = tl.arange(0, NUM_EXPERTS)
    allowed = tl.load(Mask + e)
    for row in tl.range(p, N, tl.num_programs(0)):
        role = tl.load(Roles + row)
        if role == 2:
            x = tl.load(X + row * S0 + e * S1)
            tl.store(X + row * S0 + e * S1, tl.where(allowed, x, NEG))


@triton.jit(do_not_specialize=["N"])
def _anchor_fill_dynamic(
    Top,
    Roles,
    Mask,
    N,
    TOP_K: tl.constexpr,
    SUFFIX_TOP_K: tl.constexpr,
):
    """Variable mixed-prefill rows must not create a kernel per row count."""
    p = tl.program_id(0)
    for row in tl.range(p, N, tl.num_programs(0)):
        role = tl.load(Roles + row)
        limit = tl.where(role == 1, TOP_K, tl.where(role == 2, SUFFIX_TOP_K, 0))
        for rank in range(TOP_K):
            eid = tl.load(Top + row * TOP_K + rank, rank < limit, 0)
            tl.store(Mask + eid, True, rank < limit)


@triton.jit(do_not_specialize=["N"])
def _anchor_apply_dynamic(
    X,
    Roles,
    Mask,
    N,
    S0: tl.constexpr,
    S1: tl.constexpr,
    NEG: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
):
    p = tl.program_id(0)
    e = tl.arange(0, NUM_EXPERTS)
    allowed = tl.load(Mask + e) != 0
    for row in tl.range(p, N, tl.num_programs(0)):
        role = tl.load(Roles + row)
        if role == 2:
            x = tl.load(X + row * S0 + e * S1)
            tl.store(X + row * S0 + e * S1, tl.where(allowed, x, NEG))


def _apply_anchor_reference_torch(scores, roles, top_k, suffix_pool_top_k):
    ids = torch.topk(scores, top_k, dim=-1).indices
    limits = torch.where(
        roles == 1,
        top_k,
        torch.where(roles == 2, suffix_pool_top_k, 0),
    )
    ranks = torch.arange(top_k, device=scores.device).unsqueeze(0)
    active = ranks < limits.unsqueeze(1)
    flags = torch.zeros_like(scores, dtype=torch.bool)
    flags.scatter_(1, ids, active)
    pool = flags.any(dim=0)
    return scores.masked_fill(
        (roles.unsqueeze(1) == 2) & ~pool.unsqueeze(0),
        torch.finfo(scores.dtype).min,
    )


def apply_anchor_reference(logits, roles, mask, top_k=8, suffix_pool_top_k=2):
    if logits.device.type == "cpu":
        return _apply_anchor_reference_torch(logits, roles, top_k, suffix_pool_top_k)

    # the fused kernel does not consume non-unit strides correctly
    # (validated on A3). Production router logits are always contiguous;
    # guard here so strided inputs are copied once instead of mis-read.
    if not logits.is_contiguous():
        logits = logits.contiguous()
    ids = torch.topk(logits, top_k, dim=-1).indices
    mask.zero_()
    n = logits.shape[0]
    fill = _anchor_fill if n in FIXED_GRAPH_ROWS else _anchor_fill_dynamic
    apply = _anchor_apply if n in FIXED_GRAPH_ROWS else _anchor_apply_dynamic
    fill[(min(n, 40),)](
        ids,
        roles,
        mask,
        n,
        TOP_K=top_k,
        SUFFIX_TOP_K=suffix_pool_top_k,
        multibuffer=False,
    )
    apply[(min(n, 40),)](
        logits,
        roles,
        mask,
        n,
        logits.stride(0),
        logits.stride(1),
        float(torch.finfo(logits.dtype).min),
        NUM_EXPERTS=logits.shape[1],
        multibuffer=False,
    )
    return logits
