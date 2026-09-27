# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Fused EPLB gating, replica mapping and load recording.

One Triton kernel replaces the three-launch route ``moe_gating_top_k``
-> ``ascend_eplb_map_to_physical`` -> ``ascend_eplb_record_expert_tokens``:

  score = sigmoid(x) or softmax(x); key = score + bias
  group score = sum of the top-2 keys inside each group
  select the top ``k_group`` groups, then the top ``k`` experts among them
  weight = score / (sum(score) + eps) * routed_scaling_factor

matching the CANN ``moe_gating_top_k`` semantics (the generalized
variant) bit for bit, including left tie-breaks. The DeepSeek V4 hash
variant (``hash_gating_top_k_map_and_record``) takes the expert ids
straight from the ``tid2eid`` lookup table and scores them with
``sqrt(softplus(x))``.

The winning physical experts are gathered through the graph-stable
replica routing table and their token counts are accumulated into the
local slice of ``expert_load_view``; tokens at or beyond
``num_valid_tokens`` are padding and never counted.

The serial selection chain only beats the CANN stack below a few hundred
tokens, so callers route by batch bucket (see
``GATING_TOP_K_MAP_RECORD_MAX_TOKENS``) and keep the unfused path above it.
"""

import torch
from vllm.triton_utils import tl, triton

_BIG: tl.constexpr = 1 << 30
_NEG_INF: tl.constexpr = float("-inf")

# Batch buckets above this token count keep the unfused CANN route; the
# crossover was measured on Ascend 910C (T=512: parity, T<=256: up to 1.7x).
GATING_TOP_K_MAP_RECORD_MAX_TOKENS = 512


@triton.jit
def gating_top_k_map_and_record_kernel(
    x_ptr,  # [num_tokens, num_experts] router logits (fp32/bf16, cast in-kernel)
    bias_ptr,  # [num_experts] e_score_correction_bias
    table_ptr,  # [TABLE_ROWS, num_experts] int32 replica routing table
    record_enabled_ptr,  # 0-D device int, non-zero enables recording
    num_valid_ptr,  # 0-D device int, tokens beyond this are padding
    load_ptr,  # [num_physical] int32 expert load view, atomically accumulated
    y_ptr,  # [num_tokens, K] fp32 topk weights
    ids_ptr,  # [num_tokens, K] int32 physical expert ids
    num_tokens,
    num_experts,
    local_expert_start,
    eps,
    scaling,
    num_physical,
    K: tl.constexpr,
    GROUP_COUNT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    K_GROUP: tl.constexpr,
    LOCAL_COUNT: tl.constexpr,
    LOCAL_COUNT_POW2: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    E_ALIGN: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
    NORM_TYPE: tl.constexpr,
    RENORM: tl.constexpr,
):
    tok0 = tl.program_id(0) * TOKENS_PER_PROGRAM
    trows = tok0 + tl.arange(0, TOKENS_PER_PROGRAM)  # (TS,)
    tmask = trows < num_tokens
    offs = tl.arange(0, E_ALIGN)
    emask = offs < num_experts
    lmask = tmask[:, None] & emask[None, :]

    x = tl.load(x_ptr + trows[:, None] * num_experts + offs[None, :], mask=lmask, other=0.0).to(tl.float32)
    if NORM_TYPE == 0:
        # Full softmax over all experts; masked lanes contribute exp(-inf) = 0.
        xsafe = tl.where(lmask, x, _NEG_INF)
        xexp = tl.exp(xsafe - tl.max(xsafe, axis=1)[:, None])
        score = xexp / tl.sum(xexp, axis=1)[:, None]
    else:
        score = tl.sigmoid(x)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs, mask=emask, other=0.0).to(tl.float32)
        key = tl.where(lmask, score + bias[None, :], _NEG_INF)
    else:
        key = tl.where(lmask, score, _NEG_INF)

    g_idx = offs // GROUP_SIZE
    garange = tl.arange(0, GROUP_COUNT)
    karange = tl.arange(0, K)
    rec_on = tl.load(record_enabled_ptr) != 0
    tok_valid = trows < tl.load(num_valid_ptr)
    active = rec_on & tok_valid & tmask

    # Group score: sum of the top-2 keys per group. The group-max element is
    # removed by its within-group index (lowest on ties, like the CANN
    # kernel) so duplicated values are handled. NOTE: a 3D one-shot
    # reduction and iterative tl.argmax both miscompile on triton-ascend
    # 3.2.x, so this stays an explicit per-group loop of 2D reductions.
    gs = tl.zeros((TOKENS_PER_PROGRAM, GROUP_COUNT), dtype=tl.float32)
    for g in tl.static_range(GROUP_COUNT):
        gm = g_idx[None, :] == g
        k1 = tl.max(tl.where(gm, key, _NEG_INF), axis=1)
        i1 = tl.min(tl.where(gm & (key == k1[:, None]), offs[None, :], _BIG), axis=1)
        k2 = tl.max(tl.where(gm & (offs[None, :] != i1[:, None]), key, _NEG_INF), axis=1)
        gs = tl.where(garange[None, :] == g, k1[:, None] + k2[:, None], gs)

    # Select the top K_GROUP groups per token; record the winning expert mask.
    sel = tl.zeros((TOKENS_PER_PROGRAM, E_ALIGN), dtype=tl.int32)
    for _ in tl.static_range(K_GROUP):
        gv = tl.max(gs, axis=1)
        gi = tl.min(tl.where(gs == gv[:, None], garange[None, :], GROUP_COUNT), axis=1)
        sel = sel | (g_idx[None, :] == gi[:, None]).to(tl.int32)
        gs = tl.where(garange[None, :] == gi[:, None], _NEG_INF, gs)

    # Top-K experts per token among the selected groups, then map + record.
    # NOTE: collecting the ids first and vectorizing the map/record epilogue
    # miscompiles on this backend (loop-carried int32 accumulation); the
    # per-step loop below is the validated form.
    cand = tl.where((sel > 0) & lmask, key, _NEG_INF)
    karange = tl.arange(0, K)
    lc = tl.arange(0, LOCAL_COUNT_POW2)
    hits = tl.zeros((TOKENS_PER_PROGRAM, LOCAL_COUNT_POW2), dtype=tl.int32)
    sc = tl.zeros((TOKENS_PER_PROGRAM, K), dtype=tl.float32)

    for j in tl.static_range(K):
        v = tl.max(cand, axis=1)
        e = tl.min(tl.where(cand == v[:, None], offs[None, :], _BIG), axis=1)
        s = tl.sum(tl.where(offs[None, :] == e[:, None], score, 0.0), axis=1)
        phys = tl.load(table_ptr + (trows % TABLE_ROWS) * num_experts + e, mask=tmask, other=-1)
        tl.store(ids_ptr + trows * K + j, phys, mask=tmask)
        local = phys - local_expert_start
        hits += ((lc[None, :] == local[:, None]) & active[:, None]).to(tl.int32)
        cand = tl.where(offs[None, :] == e[:, None], _NEG_INF, cand)
        sc = tl.where(karange[None, :] == j, s[:, None], sc)

    ysum = tl.sum(sc, axis=1)
    if RENORM:
        # Sigmoid always re-normalizes over the selected subset; softmax only
        # when renorm == 1 (the CANN generalized variant's needRenorm rule).
        sc = sc / (ysum[:, None] + eps)
    elif NORM_TYPE == 0:
        pass  # full softmax values are already globally normalized
    tl.store(
        y_ptr + trows[:, None] * K + karange[None, :],
        sc * scaling,
        mask=tmask[:, None],
    )

    # One vector atomic per program instead of K scalar atomics per token.
    lc_ok = (lc < LOCAL_COUNT) & (lc < num_physical - local_expert_start)
    row_lc = tl.broadcast_to(lc[None, :], (TOKENS_PER_PROGRAM, LOCAL_COUNT_POW2))
    tl.atomic_add(
        load_ptr + local_expert_start + row_lc,
        hits,
        mask=tl.broadcast_to(lc_ok[None, :], (TOKENS_PER_PROGRAM, LOCAL_COUNT_POW2)) & active[:, None],
    )


def _launch_gating_top_k_map_and_record(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    routing_table: torch.Tensor,
    record_enabled: torch.Tensor,
    num_valid_tokens: torch.Tensor,
    expert_load_view: torch.Tensor,
    local_expert_start: int,
    local_expert_count: int,
    k: int,
    k_group: int,
    group_count: int,
    routed_scaling_factor: float,
    eps: float = 1e-20,
    norm_type: int = 1,
    renorm: bool = True,
    num_warps: int = 4,
    tokens_per_program: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (weights [T, K] fp32, physical ids [T, K] int32)."""
    num_tokens, num_experts = logits.shape
    group_size = num_experts // group_count
    if tokens_per_program is None:
        # Keep enough programs in flight for occupancy while amortizing the
        # serial per-program reduction chain. The backend only tolerates
        # power-of-two tile heights (TS=5 aborted in parseSelect) and TS=1
        # or TS>16 hit other shape quirks.
        ts = max(2, min(4, num_tokens // 96))
        tokens_per_program = 2 ** (ts.bit_length() - 1)
    weights = torch.empty((num_tokens, k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((num_tokens, k), dtype=torch.int32, device=logits.device)
    gating_top_k_map_and_record_kernel[(triton.cdiv(num_tokens, tokens_per_program),)](
        logits,
        bias if bias is not None else logits,  # dummy pointer when unused
        routing_table,
        record_enabled,
        num_valid_tokens,
        expert_load_view,
        weights,
        ids,
        num_tokens,
        num_experts,
        local_expert_start,
        eps,
        routed_scaling_factor,
        expert_load_view.numel(),
        K=k,
        GROUP_COUNT=group_count,
        GROUP_SIZE=group_size,
        K_GROUP=k_group,
        LOCAL_COUNT=local_expert_count,
        LOCAL_COUNT_POW2=triton.next_power_of_2(max(local_expert_count, 1)),
        TABLE_ROWS=routing_table.shape[0],
        HAS_BIAS=bias is not None,
        E_ALIGN=triton.next_power_of_2(num_experts),
        TOKENS_PER_PROGRAM=tokens_per_program,
        NORM_TYPE=norm_type,
        RENORM=renorm,
        num_warps=num_warps,
    )
    return weights, ids


def gating_top_k_map_and_record(
    router_logits: torch.Tensor,
    bias: torch.Tensor | None,
    expert_replica_routing_table: torch.Tensor,
    record_enabled: torch.Tensor,
    num_valid_tokens: torch.Tensor,
    expert_load_view: torch.Tensor,
    local_expert_start: int,
    local_expert_count: int,
    k: int,
    k_group: int,
    group_count: int,
    routed_scaling_factor: float,
    norm_type: int = 1,
    renorm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton gating + replica mapping + load recording in a single launch.

    ``router_logits`` may contain padding rows; ``num_valid_tokens`` (a 0-D
    device tensor) splits real rows from padding so the load recording stays
    clean. Returns ``(topk_weights fp32, topk_ids int32 physical experts)``;
    callers must skip their own mapping and recording passes when they take
    this path. All per-step mutable inputs must be device tensors with
    stable addresses (ACL-graph safe).
    """
    if num_valid_tokens.device != router_logits.device or num_valid_tokens.dim() != 0:
        raise ValueError("num_valid_tokens must be a 0-D device tensor")
    return _launch_gating_top_k_map_and_record(
        router_logits,
        bias,
        expert_replica_routing_table,
        record_enabled,
        num_valid_tokens,
        expert_load_view,
        local_expert_start,
        local_expert_count,
        k,
        k_group,
        group_count,
        routed_scaling_factor,
        norm_type=norm_type,
        renorm=renorm,
    )


@triton.jit
def hash_gating_top_k_map_and_record_kernel(
    x_ptr,  # [num_tokens, num_experts] logits providing the weights
    table_ptr,  # [TABLE_ROWS, num_experts] int32 replica routing table
    input_ids_ptr,  # [num_tokens] token ids (padding already sanitized)
    tid2eid_ptr,  # [num_ids, K] token-id -> expert-id lookup table
    record_enabled_ptr,  # 0-D device int
    num_valid_ptr,  # 0-D device int
    load_ptr,  # [num_physical] int32
    y_ptr,  # [num_tokens, K] fp32
    ids_ptr,  # [num_tokens, K] int32
    num_tokens,
    num_experts,
    local_expert_start,
    eps,
    scaling,
    num_physical,
    K: tl.constexpr,
    LOCAL_COUNT: tl.constexpr,
    LOCAL_COUNT_POW2: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    E_ALIGN: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
):
    tok0 = tl.program_id(0) * TOKENS_PER_PROGRAM
    trows = tok0 + tl.arange(0, TOKENS_PER_PROGRAM)
    tmask = trows < num_tokens
    offs = tl.arange(0, E_ALIGN)
    emask = offs < num_experts
    karange = tl.arange(0, K)
    rec_on = tl.load(record_enabled_ptr) != 0
    tok_valid = trows < tl.load(num_valid_ptr)
    active = rec_on & tok_valid & tmask

    # DeepSeek V4 hash route: the expert ids come straight from the lookup
    # table; the logits only provide the weights, scored as
    # sqrt(softplus(x)) (pre-bias) and re-normalized over the selected
    # subset exactly like the CANN regbase variant. The lookup, mapping and
    # recording run as K single-column steps: 2D gather loads with a
    # row-broadcast mask are unreliable on this backend.
    x = tl.load(
        x_ptr + trows[:, None] * num_experts + offs[None, :],
        mask=tmask[:, None] & emask[None, :],
        other=0.0,
    ).to(tl.float32)
    score = tl.sqrt(tl.log(1.0 + tl.exp(x)))

    key = tl.load(input_ids_ptr + trows, mask=tmask, other=0)
    sc = tl.zeros((TOKENS_PER_PROGRAM, K), dtype=tl.float32)
    for j in tl.static_range(K):
        e_j = tl.load(tid2eid_ptr + key * K + j, mask=tmask, other=0).to(tl.int32)
        s_j = tl.sum(tl.where(offs[None, :] == e_j[:, None], score, 0.0), axis=1)
        phys = tl.load(table_ptr + (trows % TABLE_ROWS) * num_experts + e_j, mask=tmask, other=-1)
        tl.store(ids_ptr + trows * K + j, phys, mask=tmask)
        raw_local = phys - local_expert_start
        hit = active & (raw_local >= 0) & (raw_local < LOCAL_COUNT)
        # Clamp so masked lanes (phys = -1 on padding rows) keep the address
        # in bounds even though the atomic itself is masked off.
        local = tl.maximum(raw_local, 0)
        tl.atomic_add(load_ptr + local_expert_start + local, 1, mask=hit)
        sc = tl.where(karange[None, :] == j, s_j[:, None], sc)

    ysum = tl.sum(sc, axis=1)
    tl.store(
        y_ptr + trows[:, None] * K + karange[None, :],
        sc / (ysum[:, None] + eps) * scaling,
        mask=tmask[:, None],
    )


def hash_gating_top_k_map_and_record(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    routing_table: torch.Tensor,
    record_enabled: torch.Tensor,
    num_valid_tokens: torch.Tensor,
    expert_load_view: torch.Tensor,
    local_expert_start: int,
    local_expert_count: int,
    k: int,
    routed_scaling_factor: float,
    eps: float = 1e-20,
    num_warps: int = 4,
    tokens_per_program: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek V4 hash route: ids from tid2eid, weights from sigmoid(logits)."""
    num_tokens, num_experts = router_logits.shape
    if tokens_per_program is None:
        ts = max(2, min(4, num_tokens // 96))
        tokens_per_program = 2 ** (ts.bit_length() - 1)
    weights = torch.empty((num_tokens, k), dtype=torch.float32, device=router_logits.device)
    ids = torch.empty((num_tokens, k), dtype=torch.int32, device=router_logits.device)
    hash_gating_top_k_map_and_record_kernel[(triton.cdiv(num_tokens, tokens_per_program),)](
        router_logits,
        routing_table,
        input_ids,
        tid2eid,
        record_enabled,
        num_valid_tokens,
        expert_load_view,
        weights,
        ids,
        num_tokens,
        num_experts,
        local_expert_start,
        eps,
        routed_scaling_factor,
        expert_load_view.numel(),
        K=k,
        LOCAL_COUNT=local_expert_count,
        LOCAL_COUNT_POW2=triton.next_power_of_2(max(local_expert_count, 1)),
        TABLE_ROWS=routing_table.shape[0],
        E_ALIGN=triton.next_power_of_2(num_experts),
        TOKENS_PER_PROGRAM=tokens_per_program,
        num_warps=num_warps,
    )
    return weights, ids
