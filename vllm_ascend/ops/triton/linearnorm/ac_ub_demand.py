"""Shape-derived UB screening for the fixed M2/off launch.

The estimate keeps the calibrated 1.05 factor and max(pair, odd-row) rule.
It is not a proven upper bound. Boundary-head tiles remain excluded by the
selector after the archived Hq16/D256 allocator overflow.
"""

from __future__ import annotations

import math

DTYPE_WIDTH = {"bfloat16": 2, "float16": 2}
FP32_BYTES = 4
ALIGN_BYTES = 32
ALPHA = 1.05


def _align32(bytes_: int) -> int:
    return ((bytes_ + ALIGN_BYTES - 1) // ALIGN_BYTES) * ALIGN_BYTES


def _bhq(num_q_heads: int) -> int:
    return num_q_heads if num_q_heads < 12 else 12


def pair_boundary_screen_supported(num_q_heads: int) -> bool:
    """Whether the pair path is inside the calibrated head-tile domain.

    This is a structural applicability boundary, not a case/SoC/toolchain
    whitelist or an inferred UB byte count. Positive boundary evidence and a
    revised liveness model are required before permitting N_BND > 0.
    """
    if type(num_q_heads) is not int or num_q_heads <= 0:
        raise ValueError("num_q_heads must be a positive integer")
    return num_q_heads % _bhq(num_q_heads) == 0


def _itemize_m1_row(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    has_gate: bool,
    rope_dim: int,
    dtype: str,
    has_bias: bool,
) -> int:
    """Per-iteration liveness peak of the BLOCK_M==1 dataflow (≡ odd-tail body)."""
    w = DTYPE_WIDTH[dtype]
    fp = FP32_BYTES
    half = rope_dim // 2
    slot = head_size * (2 if has_gate else 1)

    items = [
        # weights: q + k (+ q/k bias when has_bias), one physical buffer each.
        _align32(head_size * w),
        _align32(head_size * w),
    ]
    if has_bias:
        items.extend((_align32(head_size * w), _align32(head_size * w)))
    # landing / views.
    items.append(_align32(num_q_heads * slot * w))  # q_gate_landing
    items.append(_align32(num_q_heads * slot * fp))  # q_gate_fp32_view
    items.append(_align32(num_kv_heads * head_size * fp))  # in_k_fp32
    items.append(_align32(num_kv_heads * head_size * w))  # v pass-through
    # cos/sin (six bf16 loads; then two synthesis buffers; then per-head sets).
    items.append(_align32(6 * half * w))
    items.append(_align32(half * fp))  # cos_half
    items.append(_align32(half * fp))  # sin_half
    items.append(_align32(num_q_heads * half * fp))  # cos_q
    items.append(_align32(num_q_heads * half * fp))  # sin_q
    items.append(_align32(num_kv_heads * half * fp))  # cos_k
    items.append(_align32(num_kv_heads * half * fp))  # sin_k
    # q chain (one reused (Hq, D) fp32 buffer + two tiny reductions).
    items.append(_align32(num_q_heads * head_size * fp))
    items.append(_align32(num_q_heads * fp))  # q_variances
    items.append(_align32(num_q_heads * fp))  # q_reciprocal_std
    # k chain (one reused (Hkv, D) fp32 buffer + two tiny reductions).
    items.append(_align32(num_kv_heads * head_size * fp))
    items.append(_align32(num_kv_heads * fp))  # k_variances
    items.append(_align32(num_kv_heads * fp))  # k_reciprocal_std
    # store staging (bf16; v reuses its landing block).
    items.append(_align32(num_q_heads * head_size * w))  # q store
    items.append(_align32(num_kv_heads * head_size * w))  # k store
    if has_gate:
        items.append(_align32(num_q_heads * head_size * w))  # gate store
    return sum(items)


def _itemize_pair_tile(
    *,
    block_m: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    has_gate: bool,
    rope_dim: int,
    dtype: str,
    has_bias: bool,
) -> int:
    """Screen the fixed two-row, head-bounded pair body."""
    w = DTYPE_WIDTH[dtype]
    fp = FP32_BYTES
    half = rope_dim // 2
    bhq = _bhq(num_q_heads)
    slot = head_size * (2 if has_gate else 1)
    nq = block_m * bhq
    nk = block_m * num_kv_heads

    items = [
        _align32(head_size * w),
        _align32(head_size * w),
    ]
    if has_bias:
        items.extend((_align32(head_size * w), _align32(head_size * w)))
    # cos/sin.
    items.append(_align32(6 * block_m * half * w))
    items.append(_align32(block_m * half * fp))
    items.append(_align32(block_m * half * fp))
    # Q/Gate tile body (charged at PEAK for one BHQ tile, incl. boundary block).
    items.append(_align32(block_m * bhq * slot * w))  # tile_landing
    items.append(_align32(nq * slot * fp))  # tile_fp32_view
    items.append(_align32(nq * half * fp))  # cos_q
    items.append(_align32(nq * half * fp))  # sin_q
    items.append(_align32(nq * head_size * fp))  # q chain
    items.append(_align32(nq * fp))  # q_variances
    items.append(_align32(nq * fp))  # q_reciprocal_std
    items.append(_align32(block_m * bhq * head_size * w))  # q store
    if has_gate:
        items.append(_align32(block_m * bhq * head_size * w))  # gate store
    # K/V.
    items.append(_align32(nk * head_size * fp))  # in_k_fp32
    items.append(_align32(nk * half * fp))  # cos_k
    items.append(_align32(nk * half * fp))  # sin_k
    items.append(_align32(nk * head_size * fp))  # k chain
    items.append(_align32(nk * fp))  # k_variances
    items.append(_align32(nk * fp))  # k_reciprocal_std
    items.append(_align32(block_m * num_kv_heads * head_size * w))  # k store
    items.append(_align32(block_m * num_kv_heads * head_size * w))  # v pass-through
    return sum(items)


def estimate_demand(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    has_gate: bool,
    rope_dim: int,
    dtype: str,
    has_bias: bool,
    has_pair_work: bool,
) -> int:
    """Estimate the peak, not the sum of the disjoint pair/odd-row loops.

    Without pairs, only the single-row peak is screened before G3 selects
    M1. This does not introduce a separate M2 tail-only kernel instance.
    The model assumes reuse of Q/K chain buffers; mask/arange overhead is
    absorbed in ALPHA. The estimate-vs-capacity gap is not free UB headroom.
    """
    peak = _itemize_m1_row(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        has_gate=has_gate,
        rope_dim=rope_dim,
        dtype=dtype,
        has_bias=has_bias,
    )
    if has_pair_work:
        pair_peak = _itemize_pair_tile(
            block_m=2,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            has_gate=has_gate,
            rope_dim=rope_dim,
            dtype=dtype,
            has_bias=has_bias,
        )
        peak = max(peak, pair_peak)
    return math.ceil(peak * ALPHA)
