"""Shape-derived UB screening for the fixed M2/off launch.

The estimate keeps the existing 1.05 screening factor and max(pair, odd-row) rule.
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
    """Whether the pair path is inside the screened head-tile domain.

    This is a structural applicability boundary, not a case/SoC/toolchain
    whitelist or an inferred UB byte count. Positive boundary evidence and a
    revised liveness model are required before permitting N_BND > 0.
    """
    if type(num_q_heads) is not int or num_q_heads <= 0:
        raise ValueError("num_q_heads must be a positive integer")
    return num_q_heads % _bhq(num_q_heads) == 0


def _tile_body_bytes(
    *,
    rows: int,
    q_heads: int,
    kv_heads: int,
    head_size: int,
    has_gate: bool,
    rope_dim: int,
    dtype_bytes: int,
) -> int:
    """Estimate the tile body without common weights, using the original screening assumptions.

    Callers select full single-row heads or the fixed pair's Q-head tile.
    This helper does not truncate heads or solve for arbitrary row counts.
    """
    w = dtype_bytes
    fp = FP32_BYTES
    half = rope_dim // 2
    slot = head_size * (2 if has_gate else 1)
    nq, nk = rows * q_heads, rows * kv_heads

    coeff_bytes = _align32(6 * rows * half * w) + 2 * _align32(rows * half * fp)  # two synthesis halves
    q_gate_bytes = (
        _align32(nq * slot * w)  # landing
        + _align32(nq * slot * fp)  # FP32 view
        + 2 * _align32(nq * half * fp)  # cos/sin
        + _align32(nq * head_size * fp)  # Q chain
        + 2 * _align32(nq * fp)  # variance + rstd
        + _align32(nq * head_size * w)  # Q store
    )
    if has_gate:
        q_gate_bytes += _align32(nq * head_size * w)  # Gate store remains separately aligned
    kv_bytes = (
        2 * _align32(nk * head_size * fp)  # K input + K chain
        + 2 * _align32(nk * half * fp)  # cos/sin
        + 2 * _align32(nk * fp)  # variance + rstd
        + 2 * _align32(nk * head_size * w)  # K store + V pass-through
    )
    return coeff_bytes + q_gate_bytes + kv_bytes


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
    """Screen the maximum estimated working set of the disjoint pair/odd-row loops.

    Without pairs, only the single-row estimate is screened before the workload gate selects
    M1. This does not introduce a separate M2 tail-only kernel instance.
    The model assumes reuse of Q/K chain buffers; mask/arange overhead is
    absorbed in ALPHA. The estimate-vs-capacity gap is not free UB headroom.
    """
    w = DTYPE_WIDTH[dtype]
    weights_bytes = 2 * _align32(head_size * w)
    if has_bias:
        weights_bytes += 2 * _align32(head_size * w)
    row_body = _tile_body_bytes(
        rows=1,
        q_heads=num_q_heads,
        kv_heads=num_kv_heads,
        head_size=head_size,
        has_gate=has_gate,
        rope_dim=rope_dim,
        dtype_bytes=w,
    )
    body = row_body
    if has_pair_work:
        pair_body = _tile_body_bytes(
            rows=2,
            q_heads=_bhq(num_q_heads),
            kv_heads=num_kv_heads,
            head_size=head_size,
            has_gate=has_gate,
            rope_dim=rope_dim,
            dtype_bytes=w,
        )
        body = max(row_body, pair_body)
    return math.ceil((weights_bytes + body) * ALPHA)
