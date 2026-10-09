"""Host-only, stdlib-only capacity resolution + resource screening estimate.

This module serves the A+C envelope candidate's resource gate (G2).  It is a
*calibrated screening estimate*, NOT a proven demand upper bound: ``alpha``
absorbs un-itemized temporaries and planner behaviour, and the buffer-reuse
rules (A1/A2/A3) are assumptions the compiler allocator remains the authority
over.  ``estimated_margin`` is always just the difference estimate-vs-budget,
never a measured UB headroom.

Capacity sources are resolved by :func:`capacity_resolve`:

  * C1 -- ``get_device_properties().max_shared_mem``: **diagnostic only**.
    ``1`` is a known placeholder on the frozen image; even a 64 KiB..1 MiB
    integer from another backend does NOT establish that it means UB, so C1
    never participates in the envelope and only leaves an audit trail.
  * C2 -- the frozen-image backend's internal UB-budget field
    ``triton.backends.ascend.runtime.utils.ub_size_in_kbytes`` (KiB, chosen by
    the current Triton target), observed best-effort through a small local
    encapsulation in the wrapper and recorded **bound to the current compile
    target**.  It is an internal interface, not a stable public API: the value
    is unit-validated and target-bound, and a failure is an ``unknown``
    source, never a fabricated value.
  * There is **no** version-combo capacity whitelist (no curated table keyed
    by a torch/triton/torch-npu fingerprint): a toolchain version string is
    never a capacity source.  If C2 cannot be observed the capacity is
    ``unknown`` -- the caller must NOT pretend a capacity is known; the
    resource gate then conservatively fails closed to M1 (reason ``resource:
    capacity unknown``).  The only documented conservative capacity class is
    the 196,608 B observed on the Ascend 910B4 frozen image (allocator error
    string ``1,572,864 bits available``); it is an audit/calibration note with
    an applicability boundary, not a runtime lookup.

The demand estimate covers the three dataflow paths of the integrated kernel
(m1_row, pair_tile, tail_row) and reproduces, at ``BHQ == Hq``, the old
full-head liveness model byte-for-byte (design-verified). Five measured
anchors (all N_BND == 0) pin the calibration. E2c subsequently falsified
the in-envelope classification at Hq=16, N_BND=4: the allocator reported
205,120 B against a 196,608 B budget while this model estimated 166,119 B.
The numeric estimate remains useful for diagnosis, but is not an enablement
decision for pair-capable instances with N_BND > 0.

No Triton/torch/device import; stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeGuard

# ---------------------------------------------------------------------------
# Capacity resolution
# ---------------------------------------------------------------------------

# C1 is diagnostic-only: the known placeholder value on the frozen image and
# the unverifiable field semantics mean it never becomes an envelope source.
C1_PLACEHOLDER_VALUE = 1
_C1_STATE_DIAGNOSTIC_ONLY = "diagnostic_only"

_MIN_CAPACITY_BYTES = 64 * 1024
_MAX_CAPACITY_BYTES = 1024 * 1024

# Known capacity-class buckets (name -> bytes), used only as a diagnostic label
# on the resolved envelope (there is no evidence-gate bucket match anymore).
_CAPACITY_CLASS_BUCKETS: tuple[tuple[str, int], ...] = (
    ("ub192kib", 196608),
    ("ub256kib", 262144),
)


def capacity_class_name(capacity_bytes: int | None) -> str | None:
    """Map a resolved envelope byte count to a known capacity-class bucket."""
    if capacity_bytes is None:
        return None
    for name, size in _CAPACITY_CLASS_BUCKETS:
        if size == capacity_bytes:
            return name
    return None


def _parse_dist_version(spec: str) -> tuple[int, int] | None:
    """Parse one ``name==version`` segment to (major, minor); None if unusable."""
    if "==" not in spec:
        return None
    _, _, version = spec.partition("==")
    version = version.strip()
    head = version.split(".")[0:2]
    if len(head) != 2:
        return None
    major_minor: list[int] = []
    for part in head:
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            return None
        major_minor.append(int(digits))
    return (major_minor[0], major_minor[1])


def parse_toolchain_fingerprint(fingerprint: str) -> dict[str, tuple[int, int]] | None:
    """Parse the wrapper's ``torch==…;triton==…;torch-npu==…`` fingerprint.

    Returns a dist -> (major, minor) mapping, or None when any expected dist is
    missing / not parseable.  Diagnostic utility only: it feeds class-coverage
    rules in the policy and experiment records; it is NEVER a capacity source
    or a runtime selection gate (no version-combo capacity whitelist).
    """
    if not isinstance(fingerprint, str) or not fingerprint:
        return None
    parsed: dict[str, tuple[int, int]] = {}
    for segment in fingerprint.split(";"):
        name, _, _ = segment.partition("==")
        name = name.strip()
        version = _parse_dist_version(segment)
        if version is None:
            return None
        parsed[name] = version
    for required in ("torch", "triton", "torch-npu"):
        if required not in parsed:
            return None
    return parsed


def _valid_int_capacity(value: object) -> TypeGuard[int]:
    if type(value) is not int:
        return False
    return _MIN_CAPACITY_BYTES <= value <= _MAX_CAPACITY_BYTES


def capacity_resolve(
    *,
    c1: dict[str, Any],
    c2: dict[str, Any],
    toolchain_fingerprint: str,
) -> dict[str, Any]:
    """Resolve a conservative capacity envelope from observed candidates.

    ``c1`` (device properties) is diagnostic-only and never contributes.
    ``c2`` (backend runtime internal budget) contributes only when valid AND
    bound to the same compile target (``c2["target"] == toolchain_fingerprint``).
    There is no version-combo capacity whitelist: no fingerprint-keyed curated
    table participates.

    Returns:
        {
          "capacity_bytes": int | None,
          "capacity_state": "valid" | "unknown",
          "source": str | None,
          "sources_seen": [...],
          "disagreement": str | None,
        }
    """
    sources_seen: list[dict[str, Any]] = []
    # C1: diagnostic only -- record for the audit trail, never a source.
    c1_state = c1.get("state", "unknown")
    sources_seen.append(
        {
            "name": "get_device_properties.max_shared_mem",
            "value": c1.get("value"),
            "state": c1_state,
            "role": _C1_STATE_DIAGNOSTIC_ONLY,
            "reason": (
                c1.get("reason") or "max_shared_mem is not a verified UB source; placeholder==1 is diagnostic-only"
            ),
        }
    )
    valid_values: list[tuple[int, str]] = []

    # C2: backend runtime internal budget, target-bound.
    c2_state = c2.get("state", "unknown")
    c2_value = c2.get("value")
    c2_reason = c2.get("reason") or "not observed"
    c2_target_ok = c2.get("target") == toolchain_fingerprint
    if c2_state == "valid" and _valid_int_capacity(c2_value) and c2_target_ok:
        valid_values.append((int(c2_value), "backend_runtime_budget"))
        sources_seen.append(
            {
                "name": "backend_runtime_budget",
                "value": c2_value,
                "state": "valid",
                "role": "source",
                "target": toolchain_fingerprint,
                "reason": "observed internal runtime budget, target-bound",
            }
        )
    else:
        if c2_state == "valid" and not c2_target_ok:
            c2_reason = "observed value is not bound to the current compile target"
        elif c2_state == "valid" and not _valid_int_capacity(c2_value):
            c2_reason = "observed value is outside the accepted capacity range"
        sources_seen.append(
            {
                "name": "backend_runtime_budget",
                "value": c2_value,
                "state": c2_state if c2_state != "valid" else "invalid",
                "role": "source",
                "target": toolchain_fingerprint,
                "reason": c2_reason,
            }
        )

    # No version-combo capacity whitelist: there is no curated fingerprint
    # table (C3 removed in E1-R).  Capacity comes from C2 alone; if it is not
    # observed, the state is ``unknown`` and the caller fails closed.

    if not valid_values:
        return {
            "capacity_bytes": None,
            "capacity_state": "unknown",
            "capacity_class": None,
            "source": None,
            "sources_seen": sources_seen,
            "disagreement": None,
        }

    # Deterministic conservative resolution: min over the valid sources.
    min_value = min(value for value, _ in valid_values)
    sources = [name for value, name in valid_values]
    if len(sources) > 1:
        disagreement = f"source disagreement; taking conservative min over {sorted(sources)}"
    else:
        disagreement = None
    return {
        "capacity_bytes": min_value,
        "capacity_state": "valid",
        "capacity_class": capacity_class_name(min_value),
        "source": "+".join(sorted(sources)) if len(sources) > 1 else sources[0],
        "sources_seen": sources_seen,
        "disagreement": disagreement,
    }


# ---------------------------------------------------------------------------
# Resource screening estimate
# ---------------------------------------------------------------------------

# Screening domain (E1): bf16/fp16 input only (w == 2); fp32 intermediates are
# a fixed 4 B, independent of the input width.
DTYPE_WIDTH = {"bfloat16": 2, "float16": 2}
FP32_BYTES = 4
ALIGN_BYTES = 32

# Calibrated factors (design-verified against anchors A1..A5).  These are
# calibration points, NOT a proven upper-bound theorem.
ALPHA = 1.05
BETA_MULTIBUFFER_ON = 1.16
BETA_MULTIBUFFER_OFF = 1.0


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
    block_h_q: int | None = None,
) -> int:
    """Liveness peak of one tile-split pair body (BLOCK_M==2, PAIR_CAPABLE).

    ``block_h_q`` defaults to ``min(num_q_heads, 12)`` (the kernel constexpr);
    it can be pinned higher (e.g. 24) to reproduce the OLD full-head M2
    liveness footprint (design-verified collapse D_pair(BHQ=Hq) == old model),
    which is what anchors A1/A2 measure on the historical full-head source.
    """
    w = DTYPE_WIDTH[dtype]
    fp = FP32_BYTES
    half = rope_dim // 2
    bhq = _bhq(num_q_heads) if block_h_q is None else block_h_q
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


@dataclass(frozen=True)
class DemandEstimate:
    path: str
    peak_bytes: int
    alpha: float
    beta: float
    demand_estimate_bytes: int
    assumptions: tuple[str, ...]


def demand_estimate(
    *,
    variant: str,
    block_m: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    has_gate: bool,
    rope_dim: int,
    dtype: str,
    has_bias: bool,
    multibuffer: bool,
    block_h_q: int | None = None,
) -> DemandEstimate:
    """Screening estimate for one dataflow path (``m1_row`` / ``pair_tile`` /
    ``tail_row``), including the kernel-level composition rule.

    ``variant`` is one of:
      * ``"m1"``        -> the BLOCK_M==1 single-row dataflow (informational).
      * ``"tail"``      -> the odd-tail single-row loop (two M2 instances
                           share this path) == ``m1``.
      * ``"pair"``      -> BLOCK_M==2 + PAIR_CAPABLE; kernel demand is the max
                           of the pair tile body and the m1/tail body (top-level
                           disjoint live ranges, peak not sum -- assumption A1).

    ``block_h_q`` defaults to ``min(num_q_heads, 12)`` (the kernel constexpr);
    pin it higher (e.g. 24) only to reproduce the OLD full-head liveness for
    anchors A1/A2.  ``estimated_margin`` (via :func:`estimated_margin`) must
    be ``> 0`` to clear the resource gate; exact-fit (0) rejects.
    """
    if dtype not in DTYPE_WIDTH:
        raise ValueError(f"dtype {dtype!r} is outside the screening domain")
    if block_m not in (1, 2):
        raise ValueError("screening block_m must be 1 or 2")
    if multibuffer:
        beta = BETA_MULTIBUFFER_ON
    else:
        beta = BETA_MULTIBUFFER_OFF

    m1_peak = _itemize_m1_row(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        has_gate=has_gate,
        rope_dim=rope_dim,
        dtype=dtype,
        has_bias=has_bias,
    )

    assumptions: tuple[str, ...]
    if variant == "m1":
        path = "m1_row"
        peak = m1_peak
        assumptions = ("m1 is the unconditional status-quo path; not gated",)
    elif variant == "tail":
        path = "tail_row"
        peak = m1_peak
        assumptions = ("tail-only dataflow is the m1 single-row loop; not gated by capacity",)
    elif variant == "pair":
        path = "pair_tile"
        pair_peak = _itemize_pair_tile(
            block_m=block_m,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            has_gate=has_gate,
            rope_dim=rope_dim,
            dtype=dtype,
            has_bias=has_bias,
            block_h_q=block_h_q,
        )
        peak = max(pair_peak, m1_peak)
        assumptions = (
            "pair kernel demand = max(pair tile body, odd-tail body) (A1: cross top-level loop reuse)",
            "boundary tile charged at peak-not-sum (A2: unsupported after E2c screening false-positive; "
            "attribution open; diagnostic only outside the calibrated domain)",
            "arange/mask temporaries absorbed into alpha (A3); estimate is not a proven upper bound",
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")

    demand_bytes = int(math.ceil(peak * ALPHA * beta))
    return DemandEstimate(
        path=path,
        peak_bytes=peak,
        alpha=ALPHA,
        beta=beta,
        demand_estimate_bytes=demand_bytes,
        assumptions=assumptions,
    )


def estimated_margin(envelope_bytes: int, demand_estimate_bytes: int) -> int:
    """Estimate-vs-budget difference.  Only ``> 0`` clears the resource gate."""
    return envelope_bytes - demand_estimate_bytes


__all__ = [
    "ALIGN_BYTES",
    "ALPHA",
    "BETA_MULTIBUFFER_OFF",
    "BETA_MULTIBUFFER_ON",
    "DTYPE_WIDTH",
    "DemandEstimate",
    "capacity_resolve",
    "demand_estimate",
    "estimated_margin",
    "pair_boundary_screen_supported",
    "parse_toolchain_fingerprint",
]
