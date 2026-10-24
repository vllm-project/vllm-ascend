"""Input, resource and workload gates for the paired Split-QKV kernel.

No case, SoC or toolchain allowlist is consulted. Resource estimates are
screening values, not measured UB usage or a proven allocator upper bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .ac_ub_demand import estimate_demand, pair_boundary_screen_supported

_MIN_PAIR_ITERATIONS = 12  # B3-derived workload threshold, not a universal optimum.
_UB_BYTES_MIN = 64 * 1024
_UB_BYTES_MAX = 1024 * 1024


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer, not bool or another numeric type")
    return value


def _require_str(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a real bool")
    return value


@dataclass(frozen=True)
class ShapeKey:
    """Scalar input contract for adaptive M2 screening."""

    num_q_heads: int
    num_kv_heads: int
    head_size: int
    has_gate: bool
    rope_dim: int
    has_bias: bool
    is_interleaved: bool
    mrope_section: tuple[int, int, int]
    eps: float
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        for value, label in (
            (self.num_q_heads, "num_q_heads"),
            (self.num_kv_heads, "num_kv_heads"),
            (self.head_size, "head_size"),
            (self.rope_dim, "rope_dim"),
        ):
            _require_int(value, label)
        if self.num_q_heads <= 0 or self.num_kv_heads <= 0 or self.head_size <= 0:
            raise ValueError("shape dimensions must be positive")
        if self.rope_dim <= 0 or self.rope_dim > self.head_size or self.rope_dim % 2:
            raise ValueError("rope_dim must be positive, even, and no greater than head_size")
        _require_bool(self.has_gate, "has_gate")
        _require_bool(self.has_bias, "has_bias")
        _require_bool(self.is_interleaved, "is_interleaved")
        if type(self.mrope_section) is not tuple or len(self.mrope_section) != 3:
            raise TypeError("mrope_section must be a three-integer tuple")
        for value in self.mrope_section:
            _require_int(value, "mrope_section entry")
            if value < 0:
                raise ValueError("mrope_section entries must be non-negative")
        if 2 * sum(self.mrope_section) != self.rope_dim:
            raise ValueError("2 * sum(mrope_section) must equal rope_dim")
        if type(self.eps) is not float or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be a finite positive float")
        _require_str(self.dtype, "dtype")
        if self.dtype not in {"bfloat16", "float16"}:
            raise ValueError("dtype must be exactly bfloat16 or float16")


@dataclass(frozen=True)
class DispatchDecision:
    """Small, local selection result; no process-global experiment record."""

    block_m: int
    gate: str
    demand_estimate_bytes: int | None = None


def layout_valid(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    q_bias,
    k_bias,
    num_tokens,
    q_size,
    gate_size,
    kv_size,
    head_size,
    rope_dim,
) -> bool:
    """Check the shapes, contiguous storage, dtype/device and paired biases.

    M1 fallback does not make an arbitrary malformed input kernel-safe.
    """
    if (q_bias is None) != (k_bias is None):
        return False
    tensors = [
        (qkv, (num_tokens, q_size + gate_size + 2 * kv_size)),
        (q_weight, (head_size,)),
        (k_weight, (head_size,)),
        (cos_sin, (3, num_tokens, rope_dim)),
    ]
    if q_bias is not None:
        tensors.extend(((q_bias, (head_size,)), (k_bias, (head_size,))))
    try:
        return all(
            tuple(t.shape) == shape and t.is_contiguous() and t.dtype == qkv.dtype and t.device == qkv.device
            for t, shape in tensors
        )
    except Exception:
        return False


def select_dispatch(
    *,
    shape: ShapeKey,
    min_active_tokens: int,
    max_active_tokens: int,
    current_layout_valid: bool,
    capacity_bytes: int | None,
) -> DispatchDecision:
    """Use the wrapper's existing partition; keep G1 -> G2 -> G3 priority."""
    if current_layout_valid is not True or max_active_tokens <= 0 or shape.dtype not in {"bfloat16", "float16"}:
        return DispatchDecision(1, "g1_semantic")

    if type(capacity_bytes) is not int or not _UB_BYTES_MIN <= capacity_bytes <= _UB_BYTES_MAX:
        return DispatchDecision(1, "g2_resource")

    has_pair_work = max_active_tokens >= 2
    demand_bytes = estimate_demand(
        num_q_heads=shape.num_q_heads,
        num_kv_heads=shape.num_kv_heads,
        head_size=shape.head_size,
        has_gate=shape.has_gate,
        rope_dim=shape.rope_dim,
        dtype=shape.dtype,
        has_bias=shape.has_bias,
        has_pair_work=has_pair_work,
    )
    # E2c falsified the boundary estimate: retain the structural remainder
    # guard, even at a larger capacity. It is not a measured-shape allowlist.
    if has_pair_work and not pair_boundary_screen_supported(shape.num_q_heads):
        return DispatchDecision(1, "g2_resource", demand_bytes)
    if demand_bytes >= capacity_bytes:  # Exact fit is also rejected.
        return DispatchDecision(1, "g2_resource", demand_bytes)

    if min_active_tokens // 2 < _MIN_PAIR_ITERATIONS:
        return DispatchDecision(1, "g3_workload", demand_bytes)
    return DispatchDecision(2, "shape_resource", demand_bytes)
