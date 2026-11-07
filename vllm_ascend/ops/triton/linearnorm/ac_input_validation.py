"""Scalar and tensor layout contracts for adaptive Split-QKV tiling."""

from __future__ import annotations

import math


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer, not bool or another numeric type")
    return value


def validate_scalars(
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    has_gate: bool,
    rope_dim: int,
    has_bias: bool,
    is_interleaved: bool,
    mrope_section: tuple[int, int, int],
    eps: float,
    dtype: str,
) -> None:
    """Preserve the adaptive selector's strict scalar checks and their order."""
    for value, label in (
        (num_q_heads, "num_q_heads"),
        (num_kv_heads, "num_kv_heads"),
        (head_size, "head_size"),
        (rope_dim, "rope_dim"),
    ):
        _require_int(value, label)
    if num_q_heads <= 0 or num_kv_heads <= 0 or head_size <= 0:
        raise ValueError("shape dimensions must be positive")
    if rope_dim <= 0 or rope_dim > head_size or rope_dim % 2:
        raise ValueError("rope_dim must be positive, even, and no greater than head_size")
    for flag, label in ((has_gate, "has_gate"), (has_bias, "has_bias"), (is_interleaved, "is_interleaved")):
        if type(flag) is not bool:
            raise TypeError(f"{label} must be a real bool")
    if type(mrope_section) is not tuple or len(mrope_section) != 3:
        raise TypeError("mrope_section must be a three-integer tuple")
    for value in mrope_section:
        _require_int(value, "mrope_section entry")
        if value < 0:
            raise ValueError("mrope_section entries must be non-negative")
    if 2 * sum(mrope_section) != rope_dim:
        raise ValueError("2 * sum(mrope_section) must equal rope_dim")
    if type(eps) is not float or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive float")
    if type(dtype) is not str:
        raise TypeError("dtype must be a string")
    if dtype not in {"bfloat16", "float16"}:
        raise ValueError("dtype must be exactly bfloat16 or float16")


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
