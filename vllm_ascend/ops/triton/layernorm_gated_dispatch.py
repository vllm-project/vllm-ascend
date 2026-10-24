"""Pure-scalar PR1 launch selection for the gated LayerNorm kernel.

The selector deliberately knows nothing about torch, devices, or resource
queries. The wrapper supplies the initialized vector-core count for
single-group NPU tensors and None otherwise. Wide BASE16 selection also
receives the optional UB value from the initialized-properties getter.
"""

from __future__ import annotations

from typing import Literal, NamedTuple


class LaunchSpec(NamedTuple):
    impl: Literal["FT_BASE", "FT_PERSIST_HOIST"]
    block_m: int


BM_BASE16 = 16
BM_HOIST = 32
HOIST_QUARTER_WAVE_DIVISOR = 4
BASE16_MAX_N_GROUP = 512
BASE16_MIN_UB_BYTES = 196_608


class DispatchConfigError(ValueError):
    """Raised when a selector input is invalid."""


def _is_positive_exact_int(value) -> bool:
    return type(value) is int and value > 0


def _validate_inputs(M, N_group, ngroups, runtime_p, ub_bytes) -> None:
    for name, value in (("M", M), ("N_group", N_group), ("ngroups", ngroups)):
        if not _is_positive_exact_int(value):
            raise DispatchConfigError(f"{name} must be a positive exact int")
    if runtime_p is not None and not _is_positive_exact_int(runtime_p):
        raise DispatchConfigError("runtime_p must be None or a positive exact int")
    if ub_bytes is not None and not _is_positive_exact_int(ub_bytes):
        raise DispatchConfigError("ub_bytes must be None or a positive exact int")


def _select_layernorm_launch(
    M,
    N_group,
    ngroups,
    runtime_p,
    *,
    ub_bytes: int | None = None,
) -> LaunchSpec:
    """Select the qualified PR1 path, or retain the upstream BASE64 fallback.

    Multiple groups or a missing runtime_p select the upstream BASE64 launch.
    Single-group NPU callers obtain the count through the existing
    initialized-device-properties contract.
    Wide BASE16 selection additionally requires a known UB value at or above
    the minimum qualified resource budget.
    """
    _validate_inputs(M, N_group, ngroups, runtime_p, ub_bytes)
    if ngroups > 1 or runtime_p is None:
        return LaunchSpec("FT_BASE", 64)

    # For wide groups, BASE16 is qualified only for the tested BN256/512
    # envelope. The wrapper supplies UB from the existing properties getter.
    if N_group > 128:
        if N_group <= BASE16_MAX_N_GROUP and ub_bytes is not None and ub_bytes >= BASE16_MIN_UB_BYTES:
            return LaunchSpec("FT_BASE", BM_BASE16)
        return LaunchSpec("FT_BASE", 64)

    if N_group < 128:
        return LaunchSpec("FT_BASE", BM_BASE16)

    hoist_tiles = (M + BM_HOIST - 1) // BM_HOIST
    if HOIST_QUARTER_WAVE_DIVISOR * hoist_tiles >= runtime_p:
        return LaunchSpec("FT_PERSIST_HOIST", BM_HOIST)
    return LaunchSpec("FT_BASE", BM_BASE16)
