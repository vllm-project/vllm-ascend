"""Pure-scalar PR1 launch selection for the gated LayerNorm kernel.

The selector deliberately knows nothing about torch, devices, or resource
queries. The wrapper supplies the initialized vector-core count on NPU and
None for non-NPU tensors. Wide FT16 selection also receives the optional
UB value from the existing initialized-properties getter.
"""

from __future__ import annotations

from typing import Literal, NamedTuple


class LaunchSpec(NamedTuple):
    impl: Literal["FT_BASE", "FT_PERSIST_HOIST"]
    block_m: int


BM_SMALL = 16
BM_MULTI = 32
BM_HOIST = 32
HOIST_QUARTER_WAVE_DIVISOR = 4
BM_FT16 = 16
FT16_MAX_N_GROUP = 512
FT16_MIN_UB_BYTES = 196_608


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

    A missing runtime_p selects the upstream BASE64 launch. NPU callers
    obtain it through the existing initialized-device-properties contract.
    Wide FT16 selection additionally requires a known UB value at or above
    the minimum qualified resource budget.
    """
    _validate_inputs(M, N_group, ngroups, runtime_p, ub_bytes)
    if runtime_p is None:
        return LaunchSpec("FT_BASE", 64)

    # FT16 is qualified only for the tested BN256/512 envelope. The wrapper
    # supplies UB from the existing initialized-properties getter only here.
    if N_group > 128:
        if N_group <= FT16_MAX_N_GROUP and ub_bytes is not None and ub_bytes >= FT16_MIN_UB_BYTES:
            return LaunchSpec("FT_BASE", BM_FT16)
        return LaunchSpec("FT_BASE", 64)

    if N_group < 128:
        return LaunchSpec("FT_BASE", BM_SMALL)

    # Keep grouped N=128 execution on the qualified non-persistent BASE32 path.
    if ngroups > 1:
        return LaunchSpec("FT_BASE", BM_MULTI)

    hoist_tiles = (M + BM_HOIST - 1) // BM_HOIST
    if HOIST_QUARTER_WAVE_DIVISOR * hoist_tiles >= runtime_p:
        return LaunchSpec("FT_PERSIST_HOIST", BM_HOIST)
    return LaunchSpec("FT_BASE", BM_SMALL)
