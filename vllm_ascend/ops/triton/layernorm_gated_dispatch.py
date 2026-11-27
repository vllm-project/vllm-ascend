"""Select gated LayerNorm launch parameters from shape and device properties."""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal, NamedTuple


class LaunchSpec(NamedTuple):
    impl: Literal["FT_BASE", "FT_PERSIST_HOIST"]
    block_m: int


BM_BASE16 = 16
BM_HOIST = 32
HOIST_QUARTER_WAVE_DIVISOR = 4
BASE16_MAX_N_GROUP = 512
MIN_FULL_TILE_UB_BYTES = 192 * 1024
WIDE_FT_VECTOR_CORES = 40
WIDE_FT_MIN_N_GROUP = 257
WIDE_FT_MAX_N_GROUP = 8192
WIDE_FP16_BLOCK_N = 1024
_WIDE_HOIST_MIN_ROWS_PER_CORE = 512
_MAX_I32_OFFSET = (1 << 31) - 1

WideRmsMode = Literal["bf16_post_gate", "fp16_pre_gate"]

# UB budget is checked before consulting these tile choices. Each pair gives
# the BASE and HOIST row tile for gated BF16 RMSNorm, by rounded column width.
_WIDE_BF16_ROW_TILES = MappingProxyType({512: (16, 8), 1024: (4, 4), 2048: (4, 4), 4096: (2, 2), 8192: (1, None)})


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


def _select_wide_rms_launch(M: int, N_group: int, runtime_p: int, mode: WideRmsMode) -> LaunchSpec | None:
    block_n = 1 << (N_group - 1).bit_length()
    if mode == "bf16_post_gate":
        base_bm, hoist_bm = _WIDE_BF16_ROW_TILES[block_n]
    elif mode == "fp16_pre_gate" and block_n == WIDE_FP16_BLOCK_N:
        base_bm, hoist_bm = 4, None
    else:
        return None

    rows_per_core = (M + runtime_p - 1) // runtime_p
    if hoist_bm is not None and rows_per_core >= _WIDE_HOIST_MIN_ROWS_PER_CORE:
        spec = LaunchSpec("FT_PERSIST_HOIST", hoist_bm)
    else:
        spec = LaunchSpec("FT_BASE", base_bm)

    # Eligible operands and the allocated output have row stride N_group.
    # Include the masked lanes in the last row and column tiles.
    last_row = ((M + spec.block_m - 1) // spec.block_m) * spec.block_m - 1
    if last_row * N_group + block_n - 1 > _MAX_I32_OFFSET:
        return None
    return spec


def _select_layernorm_launch(
    M,
    N_group,
    ngroups,
    runtime_p,
    *,
    ub_bytes: int | None = None,
    wide_rms_mode: WideRmsMode | None = None,
) -> LaunchSpec:
    """Select a full-tile launch. The wrapper identifies eligible wide RMSNorm layouts."""
    _validate_inputs(M, N_group, ngroups, runtime_p, ub_bytes)
    if ngroups > 1 or runtime_p is None:
        return LaunchSpec("FT_BASE", 64)

    if N_group > 128:
        if ub_bytes is None or ub_bytes < MIN_FULL_TILE_UB_BYTES:
            return LaunchSpec("FT_BASE", 64)
        if (
            wide_rms_mode is not None
            and runtime_p == WIDE_FT_VECTOR_CORES
            and WIDE_FT_MIN_N_GROUP <= N_group <= WIDE_FT_MAX_N_GROUP
        ):
            spec = _select_wide_rms_launch(M, N_group, runtime_p, wide_rms_mode)
            if spec is not None:
                return spec
        if N_group <= BASE16_MAX_N_GROUP:
            return LaunchSpec("FT_BASE", BM_BASE16)
        return LaunchSpec("FT_BASE", 64)

    if N_group < 128:
        return LaunchSpec("FT_BASE", BM_BASE16)

    hoist_tiles = (M + BM_HOIST - 1) // BM_HOIST
    if HOIST_QUARTER_WAVE_DIVISOR * hoist_tiles >= runtime_p:
        return LaunchSpec("FT_PERSIST_HOIST", BM_HOIST)
    return LaunchSpec("FT_BASE", BM_BASE16)
