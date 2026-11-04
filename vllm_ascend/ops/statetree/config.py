# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tile selection and hardware limits for StateTree on Ascend 950."""

FUSED_BATCH_TILES = (16, 32, 64, 128, 256)
# Preserve native GEMM reduction schedules from this row count onward.
NATIVE_GEMM_MIN_ROWS = 256
_CUBE_CORE_COUNT = 28
_MAX_UB_BYTES = 248 * 1024


def record_sizes(n_tile: int, k_tile: int, escape_slots: int) -> tuple[int, int, int, int]:
    """Return the four byte spans in one packed decode record."""
    return (
        n_tile * k_tile // 2,
        n_tile * k_tile // 4,
        n_tile * k_tile // 64,
        escape_slots * 8,
    )


def choose_tiles(n: int, k: int) -> tuple[int, int]:
    """Choose the validated decode tile for a dense ``[N, K]`` weight."""
    if n <= 8192 and n % 256 == 0 and k % 128 == 0:
        return 256, 128
    if n > 8192 and n % 256 == 0 and k % 128 == 0:
        # A wider N tile halves repeated activation traffic. Use it only when
        # rounding work to the Cube-core count does not add padded work.
        padded_256 = ((n // 256 + _CUBE_CORE_COUNT - 1) // _CUBE_CORE_COUNT) * _CUBE_CORE_COUNT * 256
        padded_128 = ((n // 128 + _CUBE_CORE_COUNT - 1) // _CUBE_CORE_COUNT) * _CUBE_CORE_COUNT * 128
        if padded_256 <= padded_128:
            return 256, 128
    if n % 128 == 0 and k % 256 == 0:
        return 128, 256
    if n % 32 == 0 and k % 64 == 0:
        return 32, 64
    raise ValueError("StateTree requires N divisible by 32 and K divisible by 64")


def validate_tiles(batch_tile: int, n_tile: int, k_tile: int, escape_slots: int) -> None:
    if n_tile not in range(32, 257, 32) or k_tile < 64 or k_tile % 64:
        raise ValueError("N tile must be a multiple of 32 up to 256; K tile must be a multiple of 64")
    if batch_tile not in FUSED_BATCH_TILES or n_tile * k_tile > 65536:
        raise ValueError("batch tile must be 16/32/64/128/256 and the decode tile must not exceed 65536 values")
    cube_k = min(k_tile, (16384 if batch_tile >= 32 else 32768) // max(batch_tile, n_tile) // 64 * 64)
    if k_tile % cube_k or cube_k % 64 or (32768 // batch_tile) % cube_k:
        raise ValueError("Cube K partitions must divide the K tile and be aligned to 64")
    max_slots = 1 << (n_tile * k_tile // 4 - 1).bit_length()
    if escape_slots < 8 or escape_slots & (escape_slots - 1) or escape_slots > max_slots:
        raise ValueError("invalid StateTree exceptional-pair slot count")
    ub_bytes = 2 * (n_tile * k_tile + sum(record_sizes(n_tile, k_tile, escape_slots)) + 256)
    if ub_bytes > _MAX_UB_BYTES:
        raise ValueError(f"StateTree tile requires {ub_bytes} UB bytes; only {_MAX_UB_BYTES} are available")


def select_batch_tile(rows: int) -> int:
    """Select the smallest supported power-of-two tile covering the rows."""
    return min(FUSED_BATCH_TILES[-1], max(FUSED_BATCH_TILES[0], 1 << (max(1, rows) - 1).bit_length()))
