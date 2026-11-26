# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two BF16 KDA gate projections in one A3 Cube launch.

Uses the ``tile-ai/tilelang-ascend`` AscendC backend checked at commit
``9b5ac505c801ce038e31cc8ab6f1f83e14df1497``. ``T.symbolic`` follows the
upstream linear_attention_causal example; explicit Cube scopes and transposed
L1 GEMM follow the upstream GEMM examples. This is a benchmark candidate,
not a measured replacement for the serving gate projections.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

HEAD_DIM = 128
GATE_COUNT = 2
COLUMN_TILE = 128
DEFAULT_ROW_TILE = 32
ROW_TILE_CANDIDATES = (16, 32, 64)


@lru_cache(maxsize=32, typed=True)
def _compile_gate_kernel(heads: int, row_tile: int):
    # Lazy loading keeps CPU-only fixture tools independent of CANN/TileLang.
    import tilelang
    import tilelang.language as T

    width = heads * HEAD_DIM
    pitch = 3 * width + heads + GATE_COUNT * HEAD_DIM
    column_tiles = width // COLUMN_TILE
    padded_rows = T.symbolic("padded_rows")

    @tilelang.jit(out_idx=[3, 4], target="ascendc", platform="A3")
    def build():
        @T.prim_func
        def main(
            projected: T.Tensor((padded_rows, pitch), "bfloat16"),
            weight_f: T.Tensor((width, HEAD_DIM), "bfloat16"),
            weight_g: T.Tensor((width, HEAD_DIM), "bfloat16"),
            gate_f: T.Tensor((padded_rows, width), "bfloat16"),
            gate_g: T.Tensor((padded_rows, width), "bfloat16"),
        ):
            with T.Kernel(GATE_COUNT * column_tiles, is_npu=True) as (block, _):
                column = (block % column_tiles) * COLUMN_TILE
                gate = block // column_tiles
                gate_start = pitch - GATE_COUNT * HEAD_DIM + gate * HEAD_DIM
                activation_l1 = T.alloc_L1((row_tile, HEAD_DIM), "bfloat16")
                weight_l1 = T.alloc_L1((COLUMN_TILE, HEAD_DIM), "bfloat16")
                accum_l0c = T.alloc_L0C((row_tile, COLUMN_TILE), "float")

                with T.Scope("C"):
                    # Each block owns one gate/output-column tile. Its weight
                    # tile stays in L1 across every token panel in this launch.
                    if gate == 0:
                        T.copy(weight_f[column : column + COLUMN_TILE, 0:HEAD_DIM], weight_l1)
                    else:
                        T.copy(weight_g[column : column + COLUMN_TILE, 0:HEAD_DIM], weight_l1)
                    T.barrier_all()

                    for panel in T.serial(padded_rows // row_tile):
                        row = panel * row_tile
                        # Preserve the merged projection's physical row stride;
                        # no activation slice is materialized in a separate op.
                        T.copy(
                            projected[row : row + row_tile, gate_start : gate_start + HEAD_DIM],
                            activation_l1,
                        )
                        T.barrier_all()
                        T.gemm_v0(activation_l1, weight_l1, accum_l0c, transpose_B=True, init=True)
                        T.barrier_all()
                        if gate == 0:
                            T.copy(accum_l0c, gate_f[row : row + row_tile, column : column + COLUMN_TILE])
                        else:
                            T.copy(accum_l0c, gate_g[row : row + row_tile, column : column + COLUMN_TILE])
                        # Complete FIX before reusing the accumulator. This
                        # conservative baseline does not claim double buffering.
                        T.barrier_all()

        return main

    return build()


@dataclass(frozen=True)
class GateProjection:
    """A valid-row view over a cached, dynamically sized padded-row kernel."""

    rows: int
    heads: int
    row_tile: int
    compiled_kernel: Any

    @property
    def padded_rows(self) -> int:
        return (self.rows + self.row_tile - 1) // self.row_tile * self.row_tile

    @property
    def width(self) -> int:
        return self.heads * HEAD_DIM

    @property
    def pitch(self) -> int:
        return 3 * self.width + self.heads + GATE_COUNT * HEAD_DIM

    def get_kernel_source(self):
        return self.compiled_kernel.get_kernel_source()

    def __call__(self, projected, weight_f, weight_g):
        import torch

        expected_shapes = (
            ("projected", projected, (self.padded_rows, self.pitch)),
            ("weight_f", weight_f, (self.width, HEAD_DIM)),
            ("weight_g", weight_g, (self.width, HEAD_DIM)),
        )
        for name, tensor, shape in expected_shapes:
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
            if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous BF16")
        gate_f, gate_g = self.compiled_kernel(projected, weight_f, weight_g)
        # Row slicing is a tensor view, with no device copy or second launch.
        return gate_f[: self.rows], gate_g[: self.rows]


def make_gate_kernel(rows: int, heads: int, row_tile: int = DEFAULT_ROW_TILE) -> GateProjection:
    """Build a reusable A3 two-gate callable for the A5 benchmark layout.

    The returned callable takes ``(projected_padded, W_f, W_g)``. All three
    inputs are contiguous BF16 NPU tensors; the weights have shape
    ``[heads * 128, 128]`` and are consumed transposed by Cube.

    ``projected_padded`` has shape ``[padded_rows, pitch]``, where
    ``padded_rows = ceil(rows / row_tile) * row_tile`` and
    ``pitch = 3 * heads * 128 + heads + 2 * 128``. Preserve the original
    ``[Q, K, V, beta, f_a, g_a]`` column layout and zero-pad only rows.
    ``f_a`` and ``g_a`` occupy the final two 128-column regions.

    Returns two contiguous BF16 views of shape ``[rows, heads * 128]``:
    ``f_a @ W_f.T`` and ``g_a @ W_g.T``. Accumulation is FP32; no sigmoid,
    bounded decay, or other gate activation is applied here. Padding and its
    cost belong to the caller and should be included in end-to-end timing.

    Only ``heads`` and ``row_tile`` specialize the device program. Different
    ``rows`` values share that program and vary its symbolic padded-row size.
    Row tiles 16/32/64 are tuning candidates, not measured dispatch choices.
    The underlying compiled callable and generated source remain accessible
    through ``compiled_kernel`` and ``get_kernel_source()`` respectively.
    """
    for name, value in (("rows", rows), ("heads", heads)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if not isinstance(row_tile, int) or isinstance(row_tile, bool) or row_tile not in ROW_TILE_CANDIDATES:
        raise ValueError(f"row_tile must be one of {ROW_TILE_CANDIDATES}, got {row_tile!r}")
    return GateProjection(rows, heads, row_tile, _compile_gate_kernel(heads, row_tile))
