# SPDX-License-Identifier: Apache-2.0
import tilelang.ascend.language as T
GATE_TILE_ROWS = 32


def gate_projection_program(c):
    """Compute both strided gate projections while reusing each weight tile across tokens."""
    rows = T.dynamic("rows")
    width = c.heads * c.dim
    pitch = 3 * width + c.heads + 2 * c.dim
    tile_rows, tile_cols = GATE_TILE_ROWS, 512
    tiles = width // tile_cols
    padded_rows = T.ceildiv(rows, tile_rows) * tile_rows

    @T.prim_func
    def main(
        P: T.Tensor((rows, pitch), "bfloat16"),
        WF: T.Tensor((width, c.dim), "bfloat16"),
        WG: T.Tensor((width, c.dim), "bfloat16"),
        GF: T.Tensor((padded_rows, width), "bfloat16"),
        GG: T.Tensor((padded_rows, width), "bfloat16"),
    ):
        with T.Kernel(2 * tiles) as block:
            column = (block % tiles) * tile_cols
            gate = block // tiles
            start = pitch - 2 * c.dim + gate * c.dim
            activation = T.alloc_l1((tile_rows, c.dim), "bfloat16")
            weight = T.alloc_l1((tile_cols, c.dim), "bfloat16")
            accum = T.alloc_l0c((tile_rows, tile_cols), "float32")
            if gate == 0:
                T.copy(WF[column : column + tile_cols, :], weight)
            else:
                T.copy(WG[column : column + tile_cols, :], weight)
            for tile in T.Pipelined(T.ceildiv(rows, tile_rows), num_stages=2):
                row = tile * tile_rows
                count = T.min(tile_rows, rows - row)
                T.copy(P[row : row + count, start : start + c.dim], activation[:count, :])
                T.gemm(activation, weight, accum, transpose_B=True, clear_accum=True)
                # Cube stores a complete tile. The caller passes only valid rows to the consumer.
                if gate == 0:
                    T.copy(accum, GF[row : row + tile_rows, column : column + tile_cols])
                else:
                    T.copy(accum, GG[row : row + tile_rows, column : column + tile_cols])

    return main


