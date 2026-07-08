/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
#include "kernel_operator.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/solve_tri_ascend950.h"
#else
#include "lib/matmul_intf.h"
#include "solve_tri_cube.h"
#include "solve_tri_vector.h"
#endif

using namespace AscendC;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
extern "C" __global__ __aicore__ void solve_tri(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                GM_ADDR x_out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA(tilingData, tiling);

        SolveTri<DTYPE_X, DTYPE_X> op;
        op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
        op.Process();
    }
}
#else
extern "C" __global__ __aicore__ void solve_tri(GM_ADDR x, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                GM_ADDR x_out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        // MIX_AIC_1_2 模式：1 个 AIC + 2 个 AIV
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);

        int64_t ms = tilingData.matrixSize;
        int64_t totalTiles = tilingData.totalTiles;
        int64_t tilesPerCore = tilingData.tilesPerCore;
        int64_t dtypeMode = tilingData.dtypeMode;  // 0=fp16, 1=bf16

        // // AIC 核：执行 CUBE 矩阵乘法
        if ASCEND_IS_AIC {
            if (dtypeMode == 0) {
                // fp16
                if (ms == 16) {
                    NsSolveTri::SolveTriCube<16, half> op;
                    op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                    op.Process();
                } else if (ms == 32) {
                    NsSolveTri::SolveTriCube<32, half> op;
                    op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                    op.Process();
                } else if (ms == 64) {
                    NsSolveTri::SolveTriCube<64, half> op;
                    op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                    op.Process();
                } else if (ms == 128) {
                    NsSolveTri::SolveTriCube<128, half> op;
                    op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                    op.Process();
                }
            } else {
                // bf16
                if (ms == 16) {
                    NsSolveTri::SolveTriCube<16, bfloat16_t> op;
                    op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                    op.Process();
                } else if (ms == 32) {
                    NsSolveTri::SolveTriCube<32, bfloat16_t> op;
                    op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                    op.Process();
                } else if (ms == 64) {
                    NsSolveTri::SolveTriCube<64, bfloat16_t> op;
                    op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                    op.Process();
                } else if (ms == 128) {
                    NsSolveTri::SolveTriCube<128, bfloat16_t> op;
                    op.Init(x, cu_seqlens, chunk_indices, x_out, workspace, &tilingData);
                    op.Process();
                }
            }
        }

        // AIV 核：执行 Vector 操作（生成辅助矩阵）
        if ASCEND_IS_AIV {
            if (dtypeMode == 0) {
                // fp16
                if (ms == 16) {
                    NsSolveTri::SolveTriVector<16, half> op;
                    op.Init(workspace, totalTiles, ms);
                    op.Process();
                } else if (ms == 32) {
                    NsSolveTri::SolveTriVector<32, half> op;
                    op.Init(workspace, totalTiles, ms);
                    op.Process();
                } else if (ms == 64) {
                    NsSolveTri::SolveTriVector<64, half> op;
                    op.Init(workspace, totalTiles, ms);
                    op.Process();
                } else if (ms == 128) {
                    NsSolveTri::SolveTriVector<128, half> op;
                    op.Init(workspace, totalTiles, ms);
                    op.Process();
                }
            } else {
                // bf16
                if (ms == 16) {
                    NsSolveTri::SolveTriVector<16, bfloat16_t> op;
                    op.Init(workspace, totalTiles, ms);
                    op.Process();
                } else if (ms == 32) {
                    NsSolveTri::SolveTriVector<32, bfloat16_t> op;
                    op.Init(workspace, totalTiles, ms);
                    op.Process();
                } else if (ms == 64) {
                    NsSolveTri::SolveTriVector<64, bfloat16_t> op;
                    op.Init(workspace, totalTiles, ms);
                    op.Process();
                } else if (ms == 128) {
                    NsSolveTri::SolveTriVector<128, bfloat16_t> op;
                    op.Init(workspace, totalTiles, ms);
                    op.Process();
                }
            }
        }
    }
}
#endif
