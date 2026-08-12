/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_mat_mul_v3_iterbatch_basicapi_act.h
 * \brief
 */
#ifndef BATCH_MAT_MUL_V3_ITERBATCH_BASICAPI_ACT_H
#define BATCH_MAT_MUL_V3_ITERBATCH_BASICAPI_ACT_H
#include "include/matmul/kernel/kernel_matmul_iterbatch.h"
#include "batch_mat_mul_v3_iterbatch_basicapi_block_scheduler.h"
using namespace Act;
using namespace Act::Gemm;
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT>
__aicore__ inline void Mc2BatchMatMulActIterBatchKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                                        GM_ADDR workspaceGM,
                                                        const BatchMatMulV3IterBatchBasicTilingData &tilingData)
{
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    // 定义矩阵的类型和布局
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using OutType = C_TYPE;

    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;

    if (biasGM != nullptr) {
        return;
    }
    // 定义scheduler类型 来自block_scheduler_policy.h
    using BlockScheduler = BuiltInIterBatchScheduler;

    // 定义MMAD类型
    using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, OutType, LayoutC, BiasType, LayoutC,
                                              L1TileShape, L0TileShape, BlockScheduler, MatmulIterBatch<>>;

    // 定义BlockEpilogue类型
    using BlockEpilogue = Block::BlockEpilogueEmpty;

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;

    // 定义Kernel类型
    using MatmulKernel = Kernel::KernelMatMulIterBatch<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {{tilingData.m, tilingData.n, tilingData.k, tilingData.b}, // shape
                     {aGM, bGM, cGM, biasGM},                                  // gm addr
                     {},                                                       // epilogue args
                     {&tilingData}};
    AscendC::TPipe tPipe;
    MatmulKernel mm;
    mm(params);
}
#endif
