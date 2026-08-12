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
 * \file mat_mul_fixpipe_basic_act.h
 * \brief
 */
#ifndef MAT_MUL_FIXPIPE_BASIC_ACT_H
#define MAT_MUL_FIXPIPE_BASIC_ACT_H

#include "include/matmul/block/block_scheduler_policy.h"
#include "include/matmul/block/block_scheduler_utils.h"
#include "block_scheduler_aswt.h"
#include "include/epilogue/block_epilogue_fixpipe.h"
#include "include/matmul/block/block_mmad_builder.h"
#include "include/matmul/kernel/kernel_matmul_mix_fixpipe_opti.h"
namespace Mc2MatmulV3Advanced {
using namespace Act;
using namespace Act::Gemm;
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
          uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void Mc2MatMulFixpipeOptiActKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                                     GM_ADDR workspaceGM, const Mc2MatMulV3BasicTilingData &tilingData,
                                                     int64_t batch = 0)
{
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_128, _256, _256>;
    using L0TileShape = AscendC::Shape<_128, _256, _64>;

    // 定义矩阵的类型和布局
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using OutType = C_TYPE;

    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;

    if (batch > 1 && biasGM != nullptr) {
        return;
    }

    // 定义scheduler类型
    using BlockScheduler = BuiltInAswtScheduler<FULL_LOAD_MODE>;

    // 定义MMAD类型
    using DispatchPolicy = MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, FULL_LOAD_MODE>;
    using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, OutType, LayoutC, BiasType, LayoutC,
                                              L1TileShape, L0TileShape, BlockScheduler, DispatchPolicy>;

    // 定义Fusion类型
    using FusionOp = Block::DefaultFusion<OutType, OutType>;

    // 定义BlockEpilogue类型
    using BlockEpilogue = Block::BlockEpilogueFixpipe<L0TileShape, OutType, OutType>;

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;

    // 定义Kernel类型
    using MatmulKernel = Kernel::KernelMatmulMixFixpipeOpti<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {{tilingData.m, tilingData.n, tilingData.k, batch}, // shape
                     {aGM, bGM, cGM, biasGM},                           // gm addr
                     {cGM},                                             // epilogue args
                     {&tilingData}};
    MatmulKernel mm;
    mm(params);
}
} // namespace Mc2MatmulV3Advanced
#endif
