/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mat_mul_streamk_basic_cmct.h
 * \brief
 */
#pragma once

#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/kernel/kernel_matmul_streamk.h"
#include "cmct/epilogue/block_epilogue_streamk.h"
#include "cmct/epilogue/block_epilogue_streamk_fusion.h"
#include "cmct/epilogue/block_epilogue_elementwise.h"
#include "cmct/epilogue/fusion/fusion_add.h"
#include "cmct/epilogue/fusion/fusion_mul.h"
#include "cmct/epilogue/fusion/default_fusion_op.h"
#include "block_scheduler_streamk.h"

using namespace Cmct;
using namespace Cmct::Gemm;

// FusionOpSelector for StreamK — primary template defaults to DefaultFusion
namespace StreamKInternal {
template <uint64_t OpType, class OutType>
struct FusionOpSelector {
    using type = Block::DefaultFusion<OutType, OutType>;
};

template <class OutType>
struct FusionOpSelector<OP_TYPE_MUL, OutType> {
    using type = Block::FusionMul<OutType, OutType>;
};

template <class OutType>
struct FusionOpSelector<OP_TYPE_ADD, OutType> {
    using type = Block::FusionAdd<OutType, OutType>;
};

// BlockEpilogueSelector: ADD/MUL use BlockEpilogueStreamKFusion, others use BlockEpilogueStreamK
template <uint64_t OpType, class OutType, class DispatchPolicy, int8_t INNER_PRECISE>
struct BlockEpilogueSelector {
    using type = Block::BlockEpilogueStreamK<float, OutType, DispatchPolicy>;
};

template <class OutType, class DispatchPolicy, int8_t INNER_PRECISE>
struct BlockEpilogueSelector<OP_TYPE_ADD, OutType, DispatchPolicy, INNER_PRECISE> {
    using type = Block::BlockEpilogueStreamKFusion<float, OutType, DispatchPolicy, INNER_PRECISE>;
};

template <class OutType, class DispatchPolicy, int8_t INNER_PRECISE>
struct BlockEpilogueSelector<OP_TYPE_MUL, OutType, DispatchPolicy, INNER_PRECISE> {
    using type = Block::BlockEpilogueStreamKFusion<float, OutType, DispatchPolicy, INNER_PRECISE>;
};
} // namespace StreamKInternal
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
          MatMulL0C2Out MATMUL_L0C2OUT, uint64_t FUSED_OP_TYPE = 0, int8_t INNER_PRECISE = 0>
__aicore__ inline void MatMulStreamKActKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
    GM_ADDR cGM, GM_ADDR workspaceGM, const MatMulV3BasicTilingData& tilingData,
    int64_t batch = 1, GM_ADDR x3GM = nullptr, int64_t batchX3 = 1)
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

    // 定义scheduler类型 来自block_scheduler_policy.h
    using BlockScheduler = BuiltInStreamKScheduler;

    // 定义MMAD类型
    using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, OutType, LayoutC, BiasType, LayoutC,
                                              L1TileShape, L0TileShape, BlockScheduler,
                                              MatmulMultiBlockWithStreamK<MATMUL_L0C2OUT, FUSED_OP_TYPE>>;

    // 定义Fusion类型
    using FusionOp = typename StreamKInternal::FusionOpSelector<FUSED_OP_TYPE, OutType>::type;

    // 定义BlockEpilogue类型: ADD/MUL用BlockEpilogueStreamKFusion, 其他用BlockEpilogueStreamK
    using DispatchPolicy = MatmulMultiBlockWithStreamK<MATMUL_L0C2OUT, FUSED_OP_TYPE>;
    using BlockEpilogue = typename StreamKInternal::BlockEpilogueSelector<
        FUSED_OP_TYPE, OutType, DispatchPolicy, INNER_PRECISE>::type;

    // FusedEpilogue不再使用(融合已在BlockEpilogueStreamKFusion中完成)，保持Empty以兼容模板
    using FusedEpilogue = Block::BlockEpilogueEmpty;

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;

    // 定义Kernel类型
    using MatmulKernel = Kernel::KernelMatmulStreamK<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,
                                                     FUSED_OP_TYPE, OutType, FusedEpilogue>;
    using Params = typename MatmulKernel::Params;
    Params params = {
        {tilingData.m, tilingData.n, tilingData.k, batch}, // shape
        {aGM, bGM, cGM, biasGM, nullptr, workspaceGM}, // gm addr
        {cGM, workspaceGM, x3GM, (x3GM != nullptr && batch > 1 && batchX3 == 1), static_cast<uint64_t>(tilingData.m)}, // epilogue args
        {&tilingData}
    };
    MatmulKernel mm;
    mm(params);
}
