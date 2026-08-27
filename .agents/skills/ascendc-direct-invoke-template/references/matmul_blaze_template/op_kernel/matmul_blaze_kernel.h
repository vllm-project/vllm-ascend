/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_BLAZE_KERNEL_H
#define MATMUL_BLAZE_KERNEL_H

#include "kernel_operator.h"

#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/tile_trait.h"

#include "matmul_tiling_data.h"

template <bool TransA, bool TransB, uint64_t FullLoadMode>
__global__ __aicore__ __cube__ void matmul_blaze_kernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dScaleA, GM_ADDR dScaleB, GM_ADDR dC,
    const QuantMatmulTilingData tilingData)
{
    using TypeA = fp8_e4m3fn_t;
    using TypeB = fp8_e4m3fn_t;
    using TypeC = bfloat16_t;
    using BiasType = float;
    using TypeScaleA = fp8_e8m0_t;
    using TypeScaleB = fp8_e8m0_t;

    using LayoutA = AscendC::Std::conditional_t<TransA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutB = AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode>;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC, BiasType, LayoutBias>;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, FullLoadMode, LayoutA, LayoutB, TypeA>;

    using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<
        ProblemShape, BlockMmad, void, BlockScheduler>;

    using Params = typename KernelImpl::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using QBMMTiling = typename KernelImpl::QBMMTiling;

    ProblemShape problemShape{
        static_cast<int64_t>(tilingData.m),
        static_cast<int64_t>(tilingData.n),
        static_cast<int64_t>(tilingData.k),
        1L};

    BlockMmadParams mmadParams{dA, dB, dC, nullptr, dScaleA, dScaleB};

    L1Params l1Params{
        static_cast<uint64_t>(tilingData.stepK) * tilingData.baseK,
        tilingData.scaleKL1,
        2UL};

    BlockSchedulerParams schedulerParams{
        static_cast<int64_t>(tilingData.baseM),
        static_cast<int64_t>(tilingData.baseN),
        static_cast<int64_t>(tilingData.mTailTile),
        static_cast<int64_t>(tilingData.nTailTile),
        static_cast<int64_t>(tilingData.mBaseTailSplitCnt),
        static_cast<int64_t>(tilingData.nBaseTailSplitCnt),
        static_cast<int64_t>(tilingData.mTailMain),
        static_cast<int64_t>(tilingData.nTailMain)};

    QBMMTiling qbmmParams{
        1U, 1U, 1U, 1U,
        1U, 1U, 1U, 1U,
        1U, 1U, 1U, 1U,
        0U,
        tilingData.baseM,
        tilingData.baseN,
        tilingData.baseK,
        0U,
        tilingData.dbL0c};

    Params params{problemShape, mmadParams, l1Params, schedulerParams, qbmmParams};
    KernelImpl kernel;
    kernel(params);
}

#endif
