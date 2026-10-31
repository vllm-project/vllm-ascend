/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_v2_pertoken_quant.h
 * \brief
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_PERTOKEN_QUANT_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_PERTOKEN_QUANT_H

#include "cgmct/kernel/kernel_gmm_swiglu_pertoken_quant.h"
#include "cgmct/block/block_mmad_builder.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;

static constexpr uint8_t BF16_VALUE = 27;

template <uint8_t dequantDtype, typename layoutA, typename layoutB>
__aicore__ inline void GmmSwigluAswtPertokenKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                                   GM_ADDR weightAssistanceMatrix, GM_ADDR smoothScale,
                                                   GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace,
                                                   GM_ADDR tiling, TPipe *pipe)
{
    /* 1. 取 tiling 数据 */
    GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingDataParams, gmmSwigluQuantParams, gmmSwigluQuantParams_, tiling);
    GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingDataParams, mmTilingData, mmTilingData_, tiling);

    /* 2. 编译期常量决定 DequantType / C1Type */
    using DequantType =
        std::conditional_t<dequantDtype == 1, half, std::conditional_t<dequantDtype == BF16_VALUE, bfloat16_t, float>>;
    using AType = DTYPE_X;
    using BType = DTYPE_WEIGHT;
    using CType = DTYPE_Y;                                                            // y dtype
    using C1Type = std::conditional_t<std::is_same_v<AType, int8_t>, int32_t, float>; // matmul output dtype

    /* 3. 其余别名 */
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajorAlign;
    using weightscaleType = DTYPE_WEIGHT_SCALE;
    using xscaleType = float;
    using BiasType = float;
    using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;
    using BlockEpilogueDequantAndSwiglu =
        Block::BlockEpilogueDequantSwiglu<L0TileShape, DequantType, C1Type, weightscaleType, xscaleType, true>;
    using BlockEpiloguePertokenQuant = Block::BlockEpiloguePertokenQuant<DequantType, CType>;
    using ProblemShape = MatmulShape;
    using BlockMmad =
        Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, C1Type, LayoutC, BiasType, layout::RowMajor,
                                L1TileShape, L0TileShape, BlockScheduler, MatmulMultiBlock<>,
                                Tile::TileCopy<Arch::DAV3510, Tile::CopyInAndCopyOutSplitMWithParams>>;
    using QGmmKernel =
        Kernel::KernelGmmSwiGluPertokenQuant<ProblemShape, BlockMmad, BlockEpilogueDequantAndSwiglu,
                                             BlockEpiloguePertokenQuant, BlockScheduler, weightscaleType, xscaleType>;

    /* 4. 拼参数、launch */
    using Params = typename QGmmKernel::Params;
    using GMMTiling = typename QGmmKernel::GMMTiling;
    GMMTiling gmmParams{gmmSwigluQuantParams_.groupNum, gmmSwigluQuantParams_.groupListType, mmTilingData_.baseM,
                        mmTilingData_.baseN, mmTilingData_.baseK};
    gmmParams.matmulTiling = &mmTilingData_;
    Params params = {
        {1, 1, 1, 1},
        // mmad args
        {x, weight, y, nullptr, groupList},
        {workspace, weightScale, xScale, static_cast<uint32_t>(mmTilingData_.baseM),
         static_cast<uint32_t>(mmTilingData_.baseN)},
        {workspace, smoothScale, y, yScale, gmmSwigluQuantParams_.rowLen, gmmSwigluQuantParams_.ubAvail, false},
        // gmm tiling data
        gmmParams};
    QGmmKernel op(pipe);
    op(params);
}

/* ----------------------------------------------------------
 * 5. 最外层入口：只做 switch，把运行期值 → 编译期常量
 * ---------------------------------------------------------- */
template <typename layoutA, typename layoutB>
__aicore__ inline void GmmSwigluAswtPertoken(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                             GM_ADDR weightAssistanceMatrix, GM_ADDR smoothScale, GM_ADDR groupList,
                                             GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling, TPipe *pipe)
{
    GET_TILING_DATA_MEMBER(GMMSwigluQuantTilingDataParams, gmmSwigluQuantParams, gmmSwigluQuantParams_, tiling);

    switch (gmmSwigluQuantParams_.dequantDtype) {
        case 1:
            GmmSwigluAswtPertokenKernel<1, layoutA, layoutB>(x, weight, weightScale, xScale, weightAssistanceMatrix,
                                                             smoothScale, groupList, y, yScale, workspace, tiling,
                                                             pipe);
            break;
        case BF16_VALUE:
            GmmSwigluAswtPertokenKernel<BF16_VALUE, layoutA, layoutB>(x, weight, weightScale, xScale,
                                                                      weightAssistanceMatrix, smoothScale, groupList, y,
                                                                      yScale, workspace, tiling, pipe);
            break;
        default:
            GmmSwigluAswtPertokenKernel<0, layoutA, layoutB>(x, weight, weightScale, xScale, weightAssistanceMatrix,
                                                             smoothScale, groupList, y, yScale, workspace, tiling,
                                                             pipe);
            break;
    }
}
#endif