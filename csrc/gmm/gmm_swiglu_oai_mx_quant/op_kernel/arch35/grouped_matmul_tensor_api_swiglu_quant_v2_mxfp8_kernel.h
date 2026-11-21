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
 * \file grouped_matmul_tensor_api_swiglu_quant_v2_mxfp8_kernel.h
 * \brief Shared TensorApi entry point for GroupedMatmulSwigluQuantV2 MXFP8 ND/NZ scenarios.
 *        Assembles Blaze
 * BlockMmad + Blaze Epilogue + Blaze Scheduler + MIX Kernel.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_MXQUANT_TENSOR_API_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_MXQUANT_TENSOR_API_H

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_qgmm_swiglu_mx.h"
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#include "../grouped_matmul_swiglu_quant_v2_utils_kernel.h"
#include "grouped_matmul_swiglu_quant_v2_tensor_api_tiling_data.h"

template <typename layoutA, typename layoutB, bool EnableSwigluAttrs = false>
__aicore__ inline void GmmTensorApiSwigluQuantMxFp8Kernel(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale,
                                                          GM_ADDR xScale, GM_ADDR weightAssistanceMatrix,
                                                          GM_ADDR smoothScale, GM_ADDR groupList, GM_ADDR y,
                                                          GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)weightAssistanceMatrix;
    (void)smoothScale;
    (void)workspace;

    GET_TILING_DATA_WITH_STRUCT(GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData, tilingData,
                                tiling);
    const auto& gmmQuantParams_ = tilingData.gmmQuantParams;
    const auto& mmTilingData_ = tilingData.mmTilingData;

    using AType = DTYPE_X;
    using BType = DTYPE_WEIGHT;
    using CType = DTYPE_Y;
#ifndef BLAZE_SWIGLU_MXFP4_SCALE_ALG2_SUPPORTED
    static_assert(!EnableSwigluAttrs || !AscendC::IsSameType<CType, fp4x2_e2m1_t>::value,
                  "V3 MXFP4 scaleAlg=2 requires the companion ops-tensor SwiGLU MXFP4 epilogue update.");
#endif
    using C1Type = float;
    using BiasType = float;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutC1 = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using weightscaleType = AscendC::fp8_e8m0_t;

    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockMmadPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGmmSwiGluMixMx>;
    using SwigluBlockMmad = Blaze::Gemm::Block::BlockMmad<BlockMmadPolicy, AType, LayoutA, BType, LayoutB, C1Type,
                                                          LayoutC1, BiasType, LayoutBias>;

    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<
        CType, C1Type, weightscaleType, EnableSwigluAttrs && AscendC::IsSameType<CType, bfloat16_t>::value>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;

    using SwigluKernel =
        Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, SwigluBlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename SwigluKernel::Params;
    using GMMTiling = typename SwigluKernel::GMMTiling;
    using BlockMmadAddressParams = typename SwigluBlockMmad::Params;

    const uint8_t singleW = gmmQuantParams_.singleW;
    GMMTiling gmmParams{static_cast<uint32_t>(gmmQuantParams_.groupNum),
                        static_cast<int64_t>(mmTilingData_.m),
                        static_cast<int64_t>(mmTilingData_.n),
                        static_cast<int64_t>(mmTilingData_.k),
                        static_cast<uint32_t>(mmTilingData_.baseM),
                        static_cast<uint32_t>(mmTilingData_.baseN),
                        static_cast<uint32_t>(mmTilingData_.baseK),
                        static_cast<uint32_t>(mmTilingData_.kAL1),
                        static_cast<uint32_t>(mmTilingData_.kBL1),
                        static_cast<uint32_t>(mmTilingData_.scaleKAL1),
                        static_cast<uint32_t>(mmTilingData_.scaleKBL1),
                        static_cast<uint8_t>(mmTilingData_.dbL0C),
                        static_cast<int8_t>(gmmQuantParams_.groupType),
                        static_cast<uint8_t>(gmmQuantParams_.groupListType),
                        singleW};

    GM_ADDR aDataAddr = x;
    GM_ADDR scaleADataAddr = xScale;
    GM_ADDR yDataAddr = y;
    GM_ADDR yScaleDataAddr = yScale;

    GM_ADDR bDataAddr =
        singleW == 1 ? reinterpret_cast<GM_ADDR>(GroupedMatmulDequantSwigluQuant::GetTensorAddr<BType>(0, weight)) :
                       weight;
    GM_ADDR scaleBDataAddr =
        singleW == 1 ? reinterpret_cast<GM_ADDR>(
                           GroupedMatmulDequantSwigluQuant::GetTensorAddr<AscendC::fp8_e8m0_t>(0, weightScale)) :
                       weightScale;

    GM_ADDR groupListDataAddr = groupList;

    BlockMmadAddressParams blockMmadAddressParams{aDataAddr, bDataAddr,      nullptr,
                                                  nullptr,   scaleADataAddr, scaleBDataAddr};
    typename BlockEpilogue::Params epilogueParams{yDataAddr, yScaleDataAddr, gmmParams.baseM, gmmParams.baseN};
    if constexpr (EnableSwigluAttrs) {
        const auto& swigluParams_ = tilingData.swigluParams;
        epilogueParams.swigluMode = swigluParams_.swigluMode;
        epilogueParams.clampLimit = swigluParams_.clampLimit;
        epilogueParams.gluAlpha = swigluParams_.gluAlpha;
        epilogueParams.gluBias = swigluParams_.gluBias;
        epilogueParams.scaleAlg = swigluParams_.scaleAlg;
        epilogueParams.dstTypeMax = swigluParams_.dstTypeMax;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if constexpr (AscendC::IsSameType<CType, fp8_e4m3fn_t>::value ||
                      AscendC::IsSameType<CType, fp4x2_e2m1_t>::value) {
            GroupedMatmulDequantSwigluQuant::ClearUnprocessedMxScaleRows(
                groupList, yScale, gmmParams.m, gmmParams.n / GroupedMatmulDequantSwigluQuant::SWIGLU_REDUCE_FACTOR,
                gmmParams.groupNum, gmmParams.groupListType, swigluParams_.swigluMode);
        }
#endif
    }
    Params params = {{gmmParams.m, gmmParams.n, gmmParams.k, static_cast<int64_t>(1)},
                     blockMmadAddressParams,
                     epilogueParams,
                     groupListDataAddr,
                     gmmParams};

    SwigluKernel op;
    op(params);
}

#endif // GROUPED_MATMUL_SWIGLU_QUANT_V2_MXQUANT_TENSOR_API_H
