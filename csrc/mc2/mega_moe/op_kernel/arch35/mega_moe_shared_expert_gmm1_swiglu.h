/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SHARED_EXPERT_GMM1_SWIGLU_H
#define MEGA_MOE_SHARED_EXPERT_GMM1_SWIGLU_H

#include "mega_moe_impl.h"

namespace MegaMoeImpl {

using SharedExpertGmm1ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;

struct SharedExpertGmm1SwigluContext {
    BlockJobContext blockJob;
    uint32_t sharedExpertNum;
    uint32_t tokenNum;
    uint32_t tokenHiddenDim;
    uint32_t gmm1OutputDim;
    int32_t groupedMatmulMode;
    bool isPerExpertWeightTensor;
};

struct SharedExpertGmm1SwigluArgs {
    GM_ADDR inputDataPtr;
    GM_ADDR inputScalePtr;
    GM_ADDR gmm1MmadResultPtr;
    GM_ADDR weight1TensorListAddr;
    GM_ADDR weight1ScaleTensorListAddr;
    GM_ADDR swigluOutputDataPtr;
    GM_ADDR swigluOutputScalePtr;
    GM_ADDR gmmToEpilogueFlagPtr;
    float clampLimit;
    uint8_t actMode;
    uint8_t actSubMode;
    float activationAlpha;
    float activationBeta;
};

struct SharedExpertGmm1SwigluState {
    uint32_t &startBlockIdx;
    int32_t &vecSetSyncCom;
    int32_t &gmTileSequence;
    uint16_t &pingpongIdx;
};

// Prototype: MegaMoe::UpdateSharedGlobalBuffer<GMM1>. Builds one shared expert's GMM1 and SwiGLU GM views.
template <typename ActivationType, typename WeightType, typename SwigluOutType, typename QuantScaleType,
          bool EnableA8W4, typename BlockEpilogue>
__aicore__ inline void UpdateSharedExpertGmm1GlobalBuffer(const SharedExpertGmm1SwigluContext &context,
                                                          const SharedExpertGmm1SwigluArgs &args,
                                                          BlockEpilogue &epilogueOp, GMMAddrInfo &gmmAddrInfo,
                                                          uint32_t sharedExpertIdx)
{
    constexpr uint32_t weightElemsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElemsPerByte = PackedElementTraits<SwigluOutType>::ELEMENTS_PER_BYTE;
    uint64_t m = context.tokenNum;
    uint64_t n = context.gmm1OutputDim;
    uint64_t k = context.tokenHiddenDim;
    uint64_t scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t swigluN = n / SWIGLU_N_HALF;
    uint64_t scaleN = Ops::Base::CeilDiv(swigluN, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint64_t expertIdx = sharedExpertIdx;

    gmmAddrInfo.aGlobal = args.inputDataPtr;
    gmmAddrInfo.aScaleGlobal = args.inputScalePtr;
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmm1OutGlobal = args.gmm1MmadResultPtr + expertIdx * m * n * sizeof(bfloat16_t);
    }
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(
        args.weight1TensorListAddr, context.isPerExpertWeightTensor, sharedExpertIdx,
        expertIdx * n * k / weightElemsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        args.weight1ScaleTensorListAddr, context.isPerExpertWeightTensor, sharedExpertIdx,
        expertIdx * n * scaleK);
    gmmAddrInfo.swigluToGmm2Flag = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t *>(args.gmmToEpilogueFlagPtr);
    }

    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            static_cast<int64_t>(expertIdx * m * swigluN / outputElemsPerByte),
            static_cast<int64_t>(expertIdx * m * scaleN),
            0L,
            0L,
            0L,
            0L};
        epilogueOp.UpdateGlobalAddr(vecBaseOffset);
    }
}

// Prototype: MegaMoe::GroupMatmulWithSwigluQuant<true>. Dispatches one shared-expert GMM1/SwiGLU stage.
template <typename QuantOutType, typename WeightType, typename SwigluOutType, typename QuantScaleType, bool EnableA8W4,
          uint32_t Gmm1TileM, bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false,
          typename BlockEpilogue>
__aicore__ inline void RunSharedExpertGmm1SwigluStage(const SharedExpertGmm1SwigluContext &context,
                                                      const Params &gmmParams, BlockEpilogue &epilogueOp,
                                                      const GMMAddrInfo &gmmAddrInfo,
                                                      const SharedExpertGmm1ProblemShape &problemShape,
                                                      SharedExpertGmm1SwigluState &runtimeState,
                                                      uint32_t sharedExpertIdx)
{
    uint32_t expertBeforeCnt = sharedExpertIdx * context.tokenNum;
    if constexpr (EnableA8W4) {
        MegaMoeImpl::GroupMatmulSwigluQuantA8W4<QuantOutType, WeightType, bfloat16_t, QuantScaleType, QuantScaleType,
                                                Gmm1TileM, MegaMoeImpl::L1_TILE_M_256, false, true>(
            epilogueOp, gmmParams, problemShape, gmmAddrInfo, runtimeState.startBlockIdx,
            runtimeState.gmTileSequence,
            context.blockJob, expertBeforeCnt, sharedExpertIdx);
    } else if (context.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
               context.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType,
                                            QuantScaleType, true, Gmm1TileM, MegaMoeImpl::L1_TILE_M_256, false, true,
                                            IsGmm1Interleaved, IsWaveFlagGrained>(
            epilogueOp, gmmParams, problemShape, gmmAddrInfo, runtimeState.startBlockIdx, runtimeState.vecSetSyncCom,
            context.blockJob, expertBeforeCnt, sharedExpertIdx, runtimeState.pingpongIdx);
    } else {
        MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType,
                                            QuantScaleType, false, Gmm1TileM, MegaMoeImpl::L1_TILE_M_256, false, true,
                                            IsGmm1Interleaved, IsWaveFlagGrained>(
            epilogueOp, gmmParams, problemShape, gmmAddrInfo, runtimeState.startBlockIdx, runtimeState.vecSetSyncCom,
            context.blockJob, expertBeforeCnt, sharedExpertIdx, runtimeState.pingpongIdx);
    }
}

// Prototype: MegaMoe::ProcessSharedExpertGmm1. Runs shared-expert GMM1 and SwiGLU for one logical MIX block job.
template <typename QuantOutType, typename ActivationType, typename WeightType, typename SwigluOutType,
          typename QuantScaleType, bool EnableA8W4, uint32_t Gmm1TileM, bool IsGmm1Interleaved = false,
          bool IsWaveFlagGrained = false, typename BlockEpilogue>
__aicore__ inline void ControlSharedExpertGmm1SwigluPipeline(
    const SharedExpertGmm1SwigluContext &context, const SharedExpertGmm1SwigluArgs &args,
    const Params &gmmParams, BlockEpilogue &epilogueOp, SharedExpertGmm1SwigluState &runtimeState)
{
    if (context.blockJob.totalJobs == 0U || context.blockJob.jobIndex >= context.blockJob.totalJobs ||
        context.sharedExpertNum == 0U) {
        return;
    }

    typename BlockEpilogue::Params epilogueParams{
        args.swigluOutputDataPtr, args.swigluOutputScalePtr, nullptr, nullptr, nullptr, nullptr, nullptr,
        args.clampLimit, args.actMode, args.actSubMode, args.activationAlpha, args.activationBeta};
    epilogueOp.Init(epilogueParams);

    SharedExpertGmm1ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = context.tokenNum;
    Get<N_VALUE>(problemShape) = context.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = context.tokenHiddenDim;
    GMMAddrInfo gmmAddrInfo{};

    for (uint32_t sharedExpertIdx = 0; sharedExpertIdx < context.sharedExpertNum; ++sharedExpertIdx) {
        UpdateSharedExpertGmm1GlobalBuffer<ActivationType, WeightType, SwigluOutType, QuantScaleType, EnableA8W4>(
            context, args, epilogueOp, gmmAddrInfo, sharedExpertIdx);
        RunSharedExpertGmm1SwigluStage<QuantOutType, WeightType, SwigluOutType, QuantScaleType, EnableA8W4,
                                       Gmm1TileM, IsGmm1Interleaved, IsWaveFlagGrained>(
            context, gmmParams, epilogueOp, gmmAddrInfo, problemShape, runtimeState, sharedExpertIdx);
    }
    EndSync<IsGmm1Interleaved>(runtimeState.vecSetSyncCom, runtimeState.pingpongIdx);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_SHARED_EXPERT_GMM1_SWIGLU_H
