/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM1_SWIGLU_H
#define MEGA_MOE_GMM1_SWIGLU_H

#include "mega_moe_impl.h"

namespace MegaMoeImpl {

using Gmm1ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;

struct Gmm1ExpertLoopState {
    Gmm1ProblemShape problemShape;
    int64_t rowOffset;
};

struct Gmm1SwigluContext {
    BlockJobContext blockJob;
    BlockWorkspaceContext countWorkspace;
    uint32_t expertCapacityPerRank;
    uint32_t expertPerRank;
    uint32_t tokenHiddenDim;
    uint32_t gmm1OutputDim;
    int32_t dispatchFlagSlotsPerExpert;
    int32_t swigluFlagSlotsPerExpert;
    uint32_t maxTilesPerExpert;
    int32_t groupedMatmulMode;
    uint32_t activationElementsPerByte;
    bool isPerExpertWeightTensor;
};

struct Gmm1SwigluArgs {
    GM_ADDR expertRevTokenNumsPtr;
    GM_ADDR dispatchRevDataPtr;
    GM_ADDR dispatchRevScalePtr;
    GM_ADDR gmm1MmadResPtr;
    GM_ADDR weight1TensorListAddr;
    GM_ADDR weight1ScaleTensorListAddr;
    GM_ADDR dispatchToGmm1FlagPtr;
    GM_ADDR sendCntFlagPtr;
    GM_ADDR gmm1TileStatusPtr;
    GM_ADDR gmmToEpilogueFlagPtr;
};

struct Gmm1SwigluScratch {
    GlobalTensor<int32_t> expertRevNumsGlobalTensor;
};

struct Gmm1SwigluState {
    uint32_t &startBlockIdx;
    int32_t &vecSetSyncCom;
    int32_t &gmTileSequence;
    uint16_t &pingpongIdx;
};

__aicore__ inline Gmm1ExpertLoopState CreateGmm1ExpertLoopState(const Gmm1SwigluContext &context)
{
    Gmm1ProblemShape shape;
    Get<N_VALUE>(shape) = context.gmm1OutputDim;
    Get<K_VALUE>(shape) = context.tokenHiddenDim;
    return Gmm1ExpertLoopState{shape, 0};
}

// Waits for the count producer when needed, then updates the current expert's GMM1 M dimension.
template <bool EnableA8W4>
__aicore__ inline bool WaitAndUpdateGmm1GroupParams(const Gmm1SwigluContext &context,
                                                    const Gmm1SwigluArgs &args, Gmm1SwigluScratch &scratch,
                                                    Gmm1ExpertLoopState &state, uint32_t expertIdx, uint64_t sendCnt)
{
    if constexpr (g_coreType == AIV && !EnableA8W4) {
        if (GetSubBlockIdx() != 0) {
            Get<M_VALUE>(state.problemShape) = sendCnt;
            return sendCnt != 0;
        }
    }

    if (expertIdx != 0) {
        state.rowOffset += Get<M_VALUE>(state.problemShape);
    }

    if (GetSubBlockIdx() == 0) {
        __gm__ int32_t *sendCntFlag =
            reinterpret_cast<__gm__ int32_t *>(args.sendCntFlagPtr) +
            static_cast<uint64_t>(expertIdx) * context.countWorkspace.blockNum * INT_CACHELINE +
            static_cast<uint64_t>(context.countWorkspace.blockIdx) * INT_CACHELINE;
        while (AscendC::ReadGmByPassDCache(sendCntFlag) == 0) {
            int64_t startCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - startCycle < 100) {
            }
        }
        uint64_t offset = expertIdx * INT32_PER_256B * context.countWorkspace.blockNum +
                          INT32_PER_256B * context.countWorkspace.blockIdx;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            scratch.expertRevNumsGlobalTensor[offset]);
        Get<M_VALUE>(state.problemShape) = scratch.expertRevNumsGlobalTensor.GetValue(offset);
    } else {
        Get<M_VALUE>(state.problemShape) = sendCnt;
    }
    return Get<M_VALUE>(state.problemShape) != 0;
}

template <typename ActivationType, typename WeightType, typename SwigluOutType, typename QuantScaleType,
          bool EnableA8W4, bool TopkWeightsPrefetch, typename BlockEpilogue>
__aicore__ inline void UpdateGmm1GlobalBuffer(
    const Gmm1SwigluContext &context, const Gmm1SwigluArgs &args, BlockEpilogue &epilogueOp,
    GMMAddrInfo &gmmAddrInfo, const Gmm1ExpertLoopState &state, uint32_t expertIdx)
{
    if constexpr (g_coreType == AIV && !EnableA8W4) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
    }

    constexpr uint32_t weightElementsPerByte = PackedElementTraits<WeightType>::ELEMENTS_PER_BYTE;
    constexpr uint32_t outputElementsPerByte = PackedElementTraits<SwigluOutType>::ELEMENTS_PER_BYTE;
    int64_t n = Get<N_VALUE>(state.problemShape);
    int64_t k = Get<K_VALUE>(state.problemShape);
    int64_t expertOffset = expertIdx;
    int64_t scaleK = Ops::Base::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    if constexpr (EnableA8W4 || TopkWeightsPrefetch) {
        gmmAddrInfo.gmm1OutGlobal = args.gmm1MmadResPtr + state.rowOffset * n * sizeof(bfloat16_t);
    }
    gmmAddrInfo.aGlobal =
        args.dispatchRevDataPtr + state.rowOffset * k / context.activationElementsPerByte * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = args.dispatchRevScalePtr + state.rowOffset * scaleK * sizeof(QuantScaleType);
    gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(
        args.weight1TensorListAddr, context.isPerExpertWeightTensor, expertIdx,
        static_cast<uint64_t>(expertIdx) * n * k / weightElementsPerByte);
    gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleType>(
        args.weight1ScaleTensorListAddr, context.isPerExpertWeightTensor, expertIdx,
        static_cast<uint64_t>(expertIdx) * n * scaleK);
    if constexpr (g_coreType == AIV) {
        bool runsSwiglu = true;
        if constexpr (EnableA8W4) {
            runsSwiglu = GetSubBlockIdx() == 1;
        }
        if (runsSwiglu) {
            int64_t scaleN =
                Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                MXFP_MULTI_BASE_SIZE;
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                state.rowOffset * n / SWIGLU_N_HALF / outputElementsPerByte,
                state.rowOffset * scaleN,
                expertOffset * context.swigluFlagSlotsPerExpert / INT_CACHELINE,
                0L,
                0L,
                0L};
            epilogueOp.UpdateGlobalAddr(vecBaseOffset);
        }
    }
    if constexpr (g_coreType == AIC) {
        gmmAddrInfo.dispatchToGmm1Flag = reinterpret_cast<__gm__ int32_t *>(args.dispatchToGmm1FlagPtr) +
                                         expertOffset * context.dispatchFlagSlotsPerExpert;
    }
    if constexpr (EnableA8W4) {
        gmmAddrInfo.gmmToEpilogueFlag = reinterpret_cast<__gm__ int32_t *>(args.gmmToEpilogueFlagPtr);
    }
}

template <typename QuantOutType, typename WeightType, typename SwigluOutType, typename QuantScaleType, bool EnableA8W4,
          uint32_t Gmm1TileM, uint32_t EpilogueTileM, bool TopkWeightsPrefetch,
          bool IsGmm1Interleaved = false, bool IsWaveFlagGrained = false, typename BlockEpilogue>
__aicore__ inline void ControlGmm1SwigluPipelineByMode(
    const Gmm1SwigluContext &context, const Params &gmmParams, BlockEpilogue &epilogueOp,
    const GMMAddrInfo &gmmAddrInfo, const Gmm1ExpertLoopState &state, Gmm1SwigluState &runtimeState,
    uint32_t expertIdx)
{
    if constexpr (EnableA8W4) {
        MegaMoeImpl::GroupMatmulSwigluQuantA8W4<QuantOutType, WeightType, bfloat16_t, QuantScaleType, QuantScaleType,
                                                Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch, false>(
            epilogueOp, gmmParams, state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx,
            runtimeState.gmTileSequence, context.blockJob, static_cast<uint32_t>(state.rowOffset), expertIdx);
    } else if (context.groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
               context.groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
        MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType,
                                            QuantScaleType, true, Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch,
                                            false, IsGmm1Interleaved, IsWaveFlagGrained>(
            epilogueOp, gmmParams, state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx,
            runtimeState.vecSetSyncCom, context.blockJob, static_cast<uint32_t>(state.rowOffset), expertIdx,
            runtimeState.pingpongIdx);
    } else {
        MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluOutType, QuantOutType, bfloat16_t, QuantScaleType,
                                            QuantScaleType, false, Gmm1TileM, EpilogueTileM, TopkWeightsPrefetch,
                                            false, IsGmm1Interleaved, IsWaveFlagGrained>(
            epilogueOp, gmmParams, state.problemShape, gmmAddrInfo, runtimeState.startBlockIdx,
            runtimeState.vecSetSyncCom, context.blockJob, static_cast<uint32_t>(state.rowOffset), expertIdx,
            runtimeState.pingpongIdx);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM1_SWIGLU_H
