/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_DISPATCH_GMM1_SWIGLU_H
#define MEGA_MOE_DISPATCH_GMM1_SWIGLU_H

#include "aiv_comm/mega_moe_token_dispatch.h"
#include "aiv_compute/mega_moe_expert_token_count.h"
#include "mega_moe_gmm1_swiglu.h"

namespace MegaMoeImpl {

// Composition-only structures keep the independent Token Dispatch and
// GMM1/SwiGLU interfaces together for the current MegaMoe template.
struct DispatchGmm1SwigluContext {
    TokenDispatchContext dispatch;
    Gmm1SwigluContext gmm1;
};

struct DispatchGmm1SwigluArgs {
    TokenDispatchArgs dispatch;
    Gmm1SwigluArgs gmm1;
};

using DispatchGmm1SwigluState = Gmm1SwigluState;

// Controls the current template's SendCount -> Dispatch -> GMM1 -> SwiGLU pipeline.
// Every physical core in one MIX_AIC_1_2 block receives the same logical block job;
// the stage modules retain only the synchronization required by their own contract.
template <typename QuantOutType, typename ActivationType, typename WeightType, typename SwigluOutType,
          typename QuantScaleType, bool EnableA8W4, uint32_t PipelineTileM, uint32_t EpilogueTileM,
          bool TopkWeightsPrefetch, typename BlockEpilogue>
__aicore__ inline void ControlDispatchGmm1SwigluPipeline(
    const DispatchGmm1SwigluContext &context, const DispatchGmm1SwigluArgs &args, const Params &gmmParams,
    BlockEpilogue &epilogueOp, TokenDispatchScratch<ActivationType> &dispatchScratch,
    Gmm1SwigluScratch &gmm1Scratch, DispatchGmm1SwigluState &runtimeState)
{
    const TokenDispatchContext &dispatchContext = context.dispatch;
    const Gmm1SwigluContext &gmm1Context = context.gmm1;
    if (dispatchContext.blockJob.totalJobs == 0U ||
        dispatchContext.blockJob.jobIndex >= dispatchContext.blockJob.totalJobs ||
        dispatchContext.countWorkspace.blockNum == 0U ||
        dispatchContext.countWorkspace.blockIdx >= dispatchContext.countWorkspace.blockNum ||
        gmm1Context.expertPerRank == 0U || dispatchContext.blockNumPerRank == 0U) {
        return;
    }

    InitTokenDispatchBuffers<ActivationType, EnableA8W4>(dispatchContext, args.dispatch, dispatchScratch);
    gmm1Scratch.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(args.gmm1.expertRevTokenNumsPtr));
    Gmm1ExpertLoopState gmm1State = CreateGmm1ExpertLoopState(gmm1Context);
    GMMAddrInfo gmm1AddrInfo;
    uint64_t currentSendCnt = 0;
    uint64_t nextSendCnt = 0;

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1) {
            ComputeExpertTokenCountAndNotify<ActivationType, EnableA8W4, TopkWeightsPrefetch>(
                dispatchContext, args.dispatch, dispatchScratch, 0, nextSendCnt);
            if (nextSendCnt != 0) {
                DispatchExpertTokens<ActivationType, QuantScaleType, EnableA8W4, PipelineTileM, TopkWeightsPrefetch>(
                    dispatchContext, args.dispatch, dispatchScratch, 0);
            }
        }
    }

    for (uint32_t expertIdx = 0; expertIdx < gmm1Context.expertPerRank; ++expertIdx) {
        if constexpr (g_coreType == AIV) {
            if (GetSubBlockIdx() == 1) {
                currentSendCnt = nextSendCnt;
                if (expertIdx + 1 < gmm1Context.expertPerRank) {
                    ComputeExpertTokenCountAndNotify<ActivationType, EnableA8W4, TopkWeightsPrefetch>(
                        dispatchContext, args.dispatch, dispatchScratch, expertIdx + 1, nextSendCnt);
                    if (nextSendCnt != 0) {
                        DispatchExpertTokens<ActivationType, QuantScaleType, EnableA8W4, PipelineTileM,
                                             TopkWeightsPrefetch>(
                            dispatchContext, args.dispatch, dispatchScratch, expertIdx + 1);
                    }
                }
            }
        }

        if (!WaitAndUpdateGmm1GroupParams<EnableA8W4>(
                gmm1Context, args.gmm1, gmm1Scratch, gmm1State, expertIdx, currentSendCnt)) {
            continue;
        }
        UpdateGmm1GlobalBuffer<ActivationType, WeightType, SwigluOutType, QuantScaleType, EnableA8W4,
                               TopkWeightsPrefetch>(
            gmm1Context, args.gmm1, epilogueOp, gmm1AddrInfo, gmm1State, expertIdx);
        ControlGmm1SwigluPipelineByMode<QuantOutType, WeightType, SwigluOutType, QuantScaleType, EnableA8W4,
                                        PipelineTileM, EpilogueTileM, TopkWeightsPrefetch>(
            gmm1Context, gmmParams, epilogueOp, gmm1AddrInfo, gmm1State, runtimeState, expertIdx);
    }

    if constexpr (TopkWeightsPrefetch) {
        int32_t allDoneTag = static_cast<int32_t>(gmm1Context.expertCapacityPerRank + 1U);
        __gm__ int32_t *allDoneAddr =
            reinterpret_cast<__gm__ int32_t *>(args.gmm1.gmm1TileStatusPtr) +
            static_cast<uint64_t>(gmm1Context.expertCapacityPerRank) * gmm1Context.maxTilesPerExpert * INT_CACHELINE;
        if constexpr (g_coreType == AIV) {
            constexpr uint32_t epilogueSubBlockIdx = EnableA8W4 ? 1U : 0U;
            if (GetSubBlockIdx() == epilogueSubBlockIdx) {
                AscendC::WriteGmByPassDCache(allDoneAddr, allDoneTag);
            }
        } else {
            while (AscendC::ReadGmByPassDCache(allDoneAddr) != allDoneTag) {
                int64_t startCycle = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - startCycle < 100) {
                }
            }
        }
    } else {
        EndSync(runtimeState.vecSetSyncCom);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_DISPATCH_GMM1_SWIGLU_H
