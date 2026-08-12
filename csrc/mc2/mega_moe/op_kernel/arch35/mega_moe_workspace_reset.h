/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_WORKSPACE_RESET_H
#define MEGA_MOE_WORKSPACE_RESET_H

#include "mega_moe_job_context.h"

namespace MegaMoeImpl {

struct ResetWorkspaceArgs {
    GM_ADDR flagSwiGluToGmm2Ptr;
    GM_ADDR gmm2CombineSyncCounterPtr;
    GM_ADDR sharedExpertGmm2TileCounterPtr;
    GM_ADDR gmm1TileStatusPtr;
    int32_t flagNum;
    int32_t gmm2CombineSyncCounterNum;
    int32_t sharedExpertGmm2TileCounterNum;
    int32_t gmm1StatusExpertCapacity;
    int32_t maxTilesPerExpert;
    int32_t resetBatchElementCount;
};

// Prototype: MegaMoe::ResetGmm2CombineSyncCounters. Clears one logical AIV job's combine counters in batches.
// ResetDispatchWorkspace validates the job and prepares the zero-filled resetTensor before calling this helper.
__aicore__ inline void ResetGmm2CombineSyncCounters(const AivJobContext &job, const ResetWorkspaceArgs &args,
                                                    LocalTensor<int32_t> &resetTensor)
{
    int32_t jobLen;
    int32_t jobOffset;
    TilingByJobContext(args.gmm2CombineSyncCounterNum, jobLen, jobOffset, job.jobIndex, job.totalJobs);
    GlobalTensor<int32_t> counterGm;
    counterGm.SetGlobalBuffer((__gm__ int32_t *)args.gmm2CombineSyncCounterPtr);
    for (int32_t offset = 0; offset < jobLen; offset += args.resetBatchElementCount) {
        int32_t batchCount =
            jobLen - offset < args.resetBatchElementCount ? jobLen - offset : args.resetBatchElementCount;
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(batchCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(counterGm[jobOffset + offset], resetTensor, copyParams);
    }
}

// Clears the tile counters used by shared-expert GMM2 producers and Unpermute consumers.
__aicore__ inline void ResetSharedExpertGmm2TileCounters(const AivJobContext &job, const ResetWorkspaceArgs &args,
                                                         LocalTensor<int32_t> &resetTensor)
{
    int32_t jobLen;
    int32_t jobOffset;
    TilingByJobContext(args.sharedExpertGmm2TileCounterNum, jobLen, jobOffset, job.jobIndex, job.totalJobs);
    GlobalTensor<int32_t> counterGm;
    counterGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(args.sharedExpertGmm2TileCounterPtr));
    for (int32_t offset = 0; offset < jobLen; offset += args.resetBatchElementCount) {
        int32_t batchCount =
            jobLen - offset < args.resetBatchElementCount ? jobLen - offset : args.resetBatchElementCount;
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(batchCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(counterGm[jobOffset + offset], resetTensor, copyParams);
    }
}

// Prototype: MegaMoe::ResetFlagList. Clears one logical AIV job's dispatch flags in bounded UB batches.
template <bool ResetCombineCounters, bool TopkWeightsPrefetch>
__aicore__ inline void ResetDispatchWorkspace(const DispatchPrepareContext &context, const ResetWorkspaceArgs &args,
                                              LocalTensor<int32_t> &resetTensor)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivJobContext &job = context.job;
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs || args.resetBatchElementCount <= 0) {
        return;
    }
    GlobalTensor<int32_t> flagGm;
    flagGm.SetGlobalBuffer((__gm__ int32_t *)args.flagSwiGluToGmm2Ptr);
    int32_t jobLen, jobOffset;
    TilingByJobContext(args.flagNum, jobLen, jobOffset, job.jobIndex, job.totalJobs, 1);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    for (int32_t offset = 0; offset < jobLen; offset += args.resetBatchElementCount) {
        int32_t batchCount =
            jobLen - offset < args.resetBatchElementCount ? jobLen - offset : args.resetBatchElementCount;
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(batchCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(flagGm[jobOffset + offset], resetTensor, copyParams);
    }
    if constexpr (ResetCombineCounters) {
        ResetGmm2CombineSyncCounters(job, args, resetTensor);
    }
    if (args.sharedExpertGmm2TileCounterNum > 0) {
        ResetSharedExpertGmm2TileCounters(job, args, resetTensor);
    }
    if constexpr (TopkWeightsPrefetch) {
        GlobalTensor<int32_t> statusGm;
        statusGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(args.gmm1TileStatusPtr));
        int32_t statusElementCount =
            (args.gmm1StatusExpertCapacity * args.maxTilesPerExpert + 1) * INT_CACHELINE;
        int32_t statusJobLen;
        int32_t statusJobOffset;
        TilingByJobContext(statusElementCount, statusJobLen, statusJobOffset, job.jobIndex, job.totalJobs, 1);
        for (int32_t offset = 0; offset < statusJobLen; offset += args.resetBatchElementCount) {
            int32_t batchCount = statusJobLen - offset < args.resetBatchElementCount ? statusJobLen - offset :
                                                                                       args.resetBatchElementCount;
            DataCopyExtParams copyParams{1U, static_cast<uint32_t>(batchCount * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPad(statusGm[statusJobOffset + offset], resetTensor, copyParams);
        }
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_WORKSPACE_RESET_H
