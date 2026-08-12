/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SHARED_EXPERT_PREPARE_H
#define MEGA_MOE_SHARED_EXPERT_PREPARE_H

#include "../mega_moe_job_context.h"

namespace MegaMoeImpl {

struct SharedExpertPrepareArgs {
    GM_ADDR quantTokenScalePtr;
    GM_ADDR sharedExpertInputDataPtr;
    GM_ADDR sharedExpertInputScalePtr;
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    uint32_t activationElementsPerByte;
};

template <typename ActivationType>
struct SharedExpertPrepareScratch {
    LocalTensor<ActivationType> copyBuffer0;
    LocalTensor<ActivationType> copyBuffer1;
};

// Prototype: MegaMoe::SharedExpertCopyInput. Splits interleaved token data and scale for one logical AIV job.
template <typename ActivationType, typename QuantScaleType>
__aicore__ inline void PrepareSharedExpertInput(const DispatchPrepareContext &context,
                                                const SharedExpertPrepareArgs &args,
                                                SharedExpertPrepareScratch<ActivationType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivJobContext &job = context.job;
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs) {
        return;
    }

    int32_t jobTokenNum;
    int32_t jobTokenOffset;
    TilingByJobContext(context.tokenShape.tokenNum, jobTokenNum, jobTokenOffset, job.jobIndex, job.totalJobs, 1);

    int64_t tokenDataElementCount = context.tokenShape.tokenHiddenDim / args.activationElementsPerByte;
    int64_t tokenScaleElementCount = Ops::Base::CeilDiv(static_cast<int64_t>(context.tokenShape.tokenHiddenDim),
                                                        static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                                     MXFP_MULTI_BASE_SIZE;
    uint32_t copyTokenScaleBytes = args.quantTokenAlignBytes + args.quantScaleAlignBytes;

    GlobalTensor<ActivationType> srcGlobalTensor;
    GlobalTensor<ActivationType> dataDstGlobalTensor;
    GlobalTensor<QuantScaleType> scaleDstGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(args.quantTokenScalePtr));
    dataDstGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(args.sharedExpertInputDataPtr));
    scaleDstGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleType *>(args.sharedExpertInputScalePtr));

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < jobTokenNum; ++index) {
        int32_t tokenIdx = jobTokenOffset + index;
        uint64_t srcOffset = static_cast<uint64_t>(tokenIdx) * static_cast<uint64_t>(args.quantTokenScaleAlignBytes);
        bool useFirstBuffer = index % DOUBLE_BUFFER == 0;
        auto event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        auto copyBuffer = useFirstBuffer ? scratch.copyBuffer0 : scratch.copyBuffer1;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopy(copyBuffer, srcGlobalTensor[srcOffset], copyTokenScaleBytes);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);

        LocalTensor<QuantScaleType> scaleBuffer =
            copyBuffer[args.quantTokenAlignBytes].template ReinterpretCast<QuantScaleType>();
        DataCopyPad(dataDstGlobalTensor[static_cast<int64_t>(tokenIdx) * tokenDataElementCount], copyBuffer,
                    {1, static_cast<uint16_t>(tokenDataElementCount * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleDstGlobalTensor[static_cast<int64_t>(tokenIdx) * tokenScaleElementCount], scaleBuffer,
                    {1, static_cast<uint16_t>(tokenScaleElementCount * sizeof(QuantScaleType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_SHARED_EXPERT_PREPARE_H
