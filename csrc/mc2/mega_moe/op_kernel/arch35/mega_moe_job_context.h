/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_JOB_CONTEXT_H
#define MEGA_MOE_JOB_CONTEXT_H

#include "kernel_operator.h"
#include "mega_moe_base.h"

namespace MegaMoeImpl {

// AIV logical job information supplied by the caller. jobIndex must be a
// zero-based, contiguous index in [0, totalJobs), independent of physical AIV IDs.
struct AivJobContext {
    uint32_t jobIndex;
    uint32_t totalJobs;
};

struct MoeTopologyContext {
    uint32_t rankId;
    uint32_t worldSize;
    uint32_t expertPerRank;
};

struct TokenShapeContext {
    int32_t tokenNum;
    uint32_t topK;
    uint32_t tokenHiddenDim;
};

// Stable information shared by the dispatch preparation stage.
struct DispatchPrepareContext {
    AivJobContext job;
    MoeTopologyContext topology;
    TokenShapeContext tokenShape;
};

__aicore__ inline void TilingByJobContext(int32_t totalLen, int32_t &jobLen, int32_t &jobOffset, uint32_t jobIndex,
                                          uint32_t totalJobs, int32_t align = ALIGN_32)
{
    if (totalJobs == 0U || jobIndex >= totalJobs) {
        jobLen = 0;
        jobOffset = 0;
        return;
    }
    int32_t lenPerJob = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), totalJobs);
    int32_t lenPerJobAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerJob), static_cast<uint32_t>(align));
    jobLen = lenPerJobAlign;
    jobOffset = static_cast<int32_t>(jobIndex) * lenPerJobAlign;
    if (jobOffset >= totalLen) {
        jobLen = 0;
        return;
    }
    if (jobOffset + jobLen > totalLen) {
        jobLen = totalLen - jobOffset;
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_JOB_CONTEXT_H
