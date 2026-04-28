/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fill_utils.h
 * \brief
 */

#ifndef UTILS_FILL_UTILS_H
#define UTILS_FILL_UTILS_H
#include "common_utils.h"
#include "kernel_operator.h"

namespace Act {
namespace Gemm {

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 3102)
template <typename T>
__aicore__ inline void InitOutputWithZero(AscendC::GlobalTensor<T> yInitGlobal, AscendC::LocalTensor<T>& initLocal,
                                          uint64_t ySize, int32_t usedCoreNum, bool& isKZeroInit)
{
    if ASCEND_IS_AIC {
        return;
    }
    // Shared bandwidth transfer: Only need to move aiv0's 0 to GM, also support aic:aiv=1:1 cases
    if (AscendC::GetSubBlockIdx() >= 1) {
        return;
    }
    uint32_t blockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    // Fetch values following the InitOutput interface pattern
    // Maximum number of elements storable in output dtype
    uint64_t initSize = (AscendC::MAX_REPEAT_TIMES * AscendC::ONE_BLK_SIZE) / sizeof(T);
    uint64_t perCoreSize = CeilDiv(ySize, usedCoreNum);
    perCoreSize = Align(perCoreSize * sizeof(T), AscendC::ONE_BLK_SIZE) / sizeof(T);
    initSize = Min(initSize, perCoreSize);
    uint64_t realCoreNum = Min(CeilDiv(ySize, initSize), static_cast<uint64_t>(usedCoreNum));
    if (blockIdx >= realCoreNum) { // Return excess cores, minimum 32B alignment per core
        return;
    }
    if (!isKZeroInit) { // Initialize all buffers in ub to zero when k==0 (first iter)
        AscendC::Duplicate<T>(initLocal, 0, initSize);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        isKZeroInit = true;
    }
    uint64_t yOffset = perCoreSize * blockIdx;
    uint64_t outCurSize = (blockIdx == realCoreNum - 1) ? (ySize - yOffset) : perCoreSize;
    uint64_t movRound = outCurSize / initSize;
    uint64_t movTail = outCurSize - movRound * initSize;

    AscendC::DataCopyExtParams ub2GmParams{1, static_cast<uint32_t>(initSize * sizeof(T)), 0, 0, 0};
    for (uint64_t i = 0; i < movRound; ++i) {
        AscendC::DataCopyPad(yInitGlobal[yOffset], initLocal, ub2GmParams);
        yOffset += initSize;
    }
    if (movTail != 0) { // mov tail zero data
        ub2GmParams.blockLen = static_cast<uint32_t>(movTail * sizeof(T));
        AscendC::DataCopyPad(yInitGlobal[yOffset], initLocal, ub2GmParams);
    }
}
#endif
} // namespace Gemm
} // namespace Act
#endif