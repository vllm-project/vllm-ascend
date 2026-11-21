/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GMM_SWIGLU_OAI_MX_QUANT_UTILS_H
#define GMM_SWIGLU_OAI_MX_QUANT_UTILS_H

#include "adv_api/utils/init_global_memory.h"

namespace GroupedMatmulDequantSwigluQuant {
using namespace AscendC;

constexpr int64_t SWIGLU_REDUCE_FACTOR = 2;

// V3 permits group_list to cover fewer than M rows. Those rows have no
// epilogue invocation, so preserve their zero scale codes inside this kernel
// rather than launching a separate Torch InplaceZero for the entire output.
__aicore__ inline void ClearUnprocessedMxScaleRows(GM_ADDR groupList, GM_ADDR yScale, int64_t m, int64_t outputN,
                                                   uint32_t groupNum, uint8_t groupListType, uint32_t swigluMode)
{
    if ASCEND_IS_AIC {
        return;
    }
    constexpr uint32_t splitClampSwigluMode = 2;
    if (swigluMode != splitClampSwigluMode || GetBlockIdx() != 0 || GetSubBlockIdx() != 0 || groupNum == 0 || m <= 0 ||
        outputN <= 0) {
        return;
    }
    GlobalTensor<int64_t> groups;
    groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupList));
    int64_t processedRows = 0;
    if (groupListType == 0) {
        processedRows = groups.GetValue(groupNum - 1);
    } else {
        for (uint32_t group = 0; group < groupNum; ++group) {
            const int64_t rows = groups.GetValue(group);
            if (rows < 0 || rows > m - processedRows) {
                return;
            }
            processedRows += rows;
        }
    }
    if (processedRows < 0 || processedRows >= m) {
        return;
    }
    // Each public row contains ceil(outputN / 64) pairs of one-byte E8M0
    // codes. Use uint16_t so every row starts at a naturally aligned pair.
    constexpr int64_t mxGroupSize = 64;
    const uint64_t pairsPerRow = static_cast<uint64_t>((outputN + mxGroupSize - 1) / mxGroupSize);
    GlobalTensor<uint16_t> tailScale;
    tailScale.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(yScale) + processedRows * pairsPerRow);
    InitGlobalMemory(tailScale, static_cast<uint64_t>(m - processedRows) * pairsPerRow, static_cast<uint16_t>(0));
}

template <typename T>
__aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;

    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}
} // namespace GroupedMatmulDequantSwigluQuant

#endif
