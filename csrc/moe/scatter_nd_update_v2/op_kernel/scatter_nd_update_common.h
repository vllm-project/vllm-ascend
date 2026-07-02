/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_update_common.h
 * \brief ScatterNdUpdateV2 公共定义和工具函数
 */

#ifndef SCATTER_ND_UPDATE_V2_COMMON_H
#define SCATTER_ND_UPDATE_V2_COMMON_H

#include "kernel_operator.h"

namespace ScatterNdUpdateV2 {
using namespace AscendC;

// 公共常量定义
constexpr uint64_t DOUBLE_BUFFER = 1;
constexpr uint64_t SORT_RES_NUM = 2;
constexpr uint64_t SORT_TMP_NUM = 3;
constexpr uint64_t ALIGNED_BLOCK_NUM = 32;
constexpr uint64_t ALIGN_NUM = 8;  // 32 字节对齐 = 8 个 int32
constexpr uint64_t ALIGNED_SIZE = 512;

// 公共同步函数
__aicore__ inline void PipeMte2ToS()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventID);
    WaitFlag<HardEvent::MTE2_S>(eventID);
}

__aicore__ inline void PipeMte3ToS()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventID);
    WaitFlag<HardEvent::MTE3_S>(eventID);
}

__aicore__ inline void PipeVToMte3()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
}

// 计算 block 分布参数
__aicore__ inline void CalcBlockDistribution(
    uint64_t blockIdx, uint64_t frontNum, uint64_t frontRow, uint64_t tailRow,
    uint64_t& computeRow, uint64_t& start)
{
    if (blockIdx >= frontNum) {
        computeRow = tailRow;
        start = frontNum * frontRow + (blockIdx - frontNum) * computeRow;
    } else {
        computeRow = frontRow;
        start = blockIdx * computeRow;
    }
}

} // namespace ScatterNdUpdateV2

#endif // SCATTER_ND_UPDATE_V2_COMMON_H
