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
 * \brief ScatterNdUpdate 公共定义和工具函数
 */

#ifndef SCATTER_ND_UPDATE_COMMON_H
#define SCATTER_ND_UPDATE_COMMON_H

#include "kernel_operator.h"

namespace ScatterNdUpdate {
using namespace AscendC;

// 公共常量定义
constexpr uint64_t DOUBLE_BUFFER = 1;
constexpr uint64_t SORT_RES_NUM = 2;
constexpr uint64_t SORT_TMP_NUM = 3;
constexpr uint64_t ALIGNED_BLOCK_NUM = 32;
constexpr uint64_t ALIGN_NUM = 8; // 32 字节对齐 = 8 个 int32
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

__aicore__ inline void PipeVToS()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID);
    WaitFlag<HardEvent::V_S>(eventID);
}

__aicore__ inline uint64_t ComputeViewedRowOffset(uint64_t linearIndex, uint64_t firstDimStrideRows,
                                                  uint64_t varStride0Elements, uint64_t scatterLength)
{
    if (firstDimStrideRows == 0) {
        return linearIndex * scatterLength;
    }
    uint64_t i0 = linearIndex / firstDimStrideRows;
    uint64_t rest = linearIndex - i0 * firstDimStrideRows;
    return i0 * varStride0Elements + rest * scatterLength;
}

// 统一处理 view-stride0 与连续两条路径的输出偏移计算。
template <bool isViewStride0>
__aicore__ inline uint64_t ResolveOutOffset(uint64_t linearIndex, uint64_t scatterLength, uint64_t firstDimStrideRows,
                                            uint64_t varStride0Elements, uint64_t tileOffsetElements)
{
    if constexpr (isViewStride0) {
        return ComputeViewedRowOffset(linearIndex, firstDimStrideRows, varStride0Elements, scatterLength) +
               tileOffsetElements;
    } else {
        return linearIndex * scatterLength + tileOffsetElements;
    }
}

// 计算 block 分布参数
__aicore__ inline void CalcBlockDistribution(uint64_t blockIdx, uint64_t frontNum, uint64_t frontRow, uint64_t tailRow,
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

__aicore__ inline void ComputeLinearIndexFromIndices(LocalTensor<int>& indicesLocal,
                                                     LocalTensor<int>& indicesOriginLocal,
                                                     LocalTensor<int>& addTmpLocal, LocalTensor<int>& rangeLocal,
                                                     const uint64_t* indicesMask, uint64_t indexDim, uint64_t rows)
{
    int32_t mulValue = static_cast<int32_t>(indexDim * sizeof(int));
    Duplicate<int>(indicesLocal, 0, rows);
    CreateVecIndex(rangeLocal, (int)0, rows);
    PipeBarrier<PIPE_V>();
    Muls(rangeLocal, rangeLocal, mulValue, rows);
    PipeBarrier<PIPE_V>();
    for (uint64_t i = 0; i < indexDim; ++i) {
        if (i != 0) {
            Adds(rangeLocal, rangeLocal, (int)(sizeof(int)), rows);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<uint32_t> rangeCasted = rangeLocal.ReinterpretCast<uint32_t>();
        Gather(addTmpLocal, indicesOriginLocal, rangeCasted, (uint32_t)0, (uint32_t)rows);
        PipeBarrier<PIPE_V>();
        Muls(addTmpLocal, addTmpLocal, (int)indicesMask[i], rows);
        PipeBarrier<PIPE_V>();
        Add(indicesLocal, indicesLocal, addTmpLocal, rows);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, bool isViewStride0>
__aicore__ inline void DoScatterCopy(LocalTensor<T>& updateLocal, GlobalTensor<T>& updatesGm, GlobalTensor<T>& outputGm,
                                     uint64_t gmOffset, uint64_t tileLength, int64_t linearIndex, uint64_t tileIdx,
                                     uint64_t scatterTileLength, uint64_t firstDimStrideRows,
                                     uint64_t varStride0Elements, uint64_t scatterLength)
{
    DataCopyExtParams updateCopyParams{1, static_cast<uint32_t>(tileLength * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(updateLocal, updatesGm[gmOffset], updateCopyParams, padParams);
    PipeMte2ToS();

    uint64_t outOffset = ResolveOutOffset<isViewStride0>(static_cast<uint64_t>(linearIndex), scatterLength,
                                                         firstDimStrideRows, varStride0Elements,
                                                         tileIdx * scatterTileLength);
    DataCopyExtParams outParams{1, static_cast<uint32_t>(tileLength * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm[outOffset], updateLocal, outParams);
    PipeMte3ToS();
}

} // namespace ScatterNdUpdate

#endif // SCATTER_ND_UPDATE_COMMON_H
