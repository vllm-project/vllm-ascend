/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tile/x3_copy_utils.h
 * \brief Utilities for copying the x3 tensor used by fused add/mul epilogues.
 */

#pragma once

namespace Cmct {
namespace Gemm {
namespace Block {
namespace Detail {
static constexpr int32_t FIRST_DIM = 0;
static constexpr int32_t SECOND_DIM = 1;
static constexpr int32_t THIRD_DIM = 2;
static constexpr uint16_t ZERO_FLAG = 0;

template <typename DataTypeIn, typename DataTypeOut, typename LocalTensor>
__aicore__ inline bool CopyFusionX3NdDma(LocalTensor inputLocal, AscendC::GlobalTensor<DataTypeIn>& inputGlobal,
                                         int64_t curAivM, int64_t curAivNAlign, int64_t strideN, int64_t stageSize)
{
    if (curAivNAlign <= 0 || curAivM <= 0 || strideN <= 0 || stageSize <= 0) {
        return false;
    }
    int64_t curBatch = stageSize / curAivNAlign / curAivM;
    AscendC::MultiCopyParams<DataTypeIn, DIM_SIZE_THREE> ndDmaParams;
    ndDmaParams.loopInfo.loopSrcStride[FIRST_DIM] = 1;
    ndDmaParams.loopInfo.loopSrcStride[SECOND_DIM] = static_cast<uint32_t>(strideN);
    ndDmaParams.loopInfo.loopSrcStride[THIRD_DIM] = 0;
    ndDmaParams.loopInfo.loopDstStride[FIRST_DIM] = 1;
    ndDmaParams.loopInfo.loopDstStride[SECOND_DIM] = static_cast<uint32_t>(curAivNAlign);
    ndDmaParams.loopInfo.loopDstStride[THIRD_DIM] = static_cast<uint32_t>(curAivNAlign * curAivM);
    ndDmaParams.loopInfo.loopSize[FIRST_DIM] = static_cast<uint32_t>(strideN);
    ndDmaParams.loopInfo.loopSize[SECOND_DIM] = static_cast<uint32_t>(curAivM);
    ndDmaParams.loopInfo.loopSize[THIRD_DIM] = static_cast<uint32_t>(curBatch);
    ndDmaParams.loopInfo.loopLpSize[FIRST_DIM] = 0;
    ndDmaParams.loopInfo.loopLpSize[SECOND_DIM] = 0;
    ndDmaParams.loopInfo.loopLpSize[THIRD_DIM] = 0;
    ndDmaParams.loopInfo.loopRpSize[FIRST_DIM] = static_cast<uint8_t>(AscendC::Std::max(curAivNAlign - strideN, 0L));
    ndDmaParams.loopInfo.loopRpSize[SECOND_DIM] = 0;
    ndDmaParams.loopInfo.loopRpSize[THIRD_DIM] = 0;
    AscendC::DataCopy(inputLocal, inputGlobal, ndDmaParams);
    return true;
}

template <typename DataTypeIn, typename LocalTensor>
__aicore__ inline void CopyFusionX3Broadcast(LocalTensor inputLocal, AscendC::GlobalTensor<DataTypeIn>& inputGlobal,
                                             int64_t offset, int64_t strideN, int64_t curAivNAlign, int64_t stageSize,
                                             int64_t x3M, uint32_t blockLen, uint32_t srcStride, uint32_t dstStride,
                                             const AscendC::DataCopyPadExtParams<DataTypeIn>& padParams)
{
    if (curAivNAlign <= 0 || strideN <= 0 || x3M <= 0 || stageSize <= 0) {
        return;
    }
    int64_t rowCount = stageSize / curAivNAlign;
    int64_t colOffset = offset % strideN;
    int64_t rowOffset = offset / strideN;
    for (int64_t row = 0; row < rowCount;) {
        int64_t x3Row = (rowOffset + row) % x3M;
        int64_t segRows = AscendC::Std::min(rowCount - row, x3M - x3Row);
        int64_t x3Offset = x3Row * strideN + colOffset;
        AscendC::DataCopyExtParams segmentCopyParams{static_cast<uint16_t>(segRows), blockLen, srcStride, dstStride, 0};
        AscendC::DataCopyPad(inputLocal[row * curAivNAlign], inputGlobal[x3Offset], segmentCopyParams, padParams);
        row += segRows;
    }
}

template <typename DataTypeIn, typename DataTypeOut, typename LocalTensor>
__aicore__ inline bool CopyFusionX3Aligned(LocalTensor inputLocal, AscendC::GlobalTensor<DataTypeIn>& inputGlobal,
                                           int64_t offset, int64_t curAivM, int64_t curAivN, int64_t strideN,
                                           int64_t stageSize, bool needNdDma, int64_t curAivNAlign, uint32_t dstStride,
                                           bool x3BatchBroadcast, int64_t x3M, bool enablePad,
                                           bool enableMte3Mte2Sync = true)
{
    if (curAivNAlign <= 0 || curAivN <= 0 || strideN <= 0 || stageSize <= 0) {
        return false;
    }
    if (needNdDma) {
        return CopyFusionX3NdDma<DataTypeIn, DataTypeOut>(inputLocal, inputGlobal, curAivM, curAivNAlign, strideN,
                                                          stageSize);
    }

    if (enableMte3Mte2Sync) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
    }
    uint16_t blockCount = static_cast<uint16_t>(stageSize / curAivNAlign);
    uint32_t blockLen = static_cast<uint32_t>(curAivN * sizeof(DataTypeOut));
    uint32_t srcStride = static_cast<uint32_t>((strideN - curAivN) * sizeof(DataTypeIn));
    AscendC::DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
    AscendC::DataCopyPadExtParams<DataTypeIn> padParams{enablePad, 0, 0, 0};
    if (!x3BatchBroadcast || x3M <= 0) {
        AscendC::DataCopyPad(inputLocal, inputGlobal[offset], copyParams, padParams);
    } else {
        CopyFusionX3Broadcast(inputLocal, inputGlobal, offset, strideN, curAivNAlign, stageSize, x3M, blockLen,
                              srcStride, dstStride, padParams);
    }
    return true;
}

template <typename DataTypeIn, typename DataTypeOut, typename LocalTensor>
__aicore__ inline bool CopyFusionX3(LocalTensor inputLocal, AscendC::GlobalTensor<DataTypeIn>& inputGlobal,
                                    int64_t offset, int64_t curAivM, int64_t curAivN, int64_t strideN,
                                    int64_t stageSize, bool needNdDma, bool fixp1v2, bool x3BatchBroadcast, int64_t x3M,
                                    bool enablePad)
{
    int64_t curAivNAlign = fixp1v2 ? CeilAlign(curAivN, AscendC::BLOCK_CUBE) : AlignBlock<DataTypeIn>(curAivN);
    uint32_t dstStride = fixp1v2 ?
                             static_cast<uint32_t>((curAivNAlign - curAivN) * sizeof(DataTypeOut) / UB_ALIGN_SIZE) :
                             0;
    return CopyFusionX3Aligned<DataTypeIn, DataTypeOut>(inputLocal, inputGlobal, offset, curAivM, curAivN, strideN,
                                                        stageSize, needNdDma, curAivNAlign, dstStride, x3BatchBroadcast,
                                                        x3M, enablePad);
}
} // namespace Detail
} // namespace Block
} // namespace Gemm
} // namespace Cmct
