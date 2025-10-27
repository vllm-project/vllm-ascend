/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dropmask.h
 * \brief
 */

#ifndef DROPMASK_H
#define DROPMASK_H

#include "util.h"

using AscendC::DROPOUT_MODE_BIT_MISALIGN;
using AscendC::DropOutShapeInfo;
using AscendC::DropOut;

struct DropMaskInfo {
    // for compute dropout mask offset
    // 参数按B N G S1 S2全部切分设置进行偏移计算，没有切分的轴对应的参数设置为合适的0或者原始值
    int64_t n2G; // n2 * g
    int64_t gSize; // g
    int64_t s1Size; // s1
    int64_t s2Size; // s2
    int64_t gOutIdx; // g out index
    int64_t bSSOffset; // boidx * s1 * s2 ===bSSOffset
    int64_t n2OutIdx; // n out index
    int64_t s1OutIdx; // s1 out index   ===s1oIdx
    int64_t s1InnerIdx; // s1 inner index, 配比 ===loopIdx
    int64_t s1BaseSize; // S1基本块大小
    int64_t splitS1BaseSize; // s1 split size ===vec1S1BaseSize
    int64_t s2StartIdx; // s2 start index
    int64_t s2Idx; // s2 index =====s2LoopCount
    int64_t s2BaseNratioSize; // s2的配比长度: s2BaseSize(S2基本块大小) * nRatio

    // for copy in dropout mask
    uint32_t s1CopySize;
    uint32_t s2CopySize;
    int64_t s2TotalSize;

    // for compute dropout mask
    uint32_t firstAxis;
    uint32_t lstAxis;
    uint32_t maskLstAxis;
    int64_t vecCoreOffset = 0;
    float keepProb;

    bool boolMode;
};

template <bool hasDrop>
__aicore__ inline int64_t ComputeDropOffset(DropMaskInfo &dropMaskInfo)
{
    if constexpr (hasDrop == true) {
        // boidx * n2 * g* s1 * s2
        int64_t bOffset = dropMaskInfo.bSSOffset * dropMaskInfo.n2G;
        // n2oIdx * g * s1 *s2
        int64_t n2Offset = dropMaskInfo.n2OutIdx * dropMaskInfo.gSize * dropMaskInfo.s1Size * dropMaskInfo.s2Size;
        // goIdx * s1 * s2
        int64_t gOffset = dropMaskInfo.gOutIdx * dropMaskInfo.s1Size * dropMaskInfo.s2Size;
        // s1oIdx * s1BaseSize * s2Size + s1innerindex * vec1S1BaseSize * s2Size
        int64_t s1Offset = (dropMaskInfo.s1OutIdx * dropMaskInfo.s1BaseSize + dropMaskInfo.vecCoreOffset +
                            dropMaskInfo.s1InnerIdx * dropMaskInfo.splitS1BaseSize) * dropMaskInfo.s2Size;
        // s2StartIdx + s2index * s2BaseNratioSize
        int64_t s2Offset = dropMaskInfo.s2StartIdx + dropMaskInfo.s2Idx * dropMaskInfo.s2BaseNratioSize;
        return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
    } else {
        return 0;
    }
}

template <bool hasDrop>
__aicore__ inline void CopyInDropMask(LocalTensor<uint8_t>&dstTensor, GlobalTensor<uint8_t>& srcBoolTensor,
    GlobalTensor<uint8_t>& srcByteTensor, DropMaskInfo &dropMaskInfo, int64_t alignedSize = blockBytes)
{
    if constexpr (hasDrop == true) {
        int64_t dropMaskOffset = ComputeDropOffset<hasDrop>(dropMaskInfo);
        if (unlikely(dropMaskInfo.boolMode)) {
            BoolCopyIn(dstTensor, srcBoolTensor, dropMaskOffset,
                       dropMaskInfo.s1CopySize, dropMaskInfo.s2CopySize, dropMaskInfo.s2TotalSize, alignedSize);
        } else {
            Bit2Int8CopyIn(dstTensor, srcByteTensor, dropMaskOffset, 1,
                           dropMaskInfo.s1CopySize, dropMaskInfo.s2CopySize, dropMaskInfo.s2TotalSize, alignedSize);
        }
        return;
    }
}

template <typename T, bool hasDrop>
__aicore__ inline void ComputeDropMask(LocalTensor<T>& dstTensor, LocalTensor<T>& srcTensor,
    LocalTensor<uint8_t>& dropoutBuffer, LocalTensor<uint8_t>& tmpDropBuffer, DropMaskInfo &dropMaskInfo)
{
    if constexpr (hasDrop == true) {
        DropOutShapeInfo dropOutShapeInfo;
        dropOutShapeInfo.firstAxis = dropMaskInfo.firstAxis;
        dropOutShapeInfo.srcLastAxis = dropMaskInfo.lstAxis;

        if (unlikely(dropMaskInfo.boolMode)) {
            dropOutShapeInfo.maskLastAxis = CeilDiv(dropMaskInfo.maskLstAxis, blockBytes) * blockBytes;
            DropOut(dstTensor, srcTensor, dropoutBuffer, tmpDropBuffer, dropMaskInfo.keepProb, dropOutShapeInfo);
        } else {
            dropOutShapeInfo.maskLastAxis = CeilDiv(dropMaskInfo.maskLstAxis / byteBitRatio, blockBytes) * blockBytes;
            if (likely(dropMaskInfo.lstAxis / byteBitRatio % blockBytes == 0)) {
                DropOut(dstTensor, srcTensor, dropoutBuffer, tmpDropBuffer, dropMaskInfo.keepProb, dropOutShapeInfo);
            } else {
                DropOut<T, false, DROPOUT_MODE_BIT_MISALIGN>(dstTensor, srcTensor, dropoutBuffer, tmpDropBuffer,
                                                             dropMaskInfo.keepProb, dropOutShapeInfo);
            }
        }
        return;
    }
}

#endif // DROPMASK_H
