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
 * \file util.h
 * \brief
 */

#ifndef FLASH_ATTENTION_UTIL_H
#define FLASH_ATTENTION_UTIL_H

constexpr int32_t blockBytes = 32;
constexpr int32_t byteBitRatio = 8;
constexpr int64_t prefixAttenMaskDownHeight = 1024;
constexpr static int32_t blockSize = blockBytes / 4; // 4 means sizeof(T)
constexpr static int32_t repeatMaxBytes = 256;
constexpr static int32_t repeatMaxTimes = 255;
constexpr static int32_t repeatMaxSize = repeatMaxBytes / 4; // 4 means sizeof(T)

using AscendC::LocalTensor;
using AscendC::GlobalTensor;
using AscendC::DataFormat;
using AscendC::ShapeInfo;
using AscendC::DataCopyParams;
using AscendC::DataCopyPadParams;
using AscendC::BinaryRepeatParams;
using AscendC::IsSameType;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::WaitFlag;

enum class LayOutTypeEnum { None = 0, LAYOUT_BSH = 1, LAYOUT_SBH = 2, LAYOUT_BNSD = 3, LAYOUT_TND = 4, LAYOUT_NTD_TND = 5};

namespace math {
template <typename T> __aicore__ inline T Ceil(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T> __aicore__ inline T Align(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

__aicore__ inline void BoolCopyIn(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
    int64_t srcOffset, uint32_t s1Size, uint32_t s2Size, int64_t totalS2Size, int64_t alignedSize = blockBytes)
{
    uint32_t alignedS2Size = CeilDiv(s2Size, alignedSize) * alignedSize;
    uint32_t shapeArray[] = {s1Size, alignedS2Size};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dstTensor.SetSize(s1Size * alignedS2Size);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Size;
    dataCopyParams.dstStride = 0;
    if (totalS2Size == blockBytes && alignedSize == 64) { // totalS2Size < 64 && totalS2Size % blockBytes == 0
        dataCopyParams.dstStride = 1;
        alignedSize = blockBytes;
        alignedS2Size = CeilDiv(s2Size, blockBytes) * blockBytes;
    }
    if (totalS2Size % alignedSize == 0) {
        dataCopyParams.blockLen = alignedS2Size / blockBytes;
        dataCopyParams.srcStride = (totalS2Size - alignedS2Size) / blockBytes;
        DataCopy(dstTensor, srcTensor[srcOffset], dataCopyParams);
    } else {
        dataCopyParams.blockLen = s2Size;
        dataCopyParams.srcStride = totalS2Size - s2Size;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.rightPadding = Min(alignedS2Size - s2Size, blockBytes);
        dataCopyPadParams.paddingValue = 1;
        DataCopyPad(dstTensor, srcTensor[srcOffset], dataCopyParams, dataCopyPadParams);
    }
}

__aicore__ inline void Bit2Int8CopyIn(LocalTensor<uint8_t> &dstTensor, GlobalTensor<uint8_t> &srcTensor,
    int64_t srcOffset, uint32_t batchSize, uint32_t s1BaseSize, uint32_t s2BaseSize, int64_t s2TotalSize,
    int64_t alignedSize = blockBytes)
{
    uint32_t alignedS2Size = CeilDiv(s2BaseSize / byteBitRatio, alignedSize) * alignedSize;
    uint32_t shapeArray[] = {batchSize * s1BaseSize, alignedS2Size};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dstTensor.SetSize(batchSize * s1BaseSize * alignedS2Size);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = batchSize * s1BaseSize;
    dataCopyParams.blockLen = CeilDiv(s2BaseSize / byteBitRatio, blockBytes);
    dataCopyParams.dstStride = 0;
    if (s2TotalSize / byteBitRatio % alignedSize == 0 && s2BaseSize / byteBitRatio % alignedSize == 0) {
        dataCopyParams.srcStride =
            (s2TotalSize / byteBitRatio - dataCopyParams.blockLen * blockBytes) / blockBytes;
        DataCopy(dstTensor, srcTensor[srcOffset / byteBitRatio], dataCopyParams);
    } else {
        dataCopyParams.blockLen = CeilDiv(s2BaseSize , byteBitRatio);
        dataCopyParams.srcStride = (s2TotalSize  - s2BaseSize) / byteBitRatio;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.rightPadding = 0;
        dataCopyPadParams.paddingValue = 0;
        DataCopyPad(dstTensor, srcTensor[srcOffset / byteBitRatio], dataCopyParams, dataCopyPadParams);
    }
}

__aicore__ inline int32_t Align(int32_t shape)
{
    int32_t alignFactor = 16;
    int32_t alignedSize = CeilDiv<int32_t, int32_t>(shape, alignFactor) * alignFactor;
    return alignedSize;
}

#endif // FLASH_ATTENTION_UTIL_H
