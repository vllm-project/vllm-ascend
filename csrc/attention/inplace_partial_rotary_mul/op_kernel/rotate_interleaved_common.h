/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotate_interleaved_common.h
 * \brief
 */
#ifndef ROTATE_INTERLEAVED_COMMON_H
#define ROTATE_INTERLEAVED_COMMON_H
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"

namespace RotateInterleavedN {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t NUM_8 = 8;
constexpr uint8_t REPEAT_MAX = 255;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t MASK_FP16 = 128;
constexpr int32_t MASK_FP32 = 64;
constexpr int32_t ALIGN_16 = 16;
constexpr int32_t ALIGN_32 = 8;

// SD -> BSND
template <typename T>
__aicore__ inline void BroadCastTriToBsnd(LocalTensor<T> &tri, uint32_t batchSize, uint32_t calcLen, uint32_t numHeads,
                                          uint32_t headDimAlign)
{
    DataCopyParams intriParams;
    intriParams.blockCount = static_cast<uint16_t>(calcLen);
    intriParams.blockLen = static_cast<uint16_t>(headDimAlign * sizeof(T) / BLOCK_SIZE);
    intriParams.srcStride = static_cast<uint16_t>((numHeads - 1) * headDimAlign * sizeof(T) / BLOCK_SIZE);
    intriParams.dstStride = static_cast<uint16_t>((numHeads - 1) * headDimAlign * sizeof(T) / BLOCK_SIZE);
    // SD -> SND
    for (uint32_t numHeadsIdx = 1; numHeadsIdx < numHeads; ++numHeadsIdx) {
        DataCopy(tri[numHeadsIdx * headDimAlign], tri, intriParams);
    }

    intriParams.blockCount = 1;
    intriParams.blockLen = static_cast<uint16_t>(calcLen * numHeads * headDimAlign * sizeof(T) / BLOCK_SIZE);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    // SND -> BSND
    for (uint32_t batchIdx = 1; batchIdx < batchSize; ++batchIdx) {
        DataCopy(tri[batchIdx * calcLen * numHeads * headDimAlign], tri, intriParams);
    }
}

// D -> BND
template <typename T>
__aicore__ inline void BroadCastTriToB1nd(LocalTensor<T> &tri, uint32_t calcLen, uint32_t numHeads,
                                          uint32_t headDimAlign)
{
    using ElementType =
        typename std::conditional<std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value, half, T>::type;
    LocalTensor<ElementType> triNew = tri.template ReinterpretCast<ElementType>();

    const int32_t mask = (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) ? MASK_FP16 : MASK_FP32;
    const int32_t count = headDimAlign / mask;
    const int32_t remain = headDimAlign % mask;
    const int32_t repeatTimes = calcLen * numHeads - 1;
    const int32_t repeatTimesLoop = repeatTimes / REPEAT_MAX;
    const int32_t repeatTimesRemain = repeatTimes % REPEAT_MAX;

    CopyRepeatParams repeatParams;
    repeatParams.dstStride = 1;
    repeatParams.srcStride = 1;
    repeatParams.dstRepeatSize = headDimAlign * sizeof(ElementType) / BLOCK_SIZE;
    repeatParams.srcRepeatSize = 0;

    for (uint32_t loopIdx = 0; loopIdx < count; ++loopIdx) {
        for (uint32_t i = 0; i < repeatTimesLoop; ++i) {
            Copy(triNew[headDimAlign + loopIdx * mask + i * REPEAT_MAX * headDimAlign], triNew[loopIdx * mask], mask,
                 REPEAT_MAX, repeatParams);
        }
        Copy(triNew[headDimAlign * (repeatTimesLoop * REPEAT_MAX + 1) + loopIdx * mask], triNew[loopIdx * mask], mask,
             repeatTimesRemain, repeatParams);
    }
    if (remain != 0) {
        for (uint32_t i = 0; i < repeatTimesLoop; ++i) {
            Copy(triNew[headDimAlign + count * mask + i * REPEAT_MAX * headDimAlign], triNew[count * mask], remain,
                 REPEAT_MAX, repeatParams);
        }
        Copy(triNew[headDimAlign * (repeatTimesLoop * REPEAT_MAX + 1) + count * mask], triNew[count * mask], remain,
             repeatTimesRemain, repeatParams);
    }
}

// The minimum amount of data set by offset is 8
__aicore__ inline void SetGatherSrcOffset(LocalTensor<int32_t> &gatherOffset, int32_t count, int32_t srcSizeof)
{
    for (int32_t i = 0; i < NUM_8; ++i) {
        gatherOffset.SetValue(i, i ^ 1); // XOR with 1 to swap even and odd indices
    }

    int32_t scalarValue = 8;
    while (scalarValue < count) {
        int32_t nextValue = scalarValue * 2;
        if (nextValue < count) {
            Adds(gatherOffset[scalarValue], gatherOffset, scalarValue, scalarValue);
        } else {
            Adds(gatherOffset[scalarValue], gatherOffset, scalarValue, count - scalarValue);
            break;
        }
        scalarValue = nextValue;
    }
    Muls(gatherOffset, gatherOffset, srcSizeof, count);
}

// count < 256 * 64 and count % 8 == 0
__aicore__ inline void InterleavedInversion(LocalTensor<float> &srcInversion, int32_t count, bool isOffset = false)
{
    SetMaskNorm();

    const int32_t mask = MASK_FP32;
    const int32_t repeatTimes = count / mask;
    const int32_t remainder = count % mask;

    // Define masks based on the 'isOffset' flag
    const uint64_t fullMask = isOffset ? 0xAAAAAAAAAAAAAAAA : 0x5555555555555555;
    const uint64_t partialMask = isOffset ? 0xAA : 0x55;

    // Apply the mask and multiplication for the full
    SetVectorMask<float, MaskMode::NORMAL>(0, fullMask);
    Muls<float, false>(srcInversion, srcInversion, float(-1), MASK_PLACEHOLDER, repeatTimes, {1, 1, 8, 8});

    // Apply the mask and multiplication for the remainder if needed
    if (remainder) {
        SetVectorMask<float, MaskMode::NORMAL>(0, partialMask);
        Muls<float, false>(srcInversion[repeatTimes * MASK_FP32], srcInversion[repeatTimes * MASK_FP32], float(-1),
                           MASK_PLACEHOLDER, count % MASK_FP32 / NUM_8, {1, 1, 1, 1});
    }
    ResetMask();
}

} // namespace RotateInterleavedN

#endif // ROTATE_INTERLEAVED_COMMON_H
