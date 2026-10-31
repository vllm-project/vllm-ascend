/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file reduce_common.h
 */
#ifndef REDUCE_COMMON_H_RMS_NORM
#define REDUCE_COMMON_H_RMS_NORM
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t MAX_REP_NUM = 255;
constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr uint32_t ELEM_PER_BLK_FP32 = 8;
constexpr float ZERO = 0;
constexpr int32_t HALf_INTERVAL = 2;
constexpr int32_t INDEX_TWO = 2;
constexpr int32_t INDEX_FOUR = 4;
constexpr int32_t INDEX_EIGHT = 8;
constexpr int32_t INDEX_SIXTEEN = 16;

__aicore__ inline void ReduceSumForSmallReduceDimPreRepeat(
    const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal, const LocalTensor<float>& tmpLocal,
    const uint32_t elemNum, const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat,
    const uint8_t repStride)
{
    uint32_t elemIndex = 0;
    for (; elemIndex + ELEM_PER_REP_FP32 <= numLastDim; elemIndex += ELEM_PER_REP_FP32) {
        Add(tmpLocal, srcLocal[elemIndex], tmpLocal, elemNum, repeat,
            {1, 1, 1, ELEM_PER_BLK_FP32, repStride, ELEM_PER_BLK_FP32});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(tmpLocal, srcLocal[elemIndex], tmpLocal, tailCount, repeat,
            {1, 1, 1, ELEM_PER_BLK_FP32, repStride, ELEM_PER_BLK_FP32});
    }
    PipeBarrier<PIPE_V>();
    AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32); // set mask = 64
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
        WholeReduceSum<float, false>(dstLocal, tmpLocal, elemNum, repeat, 1, 1, ELEM_PER_BLK_FP32);
    }
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003
    WholeReduceSum(dstLocal, tmpLocal, elemNum, repeat, 1, 1, ELEM_PER_BLK_FP32);
#else
    WholeReduceSum<float, false>(dstLocal, tmpLocal, elemNum, repeat, 1, 1, ELEM_PER_BLK_FP32);
#endif
}

/*
 * reduce dim form (N, D) to (N, 1)
 * this reduce sum is for small reduce dim.
 */
__aicore__ inline void ReduceSumForSmallReduceDim(
    const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal, const LocalTensor<float>& tmpLocal,
    const uint32_t numLastDimAligned, const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat,
    const uint8_t repStride)
{
    uint32_t repeatTimes = repeat / MAX_REP_NUM;
    if (repeatTimes == 0) {
        ReduceSumForSmallReduceDimPreRepeat(
            dstLocal, srcLocal, tmpLocal, ELEM_PER_REP_FP32, numLastDim, tailCount, repeat, repStride);
    } else {
        uint32_t repTailNum = repeat % MAX_REP_NUM;
        uint32_t repIndex = 0;
        uint32_t repElem;
        for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
            ReduceSumForSmallReduceDimPreRepeat(
                dstLocal[repIndex], srcLocal[repIndex * numLastDimAligned], tmpLocal[repIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32, numLastDim, tailCount, MAX_REP_NUM, repStride);
        }
        if (repTailNum != 0) {
            ReduceSumForSmallReduceDimPreRepeat(
                dstLocal[repIndex], srcLocal[repIndex * numLastDimAligned], tmpLocal[repIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32, numLastDim, tailCount, repTailNum, repStride);
        }
    }
}

/*
 * reduce dim form (N, D) to (N, 1)
 * this reduce sum is for small reduce dim, require D < 255 * 8.
 * size of tmpLocal: (N, 64)
 */
__aicore__ inline void ReduceSumMultiN(
    const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal, const LocalTensor<float>& tmpLocal,
    const uint32_t numRow, const uint32_t numCol, const uint32_t numColAlign)
{
    const uint32_t tailCount = numCol % ELEM_PER_REP_FP32;
    const uint32_t repeat = numRow;
    const uint8_t repStride = numColAlign / ELEM_PER_BLK_FP32;
    Duplicate(tmpLocal, ZERO, numRow * ELEM_PER_REP_FP32);
    PipeBarrier<PIPE_V>();
    ReduceSumForSmallReduceDim(dstLocal, srcLocal, tmpLocal, numColAlign, numCol, tailCount, repeat, repStride);
}

__aicore__ inline int32_t findPowerTwo(int32_t n)
{
    // find max power of 2 no more than n (32 bit)
    n |= n >> 1; // Set the first digit of n's binary to 1
    n |= n >> INDEX_TWO;
    n |= n >> INDEX_FOUR;
    n |= n >> INDEX_EIGHT;
    n |= n >> INDEX_SIXTEEN;
    return (n + 1) >> 1;
}

__aicore__ inline void ReduceSumHalfInterval(
    const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local, int32_t count)
{
    if (likely(count > ELEM_PER_REP_FP32)) {
        int32_t bodyCount = findPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(src_local, src_local, src_local[bodyCount], tailCount);
            PipeBarrier<PIPE_V>();
        }
        while (bodyCount > ELEM_PER_REP_FP32) {
            bodyCount = bodyCount / HALf_INTERVAL;
            Add(src_local, src_local, src_local[bodyCount], bodyCount);
            PipeBarrier<PIPE_V>();
        }

        AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(dst_local, src_local, ELEM_PER_REP_FP32, 1, 0, 1, 0);
    }
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003
    WholeReduceSum(dst_local, src_local, ELEM_PER_REP_FP32, 1, 1, 1, ELEM_PER_BLK_FP32);
#else
    WholeReduceSum<float, false>(dst_local, src_local, ELEM_PER_REP_FP32, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    PipeBarrier<PIPE_V>();
}

__aicore__ inline float ReduceSumHalfInterval(const LocalTensor<float>& src_local, int32_t count)
{
    if (likely(count > ELEM_PER_REP_FP32)) {
        int32_t bodyCount = findPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(src_local, src_local, src_local[bodyCount], tailCount);
            PipeBarrier<PIPE_V>();
        }
        while (bodyCount > ELEM_PER_REP_FP32) {
            bodyCount = bodyCount / HALf_INTERVAL;
            Add(src_local, src_local, src_local[bodyCount], bodyCount);
            PipeBarrier<PIPE_V>();
        }

        AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(src_local, src_local, ELEM_PER_REP_FP32, 1, 0, 1, 0);
    }
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003
    WholeReduceSum(src_local, src_local, ELEM_PER_REP_FP32, 1, 1, 1, ELEM_PER_BLK_FP32);
#else
    WholeReduceSum<float, false>(src_local, src_local, ELEM_PER_REP_FP32, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event_v_s);
    WaitFlag<HardEvent::V_S>(event_v_s);
    return src_local.GetValue(0);
}
#endif // _REDUCE_COMMON_H_
