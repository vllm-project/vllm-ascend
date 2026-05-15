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
 * \file add_rms_norm_dynamic_quant_helper.h
 * \brief
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_HELPER_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_HELPER_H_

#include "reduce_common.h"
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#endif

using namespace AscendC;
constexpr uint32_t FLOAT_BLOCK_ELEM = 8;
constexpr int32_t ROW_FACTOR = 128;
constexpr uint32_t ELEM_PER_BLK_FP16 = 16;
constexpr float DYNAMIC_QUANT_DIVIDEND = 127.0;
constexpr float DYNAMIC_QUANT_DIVIDEND_INT4 = 7.0;

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

__aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

__aicore__ inline uint32_t ROUND_UP32(uint32_t x)
{
    return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
}

__aicore__ inline uint32_t TWO_NUMS_MIN(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}

__aicore__ inline uint32_t TWO_NUMS_MAX(uint32_t x, uint32_t y)
{
    return x > y ? x : y;
}

template <typename T>
__aicore__ inline uint32_t CalculateBlockLen(const uint32_t len)
{
    if constexpr (std::is_same_v<T, int4b_t>) {
        return len / 2;
    } else {
        return len * sizeof(T);
    }
}

template <typename T, template <typename U> typename R, template <typename U> typename S>
__aicore__ inline void DataCopyEx(
    const R<T>& dst, const S<T>& src, const uint32_t len, const uint32_t count = 1, const bool ubAligned = false)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = count;
    copyParams.blockLen = CalculateBlockLen<T>(len);

    if constexpr (is_same<R<T>, AscendC::LocalTensor<T>>::value) {
        copyParams.srcStride = 0;
        copyParams.dstStride = (ubAligned) ? 1 : 0;
        DataCopyPad(dst, src, copyParams, {});
    } else {
        copyParams.srcStride = (ubAligned) ? 1 : 0;
        copyParams.dstStride = 0;
        DataCopyPad(dst, src, copyParams);
    }
}

template <typename T, template <typename U> typename R, template <typename U> typename S>
__aicore__ inline void DataCopyExStride(
    const R<T>& dst, const S<T>& src, const uint32_t len, const uint32_t count = 1, const uint32_t ubAligned = 0)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = count;
    copyParams.blockLen = CalculateBlockLen<T>(len);

    if constexpr (is_same<R<T>, AscendC::LocalTensor<T>>::value) {
        copyParams.srcStride = 0;
        copyParams.dstStride = ubAligned;
        DataCopyPad(dst, src, copyParams, {});
    } else {
        copyParams.srcStride = ubAligned;
        copyParams.dstStride = 0;
        DataCopyPad(dst, src, copyParams);
    }
}

/*
 * only support count in (128, 255 * 64)
 * about 20us faster than above in case fp16:(1024, 11264) on 910B
 */
__aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& srcLocal, int32_t count)
{
    uint64_t repsFp32 = count >> 6;       // 6 is count / ELEM_PER_REP_FP32
    uint64_t offsetsFp32 = repsFp32 << 6; // 6 is repsFp32 * ELEM_PER_REP_FP32
    uint64_t remsFp32 = count & 0x3f;     // 0x3f 63, count % ELEM_PER_REP_FP32

    if (likely(repsFp32 > 1)) {
        // 8 is rep stride
        Max(srcLocal, srcLocal[ELEM_PER_REP_FP32], srcLocal, ELEM_PER_REP_FP32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? ELEM_PER_REP_FP32 : count;
    // 8 is rep stride
    WholeReduceMax(srcLocal, srcLocal, mask, 1, 8, 1, 8);
    PipeBarrier<PIPE_V>();
}

/*
 * only support count in (128, 255 * 64)
 * about 6us slower than above in case fp16:(1024, 11264) on 910B
 */
__aicore__ inline void ReduceSumInplace(const LocalTensor<float>& srcLocal, int32_t count)
{
    uint64_t repsFp32 = count >> 6;       // 6 is cound / ELEM_PER_REP_FP32
    uint64_t offsetsFp32 = repsFp32 << 6; // 6 is repsFp32 * ELEM_PER_REP_FP32
    uint64_t remsFp32 = count & 0x3f;     // 0x3f 63, count % ELEM_PER_REP_FP32

    if (likely(repsFp32 > 1)) {
        // 8 is rep stride
        Add(srcLocal, srcLocal[ELEM_PER_REP_FP32], srcLocal, ELEM_PER_REP_FP32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0)) {
        Add(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? ELEM_PER_REP_FP32 : count;
    // 8 is rep stride
    WholeReduceSum(srcLocal, srcLocal, mask, 1, 8, 1, 8);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void DivScalarFP32(
    LocalTensor<float>& dstTensor, LocalTensor<float>& dividendTensor, LocalTensor<float>& tmpTensor,
    float divisorScalar, uint32_t count)
{
    uint32_t repsFp32 = count >> 6;                        // 6 is devide 64
    uint32_t offsetsFp32 = count & 0xffffffc0;             // 0xffffffc0 is floor by 64
    uint32_t remsFp32 = count & 0x3f;                      // 0x3f is mod(64)
    Duplicate(tmpTensor, divisorScalar, FLOAT_BLOCK_ELEM); // FLOAT_BLOCK_ELEM);
    PipeBarrier<PIPE_V>();
    Div(dstTensor, dividendTensor, tmpTensor, ELEM_PER_REP_FP32, repsFp32, {1, 1, 0, 8, 8, 0});
    if ((remsFp32 > 0)) {
        Div(dstTensor[offsetsFp32], dividendTensor[offsetsFp32], tmpTensor, remsFp32, 1, {1, 1, 0, 8, 8, 0});
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void RoundFloat2IntQuant(LocalTensor<T>& dstTensor, LocalTensor<float>& srcTensor, int32_t size)
{
    Cast(srcTensor.ReinterpretCast<int32_t>(), srcTensor, RoundMode::CAST_RINT, size);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(srcTensor.ReinterpretCast<half>(), srcTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, size);
    PipeBarrier<PIPE_V>();
    Cast(dstTensor, srcTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, size);
    PipeBarrier<PIPE_V>();
}
#endif // __ADD_RMS_NORM_DYNAMIC_QUANT_HELPER_H_
