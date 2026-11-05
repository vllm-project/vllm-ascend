/*  Adapted from
 *      https://gitee.com/ascend/ascend-transformer-boost.git
 *
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_UTILS_COMMON_KERNEL_KERNEL_UTILS_H
#define ASCEND_OPS_UTILS_COMMON_KERNEL_KERNEL_UTILS_H
#include "kernel_operator.h"

using AscendC::HardEvent;

__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : ((x + y - 1) / y);
}

__aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y = 16)
{
    return (x + y - 1) / y * y;
}

__aicore__ inline uint32_t Min(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}

__aicore__ inline uint32_t Max(uint32_t x, uint32_t y)
{
    return x > y ? x : y;
}

template <typename T, typename Q>
__aicore__ inline void CopyIn(const AscendC::GlobalTensor<T> &gm, Q &queue, uint64_t offset, uint32_t count)
{
    AscendC::LocalTensor<T> local = queue.template AllocTensor<T>();
    DataCopy(local, gm[offset], count);
    queue.EnQue(local);
}

template <typename T, typename Q>
__aicore__ inline void CopyOut(const AscendC::GlobalTensor<T> &gm, Q &queue, uint64_t offset, uint32_t count)
{
    AscendC::LocalTensor<T> local = queue.template DeQue<T>();
    DataCopy(gm[offset], local, count);
    queue.FreeTensor(local);
}

template <typename T>
__aicore__ inline void CastFrom16To32(const AscendC::LocalTensor<float> &out, const AscendC::LocalTensor<T> &in,
                                      uint32_t count)
{
    Cast(out, in, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void CastFrom32To16(const AscendC::LocalTensor<T> &out, const AscendC::LocalTensor<float> &in,
                                      uint32_t count)
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(out, in, AscendC::RoundMode::CAST_NONE,
             count);  // 310p cast fp32->half 只能用CAST_NONE，这里拉齐310p和910b
    } else {          // bf16
        Cast(out, in, AscendC::RoundMode::CAST_RINT, count);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void CastFromF16ToI8(const AscendC::LocalTensor<int8_t> &out, const AscendC::LocalTensor<half> &in,
                                       half quantMin, uint32_t count)
{
    Maxs(in, in, quantMin, count);
    AscendC::PipeBarrier<PIPE_V>();
    Mins(in, in, (half)127, count);  // 127: limit
    AscendC::PipeBarrier<PIPE_V>();
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    Cast(out, in, AscendC::RoundMode::CAST_RINT, count);
#else
    Cast(out, in, AscendC::RoundMode::CAST_NONE, count);
#endif
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, typename Q>
__aicore__ inline void CopyInAndCastF32(const AscendC::LocalTensor<float> &out, const AscendC::GlobalTensor<T> &gm,
                                        Q &queue, uint64_t offset, uint32_t count)
{
    CopyIn(gm, queue, offset, count);
    AscendC::LocalTensor<T> local = queue.template DeQue<T>();
    Cast(out, local, AscendC::RoundMode::CAST_NONE, count);
    queue.FreeTensor(local);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, typename Q>
__aicore__ inline void Cast16AndCopyOut(const AscendC::LocalTensor<float> &in, const AscendC::GlobalTensor<T> &gm,
                                        Q &queue, uint64_t offset, uint32_t count)
{
    AscendC::LocalTensor<T> local = queue.template AllocTensor<T>();
    CastFrom32To16(local, in, count);
    queue.EnQue(local);
    CopyOut(gm, queue, offset, count);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline T ComputeSum(const AscendC::LocalTensor<T> &in, const AscendC::LocalTensor<T> &tmp,
                               const AscendC::LocalTensor<T> &workLocal, uint32_t count)
{
#if __CCE_AICORE__ == 100
    float sum = 0;
    int64_t elementNumPerRep = AscendC::ONE_REPEAT_BYTE_SIZE / sizeof(T);
    AscendC::LocalTensor<T> src = in;
    while (count > elementNumPerRep) {
        int64_t repeatTimes = count / elementNumPerRep;
        int64_t tailCount = count % elementNumPerRep;
        int64_t bodyCount = repeatTimes * elementNumPerRep;
        if (repeatTimes > 0) {
            AscendC::AscendCUtils::SetMask<T>(elementNumPerRep);
            vcadd((__ubuf__ T *)tmp.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), repeatTimes, 1, 1, 8);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);  // PipeBarrier(PIPE_V)?
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }

        if (tailCount != 0) {
            AscendC::AscendCUtils::SetMask<T>(tailCount);
            vcadd((__ubuf__ T *)tmp[bodyCount].GetPhyAddr(), (__ubuf__ T *)src[bodyCount].GetPhyAddr(), 1, 1, 1, 8);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            sum += tmp.GetValue(bodyCount);
        }

        count = repeatTimes;
        src = tmp;
    }

    if (count > 1) {
        AscendC::AscendCUtils::SetMask<T>(count);
        vcadd((__ubuf__ T *)tmp.GetPhyAddr(), (__ubuf__ T *)tmp.GetPhyAddr(), 1, 1, 1, 8);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    sum += tmp.GetValue(0);
    return sum;
#else
    ReduceSum(tmp, in, workLocal, count);
    AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
    return tmp.GetValue(0);
#endif
}

__aicore__ inline float ComputeSliceSquareSum(const AscendC::LocalTensor<float> &in,
                                              const AscendC::LocalTensor<float> &tmp,
                                              const AscendC::LocalTensor<float> &workLocal, uint32_t count)
{
    Mul(tmp, in, in, count);
    AscendC::PipeBarrier<PIPE_V>();
    return ComputeSum(tmp, tmp, workLocal, count);
}
template <typename T>
__aicore__ inline void ComputeRmsNorm(const AscendC::LocalTensor<T> &out, const AscendC::LocalTensor<float> &in,
                                      float rms, const AscendC::LocalTensor<T> &gamma, uint32_t count,
                                      uint32_t precisionMode, uint32_t gemmaMode,
                                      const AscendC::LocalTensor<float> &tmp)
{
    float value = 1.0;
    Duplicate(tmp, rms, count);
    AscendC::PipeBarrier<PIPE_V>();
    Div(tmp, in, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();

    if (precisionMode == 0) {
        CastFrom16To32(in, gamma, count);
        AscendC::PipeBarrier<PIPE_V>();
        if (gemmaMode == 1) {
            Adds(in, in, value, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Mul(in, in, tmp, count);
        AscendC::PipeBarrier<PIPE_V>();
        CastFrom32To16(out, in, count);
        return;
    }
    if constexpr (std::is_same<T, half>::value) {
        CastFrom32To16(out, tmp, count);
        Mul(out, out, gamma, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T, uint32_t gemmaMode>
__aicore__ inline void CastGAndIsGemmaMode(const AscendC::LocalTensor<float> &out, const AscendC::LocalTensor<T> &gamma,
                                           uint32_t count)
{
    Cast(out, gamma, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    float value = 1.0;
    if constexpr (gemmaMode == 1) {
        Adds(out, out, value, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T, uint32_t precisionMode>
__aicore__ inline void ComputeRmsNormFast(const AscendC::LocalTensor<T> &out, const AscendC::LocalTensor<float> &in,
                                          float rms, const AscendC::LocalTensor<T> &gamma, uint32_t count,
                                          const AscendC::LocalTensor<float> &tmp,
                                          const AscendC::LocalTensor<float> &fp32_g)
{
    float value = 1.0;
    Duplicate(tmp, rms, count);
    AscendC::PipeBarrier<PIPE_V>();
    Div(tmp, in, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (precisionMode == 0) {
        Mul(in, fp32_g, tmp, count);
        AscendC::PipeBarrier<PIPE_V>();
        CastFrom32To16(out, in, count);
        return;
    }
    if constexpr (std::is_same<T, half>::value) {
        CastFrom32To16(out, tmp, count);
        Mul(out, out, gamma, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <bool WITH_BETA = true>
__aicore__ inline void ComputeRmsNorm(const AscendC::LocalTensor<float> &out, const AscendC::LocalTensor<float> &in,
                                      float rms, const AscendC::LocalTensor<half> &gamma,
                                      const AscendC::LocalTensor<half> &beta, const AscendC::LocalTensor<float> &tmp,
                                      uint32_t count)
{
    Duplicate(tmp, rms, count);
    AscendC::PipeBarrier<PIPE_V>();
    Div(out, in, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();
    CastFrom16To32(tmp, gamma, count);
    Mul(out, out, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (WITH_BETA) {
        CastFrom16To32(tmp, beta, count);
        Add(out, out, tmp, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void ComputeRmsNorm(const AscendC::LocalTensor<float> &out, const AscendC::LocalTensor<float> &in,
                                      float reciprocal_of_rms, const AscendC::LocalTensor<T> &gamma,
                                      const AscendC::LocalTensor<float> &tmp, const AscendC::LocalTensor<T> &res_out,
                                      uint32_t count)
{
    Duplicate(tmp, reciprocal_of_rms, count);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(out, in, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();
    CastFrom16To32(tmp, gamma, count);
    Mul(out, out, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();
    CastFrom32To16(res_out, out, count);
}

template <typename T>
__aicore__ inline void ComputeResidualAdd(const AscendC::LocalTensor<T> &out, const AscendC::LocalTensor<T> &in,
                                          const AscendC::LocalTensor<T> &resIn, uint32_t count)
{
    Add(out, in, resIn, count);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void ComputeMean(const AscendC::LocalTensor<T> &out, const AscendC::LocalTensor<T> &in, T aveNum,
                                   uint32_t count)
{
    Duplicate(out, aveNum, count);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(out, in, out, count);
    AscendC::PipeBarrier<PIPE_V>();
    T sum = ComputeSum(out, out, out, count);
    AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(out, sum, count);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void ComputeLayerNorm(const AscendC::LocalTensor<float> &out, const AscendC::LocalTensor<float> &in,
                                        const AscendC::LocalTensor<float> &mean, float eps, float aveNum,
                                        const AscendC::LocalTensor<T> &gamma, const AscendC::LocalTensor<T> &beta,
                                        uint32_t count)
{
    Sub(in, in, mean, count);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(out, in, in, count);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(out, out, aveNum, count);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceSum(out, out, out, count);
    AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float var = out.GetValue(0);
    AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(out, var, count);
    AscendC::PipeBarrier<PIPE_V>();
    Adds(out, out, eps, count);
    AscendC::PipeBarrier<PIPE_V>();
    Sqrt(out, out, count);
    AscendC::PipeBarrier<PIPE_V>();

    Div(out, in, out, count);
    AscendC::PipeBarrier<PIPE_V>();

    Cast(in, gamma, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(out, out, in, count);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(in, beta, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    Add(out, out, in, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void ComputeFp16ToI8Quant(const AscendC::LocalTensor<int8_t> &out,
                                            const AscendC::LocalTensor<half> &in, const AscendC::LocalTensor<half> &tmp,
                                            half scale, half offset, half quantMin, uint32_t count)
{
    Muls(tmp, in, scale, count);
    AscendC::PipeBarrier<PIPE_V>();
    Adds(tmp, tmp, offset, count);
    AscendC::PipeBarrier<PIPE_V>();
    CastFromF16ToI8(out, tmp, quantMin, count);
}

__aicore__ inline void ComputeFp32ToI8Quant(const AscendC::LocalTensor<int8_t> &out,
                                            const AscendC::LocalTensor<float> &in,
                                            const AscendC::LocalTensor<half> &tmp, half scale, half offset,
                                            half quantMin, uint32_t count)
{
    CastFrom32To16(tmp, in, count);
    AscendC::PipeBarrier<PIPE_V>();
    ComputeFp16ToI8Quant(out, tmp, tmp, scale, offset, quantMin, count);
}

__aicore__ inline void ComputeHighPrecisionFp32ToI8Quant(const AscendC::LocalTensor<int8_t> &out,
                                                         const AscendC::LocalTensor<float> &in,
                                                         const AscendC::LocalTensor<half> &tmp, float scale,
                                                         float offset, half quantMin, uint32_t count)
{
    Muls(in, in, scale, count);
    AscendC::PipeBarrier<PIPE_V>();
    Adds(in, in, offset, count);
    AscendC::PipeBarrier<PIPE_V>();
    CastFrom32To16(tmp, in, count);
    CastFromF16ToI8(out, tmp, quantMin, count);
}

__aicore__ inline void CopyGmTilingToUb(__ubuf__ uint8_t *&tilingInUb, const __gm__ uint8_t *tilingInGm,
                                        size_t tilingSize, AscendC::TPipe *pipe)
{
    uint32_t roundTilingSize = RoundUp(tilingSize, 32);
    AscendC::TBuf<AscendC::TPosition::VECCALC> tilingBuf;
    AscendC::GlobalTensor<uint8_t> tilingGm;

    tilingGm.SetGlobalBuffer((__gm__ uint8_t *)tilingInGm);
    pipe->InitBuffer(tilingBuf, roundTilingSize);

    AscendC::LocalTensor<uint8_t> tilingUb = tilingBuf.Get<uint8_t>();
    AscendC::DataCopy(tilingUb, tilingGm, roundTilingSize);

    tilingInUb = (__ubuf__ uint8_t *)tilingUb.GetPhyAddr();
}

template <typename T>
__aicore__ inline uint32_t GetReduceSumWorkLocalSize(uint32_t sliceSize)
{
    uint32_t elementsPerBlock = 32 / sizeof(T);
    uint32_t elementsPerRepeat = 256 / sizeof(T);

    uint32_t firstMaxRepeat = sliceSize < elementsPerRepeat ? 1u : (sliceSize / elementsPerRepeat);
    uint32_t iter1OutputCount = firstMaxRepeat;
    uint32_t iter1AlignEnd = RoundUp(iter1OutputCount, elementsPerBlock);
    return iter1AlignEnd;
}

#endif
