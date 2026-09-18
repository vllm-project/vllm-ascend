/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef VLLM_ASCEND_RMS_NORM_CAST_BASE_H
#define VLLM_ASCEND_RMS_NORM_CAST_BASE_H

#include "kernel_operator.h"

namespace RmsNormCast {
using namespace AscendC;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ != 220 && __CCE_AICORE__ != 310
#define bfloat16_t int16_t
#endif

constexpr int32_t NUM_PER_REP_FP32 = 64;

template <typename T>
__aicore__ inline T Min(T left, T right)
{
    return left < right ? left : right;
}

template <typename Tp, Tp value_>
struct IntegralConstant {
    static constexpr Tp value = value_;
};
using TrueType = IntegralConstant<bool, true>;
using FalseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public FalseType {};
template <typename Tp>
struct IsSame<Tp, Tp> : public TrueType {};

__aicore__ inline void ReduceSumCustom(
    const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& work, int32_t count)
{
    constexpr uint64_t mask = NUM_PER_REP_FP32;
    const int32_t repeat_times = count / NUM_PER_REP_FP32;
    const int32_t tail_count = count % NUM_PER_REP_FP32;
    const int32_t body_count = repeat_times * NUM_PER_REP_FP32;
    BinaryRepeatParams params;
    params.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
    params.src0BlkStride = 1;
    params.src1RepStride = 0;
    params.src1BlkStride = 1;
    params.dstRepStride = 0;
    params.dstBlkStride = 1;
    Duplicate(work, 0.0f, NUM_PER_REP_FP32);
    PipeBarrier<PIPE_V>();
    if (repeat_times > 0) {
        Add(work, src, work, mask, repeat_times, params);
        PipeBarrier<PIPE_V>();
    }
    if (tail_count != 0) {
        Add(work, src[body_count], work, tail_count, 1, params);
        PipeBarrier<PIPE_V>();
    }
    AscendCUtils::SetMask<float>(NUM_PER_REP_FP32);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(dst, work, MASK_PLACEHOLDER, 1, 0, 1, 0);
    }
#elif !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    WholeReduceSum<float, false>(dst, work, MASK_PLACEHOLDER, 1, 1, 1,
                                 DEFAULT_REPEAT_STRIDE);
#endif
    PipeBarrier<PIPE_V>();
}

template <typename T, typename U, typename R>
__aicore__ inline void DataCopyCustom(
    const U& dst, const R& src, uint32_t count)
{
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || \
    (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    DataCopyParams params;
    params.blockLen = count * sizeof(T);
    params.blockCount = 1;
    if constexpr (IsSame<U, LocalTensor<T>>::value) {
        DataCopyPadParams pad_params;
        DataCopyPad(dst, src, params, pad_params);
    } else {
        DataCopyPad(dst, src, params);
    }
#else
    const int32_t values_per_block = ONE_BLK_SIZE / sizeof(T);
    if (count % values_per_block == 0) {
        DataCopy(dst, src, count);
    } else if constexpr (IsSame<U, LocalTensor<T>>::value) {
        DataCopy(dst, src, AlignUp(count, values_per_block));
    } else {
        if (count < values_per_block) {
            DataCopy(dst, src, values_per_block);
        } else {
            const int32_t aligned_count = count / values_per_block * values_per_block;
            DataCopy(dst, src, aligned_count);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            for (int32_t i = 0; i < values_per_block; ++i) {
                const T value = src.GetValue(count - values_per_block + i);
                src.SetValue(i, value);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(dst[count - values_per_block], src, values_per_block);
        }
    }
#endif
}
}  // namespace RmsNormCast
#endif
