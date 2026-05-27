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
 * \file hc_pre_inv_rms_full_load_large_d.h
 * \brief inv rms for large d (R=28672, d=7168)
 */
#ifndef ASCENDC_HC_PRE_INV_RMS_FULL_LOAD_LARGE_D_H_
#define ASCENDC_HC_PRE_INV_RMS_FULL_LOAD_LARGE_D_H_
#include "kernel_operator.h"

namespace HcPreInvRmsLargeD {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t FLOAT_BTYPE_SIZE = 4;
constexpr uint32_t PER_REPEAT_LEN_B32 = 64;
constexpr uint32_t UB_BLOCK_SIZE = 32;
constexpr int32_t B16_TYPE_BYTE_SIZE = 2;
constexpr int32_t B32_TYPE_BYTE_SIZE = 4;
constexpr int32_t HALf_INTERVAL = 2;
constexpr int32_t INDEX_TWO = 2;
constexpr int32_t INDEX_FOUR = 4;
constexpr int32_t INDEX_EIGHT = 8;
constexpr int32_t INDEX_SIXTEEN = 16;

__aicore__ inline int32_t FindPowerTwo(int32_t n)
{
    n |= n >> 1;
    n |= n >> INDEX_TWO;
    n |= n >> INDEX_FOUR;
    n |= n >> INDEX_EIGHT;
    n |= n >> INDEX_SIXTEEN;
    return (n + 1) >> 1;
}

__aicore__ inline void ReduceSumHalfInterval(
    const LocalTensor<float>& dst_local, const LocalTensor<float>& src_local, int32_t count)
{
    if (likely(count > PER_REPEAT_LEN_B32)) {
        int32_t bodyCount = FindPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(src_local, src_local, src_local[bodyCount], tailCount);
            PipeBarrier<PIPE_V>();
        }
        while (bodyCount > PER_REPEAT_LEN_B32) {
            bodyCount = bodyCount / HALf_INTERVAL;
            Add(src_local, src_local, src_local[bodyCount], bodyCount);
            PipeBarrier<PIPE_V>();
        }
        AscendCUtils::SetMask<float>(PER_REPEAT_LEN_B32);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(dst_local, src_local, MASK_PLACEHOLDER, 1, 0, 1, 0);
    }
#else
    WholeReduceSum<float, false>(dst_local, src_local, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    PipeBarrier<PIPE_V>();
}

template <typename T>
class HcPreInvRmsFullLoadLargeD {
public:
    __aicore__ inline HcPreInvRmsFullLoadLargeD() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const HcPreInvRmsFullLoadTilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(uint64_t idx, uint64_t curUbFactorA);
    __aicore__ inline void Compute(uint64_t curUbFactorA);
    __aicore__ inline void ComputeB16(uint64_t curUbFactorA);
    __aicore__ inline void ComputeB32(uint64_t curUbFactorA);
    __aicore__ inline void CopyOut(uint64_t idx, uint64_t curUbFactorA);

private:
    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> castBuf;
    TBuf<TPosition::VECCALC> reduceBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<float> yGm;

    int64_t A;
    int64_t R;
    int64_t blockNumA;
    int64_t blockFactorA;
    int64_t blockTailFactorA;
    int64_t ubFactorA;
    int32_t blockIdx_;
    float epsilon;
    uint32_t curBlockFactorA;
    uint32_t rAlign;
    uint32_t rAlignB32;
    uint32_t reduceBufNum;
};

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadLargeD<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const HcPreInvRmsFullLoadTilingData* tiling, TPipe* pipe)
{
    A = tiling->A;
    R = tiling->R;
    blockNumA = tiling->blockNumA;
    blockFactorA = tiling->blockFactorA;
    blockTailFactorA = tiling->blockTailFactorA;
    ubFactorA = tiling->ubFactorA;
    epsilon = tiling->epsilon;

    rAlign = ((R * sizeof(T) + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE) * (UB_BLOCK_SIZE / sizeof(T));
    rAlignB32 = ((R * FLOAT_BTYPE_SIZE + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE) * (UB_BLOCK_SIZE / FLOAT_BTYPE_SIZE);

    pipe_ = pipe;

    blockIdx_ = GetBlockIdx();

    if (blockIdx_ < blockNumA - 1) {
        this->curBlockFactorA = this->blockFactorA;
    } else if (blockIdx_ == blockNumA - 1) {
        this->curBlockFactorA = this->blockTailFactorA;
    } else {
        return;
    }
    xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * blockFactorA * R, curBlockFactorA * R);
    yGm.SetGlobalBuffer((__gm__ float*)y + blockIdx_ * blockFactorA, curBlockFactorA);
    pipe_->InitBuffer(inQueueX, BUFFER_NUM, ubFactorA * rAlign * sizeof(T));
    pipe_->InitBuffer(outQueueY, BUFFER_NUM, ubFactorA * FLOAT_BTYPE_SIZE);

    reduceBufNum = (rAlignB32 + PER_REPEAT_LEN_B32 - 1) / PER_REPEAT_LEN_B32;
    pipe_->InitBuffer(reduceBuf, ubFactorA * reduceBufNum * FLOAT_BTYPE_SIZE);
    if constexpr (sizeof(T) == B16_TYPE_BYTE_SIZE) {
        pipe_->InitBuffer(castBuf, ubFactorA * rAlignB32 * FLOAT_BTYPE_SIZE);
    }
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadLargeD<T>::Process()
{
    if (blockIdx_ >= blockNumA) {
        return;
    }
    uint64_t aUbLoopCount = (curBlockFactorA + ubFactorA - 1) / ubFactorA;
    uint64_t tailUbFactorA = curBlockFactorA - (aUbLoopCount - 1) * ubFactorA;
    uint64_t curUbFactorA = ubFactorA;
    for (uint64_t idx = 0; idx < aUbLoopCount; idx++) {
        if (idx == aUbLoopCount - 1) {
            curUbFactorA = tailUbFactorA;
        }
        CopyIn(idx, curUbFactorA);
        Compute(curUbFactorA);
        CopyOut(idx, curUbFactorA);
    }
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadLargeD<T>::CopyIn(uint64_t idx, uint64_t curUbFactorA)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();

    DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
    int64_t xGmStartAddr = idx * R * ubFactorA;
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(curUbFactorA), static_cast<uint32_t>(R * sizeof(T)), 0, 0, 0};
    DataCopyPad(xLocal, xGm[xGmStartAddr], dataCopyParams, dataCopyPadParams);

    inQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadLargeD<T>::Compute(uint64_t curUbFactorA)
{
    if constexpr (sizeof(T) == B16_TYPE_BYTE_SIZE) {
        ComputeB16(curUbFactorA);
    } else if constexpr (sizeof(T) == B32_TYPE_BYTE_SIZE) {
        ComputeB32(curUbFactorA);
    }
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadLargeD<T>::ComputeB16(uint64_t curUbFactorA)
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

    LocalTensor<float> castLocal = castBuf.Get<float>();
    LocalTensor<float> reduceLocal = reduceBuf.Get<float>();

    AscendC::Cast(castLocal, xLocal, AscendC::RoundMode::CAST_NONE, R);
    PipeBarrier<PIPE_V>();
    AscendC::Mul(castLocal, castLocal, castLocal, R);

    for (int idx = 0; idx < curUbFactorA; idx++) {
        ReduceSumHalfInterval(yLocal[idx], castLocal[idx * rAlignB32], R);
    }

    float meanCof = 1.0f / R;
    PipeBarrier<PIPE_V>();
    AscendC::Muls(yLocal, yLocal, meanCof, curUbFactorA);
    PipeBarrier<PIPE_V>();
    AscendC::Adds(yLocal, yLocal, epsilon, curUbFactorA);
    PipeBarrier<PIPE_V>();
    AscendC::Duplicate(reduceLocal, 1.0f, curUbFactorA);
    PipeBarrier<PIPE_V>();
    AscendC::Sqrt(yLocal, yLocal, curUbFactorA);
    PipeBarrier<PIPE_V>();
    AscendC::Div(yLocal, reduceLocal, yLocal, curUbFactorA);

    outQueueY.EnQue<float>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadLargeD<T>::ComputeB32(uint64_t curUbFactorA)
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

    LocalTensor<float> reduceLocal = reduceBuf.Get<float>();
    PipeBarrier<PIPE_V>();
    AscendC::Mul(xLocal, xLocal, xLocal, R);

    for (int idx = 0; idx < curUbFactorA; idx++) {
        ReduceSumHalfInterval(yLocal[idx], xLocal[idx * rAlignB32], R);
    }

    float meanCof = 1.0f / R;
    PipeBarrier<PIPE_V>();
    AscendC::Muls(yLocal, yLocal, meanCof, curUbFactorA);
    PipeBarrier<PIPE_V>();
    AscendC::Adds(yLocal, yLocal, epsilon, curUbFactorA);
    PipeBarrier<PIPE_V>();
    AscendC::Duplicate(reduceLocal, 1.0f, curUbFactorA);
    PipeBarrier<PIPE_V>();
    AscendC::Sqrt(yLocal, yLocal, curUbFactorA);
    PipeBarrier<PIPE_V>();
    AscendC::Div(yLocal, reduceLocal, yLocal, curUbFactorA);

    outQueueY.EnQue<float>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadLargeD<T>::CopyOut(uint64_t idx, uint64_t curUbFactorA)
{
    LocalTensor<float> yLocal = outQueueY.DeQue<float>();
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curUbFactorA * sizeof(float)), 0, 0, 0};
    DataCopyPad(yGm[idx * ubFactorA], yLocal, copyParams);
    outQueueY.FreeTensor(yLocal);
}

} // namespace HcPreInvRmsLargeD
#endif // ASCENDC_HC_PRE_INV_RMS_FULL_LOAD_LARGE_D_H_