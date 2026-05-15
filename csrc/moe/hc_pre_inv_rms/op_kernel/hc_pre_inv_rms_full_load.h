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
 * \file hc_pre_inv_rms.h
 * \brief inv rms file
 */
#ifndef ASCENDC_HC_PRE_INV_RMS_FULL_LOAD_H_
#define ASCENDC_HC_PRE_INV_RMS_FULL_LOAD_H_
#include "kernel_operator.h"

namespace HcPreInvRms {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t FLOAT_BTYPE_SIZE = 4;
constexpr uint32_t PER_REPEAT_LEN_B32 = 64;
constexpr uint32_t UB_BLOCK_SIZE = 32;
constexpr int32_t B16_TYPE_BYTE_SIZE = 2;
constexpr int32_t B32_TYPE_BYTE_SIZE = 4;
constexpr int32_t ONE_COUNT = 1;
constexpr int32_t FOUR_FOLD = 4;
constexpr int32_t DST_REP_STRIDE = 1;
constexpr int32_t SRC_BLK_STRIDE = 1;
constexpr int32_t SRC_REP_STRIDE = 8;

template <typename T>
class HcPreInvRmsFullLoad {
public:
    __aicore__ inline HcPreInvRmsFullLoad() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const HcPreInvRmsFullLoadTilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(uint64_t idx, uint64_t curUbFactorA);
    __aicore__ inline void Compute(uint64_t curUbFactorA);
    __aicore__ inline void ComputeB16(uint64_t curUbFactorA);
    __aicore__ inline void ComputeB32(uint64_t curUbFactorA);
    __aicore__ inline void CopyOut(uint64_t idx, uint64_t curUbFactorA);

private:
    TPipe* pipe_;
    
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<TPosition::VECCALC> castBuf;
    TBuf<TPosition::VECCALC> reduceBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<float> yGm;

    int64_t A; // 输入数据 A 轴大小
    int64_t R; // 输入数据 R 轴大小
    int64_t blockNumA; // 使用的核数
    int64_t blockFactorA; // 每个核处理的A个数
    int64_t blockTailFactorA; // 尾核处理的A个数
    int64_t ubFactorA; // 每次ub循环处理的A个数
    int32_t blockIdx_;
    float epsilon;  // 算子参数
    uint32_t curBlockFactorA; // 当前核处理的A个数
    uint32_t rAlign;
    uint32_t rAlignB32;
    uint32_t reduceBufNum;
};

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoad<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const HcPreInvRmsFullLoadTilingData* tiling, TPipe* pipe)
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
    // pipe alloc memory to queue, the unit is Bytes
    pipe_->InitBuffer(inQueueX, BUFFER_NUM, ubFactorA * rAlign * sizeof(T));
    pipe_->InitBuffer(outQueueY, BUFFER_NUM, ubFactorA * FLOAT_BTYPE_SIZE);

    reduceBufNum = (rAlignB32 + PER_REPEAT_LEN_B32 - 1) / PER_REPEAT_LEN_B32;
    pipe_->InitBuffer(reduceBuf, ubFactorA * reduceBufNum * FLOAT_BTYPE_SIZE);
    if constexpr (sizeof(T) == B16_TYPE_BYTE_SIZE) {
        pipe_->InitBuffer(castBuf, ubFactorA * rAlignB32 * FLOAT_BTYPE_SIZE);
    }
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoad<T>::Process()
{
    if (blockIdx_ >= blockNumA) {
        return;
    }
    uint64_t aUbLoopCount = (curBlockFactorA + ubFactorA - 1) / ubFactorA; // Ub循环次数
    uint64_t tailUbFactorA = curBlockFactorA - (aUbLoopCount - 1) * ubFactorA; // 最后一次Ub循环的A轴大小
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
__aicore__ inline void HcPreInvRmsFullLoad<T>::CopyIn(uint64_t idx, uint64_t curUbFactorA)
{
    
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();

    DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
    int32_t xGmStartAddr = idx * R * ubFactorA;
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(curUbFactorA), static_cast<uint32_t>(R * sizeof(T)), 0, 0, 0};
    DataCopyPad(xLocal, xGm[xGmStartAddr], dataCopyParams, dataCopyPadParams);
    
    inQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoad<T>::Compute(uint64_t curUbFactorA)
{
    if constexpr (sizeof(T) == B16_TYPE_BYTE_SIZE) {
        ComputeB16(curUbFactorA);
    } else if constexpr (sizeof(T) == B32_TYPE_BYTE_SIZE) {
        ComputeB32(curUbFactorA);
    }
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoad<T>::ComputeB16(uint64_t curUbFactorA)
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

    LocalTensor<float> castLocal = castBuf.Get<float>();
    LocalTensor<float> reduceLocal = reduceBuf.Get<float>();

    int32_t perFoldElems = (rAlignB32 + FOUR_FOLD - 1) / FOUR_FOLD;  // 4096
    int32_t perFoldRepTime = (perFoldElems + PER_REPEAT_LEN_B32 - 1) / PER_REPEAT_LEN_B32; // 64

    AscendC::Cast(castLocal, xLocal, AscendC::RoundMode::CAST_NONE, R);
    PipeBarrier<PIPE_V>();
    AscendC::Mul(castLocal, castLocal, castLocal, R);

    for (int idx = 0; idx < curUbFactorA; idx++) {
        
        for (int j = 0; j < FOUR_FOLD; j++) {
            PipeBarrier<PIPE_V>();
            WholeReduceSum(reduceLocal[idx * reduceBufNum + j * perFoldRepTime], castLocal[idx * rAlignB32 + j * perFoldElems], PER_REPEAT_LEN_B32, perFoldRepTime, 
                DST_REP_STRIDE, SRC_BLK_STRIDE, SRC_REP_STRIDE);
        }

        PipeBarrier<PIPE_V>(); 
        WholeReduceSum(reduceLocal, reduceLocal, PER_REPEAT_LEN_B32, FOUR_FOLD, DST_REP_STRIDE, SRC_BLK_STRIDE, SRC_REP_STRIDE);
        PipeBarrier<PIPE_V>(); 
        WholeReduceSum(yLocal[idx], reduceLocal, FOUR_FOLD, 1, DST_REP_STRIDE, SRC_BLK_STRIDE, SRC_REP_STRIDE);
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
__aicore__ inline void HcPreInvRmsFullLoad<T>::ComputeB32(uint64_t curUbFactorA)
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

    int32_t perFoldElems = (rAlignB32 + FOUR_FOLD - 1) / FOUR_FOLD;  // 4096
    int32_t perFoldRepTime = (perFoldElems + PER_REPEAT_LEN_B32 - 1) / PER_REPEAT_LEN_B32; // 64

    LocalTensor<float> reduceLocal = reduceBuf.Get<float>();
    PipeBarrier<PIPE_V>();
    AscendC::Mul(xLocal, xLocal, xLocal, R);

    for (int idx = 0; idx < curUbFactorA; idx++) {
        for (int j = 0; j < FOUR_FOLD; j++) {
            PipeBarrier<PIPE_V>();
            WholeReduceSum(reduceLocal[idx * reduceBufNum + j * perFoldRepTime], xLocal[idx * rAlignB32 + j * perFoldElems], PER_REPEAT_LEN_B32, perFoldRepTime, 
                DST_REP_STRIDE, SRC_BLK_STRIDE, SRC_REP_STRIDE);
        }
        PipeBarrier<PIPE_V>(); 
        WholeReduceSum(reduceLocal, reduceLocal, PER_REPEAT_LEN_B32, FOUR_FOLD, DST_REP_STRIDE, SRC_BLK_STRIDE, SRC_REP_STRIDE);
        PipeBarrier<PIPE_V>(); 
        WholeReduceSum(yLocal[idx], reduceLocal, FOUR_FOLD, 1, DST_REP_STRIDE, SRC_BLK_STRIDE, SRC_REP_STRIDE);
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
__aicore__ inline void HcPreInvRmsFullLoad<T>::CopyOut(uint64_t idx, uint64_t curUbFactorA)
{
    LocalTensor<float> yLocal = outQueueY.DeQue<float>();
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curUbFactorA * sizeof(float)), 0, 0, 0};
    DataCopyPad(yGm[idx * ubFactorA], yLocal, copyParams);
    outQueueY.FreeTensor(yLocal);
}

} // namespace HcPreInvRms
#endif // ASCENDC_HC_PRE_INV_RMS_FULL_LOAD_H_

