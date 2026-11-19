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
 * \file hc_post_d_split.h
 * \brief D-split kernel with batch-level coefficient reuse.
 */
#ifndef HC_POST_D_SPLIT_H
#define HC_POST_D_SPLIT_H

#include "kernel_operator.h"

namespace HcPost {
using namespace AscendC;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DEFAULT_BLOCK_STRIDE = 1;
constexpr int64_t DEFAULT_REPEAT_STRIDE = 8;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t REPEAT_NUM = 64;
constexpr int32_t CAST_BUFFER_PADDING_BYTES = 256;
constexpr int32_t SUM_BUFFER_OFFSET = 120;
constexpr int32_t SUM_BUFFER_PADDING_BYTES = SUM_BUFFER_OFFSET * sizeof(float);

__aicore__ inline int32_t CeilDiv(int32_t a, int32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int32_t CeilAlign(int32_t a, int32_t b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num)
{
    int32_t elemNum = BLOCK_SIZE / sizeof(T);
    return CeilAlign(num, elemNum);
}

template <typename T1, typename T2, int32_t FIXED_HEADS = 0>
class HcPostKernelDSplit {
public:
    __aicore__ inline HcPostKernelDSplit() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, GM_ADDR post, GM_ADDR comb, GM_ADDR y, GM_ADDR workspace,
                                const HcPostTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();
    __aicore__ inline void DataCopyInX(int64_t batchIndex, int64_t dLoopTimes, int64_t dealNum);
    __aicore__ inline void DataCopyInPost(int64_t batchIndex);
    __aicore__ inline void DataCopyInResidual(int64_t batchIndex, int64_t dLoopTimes, int64_t dealNum);
    __aicore__ inline void DataCopyInComb(int64_t batchIndex);
    __aicore__ inline void DataCopyOut(int64_t batchIndex, int64_t dLoopTimes, int64_t dealNum);
    __aicore__ inline void DoCompute(LocalTensor<float> sumTempBuf, LocalTensor<float> postBrcb,
                                     LocalTensor<float> comBrcb, int64_t batchIndex, int64_t dLoop, int64_t dealNum);
    __aicore__ inline void DoProcess(int64_t batchSize);

private:
    __aicore__ inline int64_t HeadCount() const
    {
        if constexpr (FIXED_HEADS != 0) {
            return FIXED_HEADS;
        }
        return hcParam_;
    }

    TPipe* pipe_;
    const HcPostTilingData* tiling_;

    int32_t blkIdx_ = -1;
    int64_t hcParam_ = 0;
    int64_t dParam_ = 0;
    int64_t dOnceDealing_ = 0;
    int64_t dLastDealing_ = 0;
    int64_t batchOneCoreTail_ = 0;
    int64_t batchOneCore_ = 0;
    int64_t dSplitTime_ = 0;
    int64_t isFrontCore_ = 0;
    int64_t hcParamAlign_ = 0;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> residualGm_;
    GlobalTensor<T2> postGm_;
    GlobalTensor<T2> combGm_;
    GlobalTensor<T1> yGm_;

    TQue<QuePosition::VECIN, 1> inputQue_;
    TQue<QuePosition::VECIN, 1> postQue_;
    TQue<QuePosition::VECIN, 1> combQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<QuePosition::VECCALC> inputCastBuf_;
    TBuf<QuePosition::VECCALC> postCastBuf_;
    TBuf<QuePosition::VECCALC> combCastBuf_;
    TBuf<QuePosition::VECCALC> outCastBuf_;
    TBuf<QuePosition::VECCALC> tempSumBuf_;
    TBuf<QuePosition::VECCALC> postBrcbBuf_;
    TBuf<QuePosition::VECCALC> combBrcbBuf_;
};

template <typename T1, typename T2, int32_t FIXED_HEADS>
__aicore__ inline void HcPostKernelDSplit<T1, T2, FIXED_HEADS>::Init(GM_ADDR x, GM_ADDR residual, GM_ADDR post,
                                                                     GM_ADDR comb, GM_ADDR y, GM_ADDR workspace,
                                                                     const HcPostTilingData* tilingData, TPipe* pipe)
{
    blkIdx_ = GetBlockIdx();
    if (blkIdx_ >= tilingData->usedCoreNum) {
        return;
    }
    tiling_ = tilingData;
    pipe_ = pipe;
    hcParam_ = tilingData->hcParam;
    dParam_ = tilingData->dParam;
    batchOneCoreTail_ = tilingData->batchOneCoreTail;
    batchOneCore_ = tilingData->batchOneCore;
    dOnceDealing_ = tilingData->dOnceDealing;
    dLastDealing_ = tilingData->dLastDealing;
    dSplitTime_ = tilingData->dSplitTime;
    isFrontCore_ = blkIdx_ < tilingData->frontCore;
    int64_t frontCore = tilingData->frontCore;
    hcParamAlign_ = RoundUp<T2>(HeadCount());

    int64_t xOffset = blkIdx_ * batchOneCore_ * dParam_;
    int64_t residualOffset = blkIdx_ * batchOneCore_ * HeadCount() * dParam_;
    int64_t postOffset = blkIdx_ * batchOneCore_ * HeadCount();
    int64_t combOffset = blkIdx_ * batchOneCore_ * HeadCount() * HeadCount();
    int64_t yOffset = blkIdx_ * batchOneCore_ * HeadCount() * dParam_;
    if (!isFrontCore_) {
        xOffset = (blkIdx_ * batchOneCoreTail_ + frontCore) * dParam_;
        residualOffset = (blkIdx_ * batchOneCoreTail_ + frontCore) * HeadCount() * dParam_;
        postOffset = (blkIdx_ * batchOneCoreTail_ + frontCore) * HeadCount();
        combOffset = (blkIdx_ * batchOneCoreTail_ + frontCore) * HeadCount() * HeadCount();
        yOffset = (blkIdx_ * batchOneCoreTail_ + frontCore) * HeadCount() * dParam_;
    }
    xGm_.SetGlobalBuffer((__gm__ T1*)x + xOffset);
    residualGm_.SetGlobalBuffer((__gm__ T1*)residual + residualOffset);
    postGm_.SetGlobalBuffer((__gm__ T2*)post + postOffset);
    combGm_.SetGlobalBuffer((__gm__ T2*)comb + combOffset);
    yGm_.SetGlobalBuffer((__gm__ T1*)y + yOffset);

    pipe_->InitBuffer(postQue_, 2, HeadCount() * sizeof(T2));
    pipe_->InitBuffer(combQue_, 2, HeadCount() * hcParamAlign_ * sizeof(T2));
    pipe_->InitBuffer(outQue_, 2, HeadCount() * dOnceDealing_ * sizeof(T1));
    pipe_->InitBuffer(inputQue_, 2, HeadCount() * dOnceDealing_ * sizeof(T1));

    if constexpr (sizeof(T1) == 2) {
        constexpr int32_t castPadding = sizeof(T2) == 2 ? CAST_BUFFER_PADDING_BYTES : 0;
        pipe_->InitBuffer(outCastBuf_, HeadCount() * dOnceDealing_ * sizeof(float) + castPadding);
        pipe_->InitBuffer(inputCastBuf_, HeadCount() * dOnceDealing_ * sizeof(float));
    }
    if constexpr (sizeof(T2) == 2) {
        pipe_->InitBuffer(postCastBuf_, HeadCount() * sizeof(float));
        pipe_->InitBuffer(combCastBuf_, HeadCount() * RoundUp<T2>(HeadCount()) * sizeof(float));
    }
    pipe_->InitBuffer(postBrcbBuf_, 64 * sizeof(float));
    pipe_->InitBuffer(combBrcbBuf_, HeadCount() * 64 * sizeof(float));
    constexpr int32_t sumPadding = sizeof(T1) == 2 && sizeof(T2) == 2 ? SUM_BUFFER_PADDING_BYTES : 0;
    pipe_->InitBuffer(tempSumBuf_, HeadCount() * dOnceDealing_ * sizeof(float) + sumPadding);
}

template <typename T1, typename T2, int32_t FIXED_HEADS>
__aicore__ inline void HcPostKernelDSplit<T1, T2, FIXED_HEADS>::Process()
{
    if (blkIdx_ >= tiling_->usedCoreNum) {
        return;
    }
    if (isFrontCore_) {
        DoProcess(tiling_->batchOneCore);
    } else {
        DoProcess(tiling_->batchOneCoreTail);
    }
}

template <typename T1, typename T2, int32_t FIXED_HEADS>
__aicore__ inline void HcPostKernelDSplit<T1, T2, FIXED_HEADS>::DataCopyInX(int64_t batchIndex, int64_t dLoopTimes,
                                                                            int64_t dealNum)
{
    LocalTensor<T1> xUb = inputQue_.AllocTensor<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = dealNum * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T1> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(xUb, xGm_[batchIndex * dParam_ + dLoopTimes * dOnceDealing_], copyParams, dataCopyPadParams);
    inputQue_.EnQue<T1>(xUb);
}

template <typename T1, typename T2, int32_t FIXED_HEADS>
__aicore__ inline void HcPostKernelDSplit<T1, T2, FIXED_HEADS>::DataCopyInPost(int64_t batchIndex)
{
    LocalTensor<T2> postUb = postQue_.AllocTensor<T2>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = HeadCount() * sizeof(T2);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T2> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(postUb, postGm_[batchIndex * HeadCount()], copyParams, dataCopyPadParams);
    postQue_.EnQue<T2>(postUb);
}

template <typename T1, typename T2, int32_t FIXED_HEADS>
__aicore__ inline void HcPostKernelDSplit<T1, T2, FIXED_HEADS>::DataCopyInResidual(int64_t batchIndex,
                                                                                   int64_t dLoopTimes, int64_t dealNum)
{
    LocalTensor<T1> residualUb = inputQue_.AllocTensor<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = HeadCount();
    copyParams.blockLen = dealNum * sizeof(T1);
    copyParams.srcStride = (dParam_ - dealNum) * sizeof(T1);
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T1> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(residualUb, residualGm_[batchIndex * HeadCount() * dParam_ + dLoopTimes * dOnceDealing_], copyParams,
                dataCopyPadParams);
    inputQue_.EnQue<T1>(residualUb);
}

template <typename T1, typename T2, int32_t FIXED_HEADS>
__aicore__ inline void HcPostKernelDSplit<T1, T2, FIXED_HEADS>::DataCopyInComb(int64_t batchIndex)
{
    uint8_t padNum = BLOCK_SIZE / sizeof(T2) - HeadCount();
    LocalTensor<T2> combUb = combQue_.AllocTensor<T2>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = HeadCount();
    copyParams.blockLen = HeadCount() * sizeof(T2);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T2> dataCopyPadParams{true, 0, padNum, 0};
    DataCopyPad(combUb, combGm_[batchIndex * HeadCount() * HeadCount()], copyParams, dataCopyPadParams);
    combQue_.EnQue<T2>(combUb);
}

template <typename T1, typename T2, int32_t FIXED_HEADS>
__aicore__ inline void HcPostKernelDSplit<T1, T2, FIXED_HEADS>::DataCopyOut(int64_t batchIndex, int64_t dLoopTimes,
                                                                            int64_t dealNum)
{
    LocalTensor<T1> outBuf = outQue_.DeQue<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = HeadCount();
    copyParams.blockLen = dealNum * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = (dParam_ - dealNum) * sizeof(T1);
    AscendC::DataCopyPad(yGm_[batchIndex * HeadCount() * dParam_ + dLoopTimes * dOnceDealing_], outBuf, copyParams);
    outQue_.FreeTensor(outBuf);
}

template <typename T>
__aicore__ inline void DoBrcb(LocalTensor<T> srcLocal, LocalTensor<float> dstLocal,
                              TBuf<QuePosition::VECCALC> castLocalBuf, int64_t dealNum)
{
    uint32_t repeatTimes = CeilDiv(dealNum, REPEAT_NUM);
    if constexpr (sizeof(T) == 2) {
        LocalTensor<float> castBuf = castLocalBuf.Get<float>();
        Cast(castBuf, srcLocal, RoundMode::CAST_NONE, dealNum);
        PipeBarrier<PIPE_V>();
        Brcb(dstLocal, castBuf, repeatTimes, {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
    } else {
        Brcb(dstLocal, srcLocal, repeatTimes, {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void DoMal(LocalTensor<T> src0Local, LocalTensor<T> src1Local, LocalTensor<T> dstLocal,
                             int64_t curRowNum, int64_t curColNum)
{
    int64_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    int64_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    int64_t curColNumAlign = RoundUp<T>(curColNum);
    int64_t numRepeatPerLine = curColNum / elemInOneRepeat;
    int64_t tailCount = curColNum % elemInOneRepeat;
    BinaryRepeatParams instrParams;
    instrParams.dstBlkStride = 1;
    instrParams.src0BlkStride = 1;
    instrParams.src1BlkStride = 0;
    instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src1RepStride = 0;
    for (uint32_t i = 0; i < curRowNum; i++) {
        if (numRepeatPerLine > 0) {
            Mul(dstLocal[i * curColNumAlign], src0Local[0], src1Local[i * elemInOneBlock], elemInOneRepeat,
                numRepeatPerLine, instrParams);
        }
        if (tailCount > 0) {
            int64_t tailOffset = numRepeatPerLine * elemInOneRepeat;
            Mul(dstLocal[i * curColNumAlign + tailOffset], src0Local[tailOffset],
                src1Local[i * elemInOneBlock], tailCount, 1, instrParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T1, typename T2, int32_t FIXED_HEADS>
__aicore__ inline void HcPostKernelDSplit<T1, T2, FIXED_HEADS>::DoCompute(LocalTensor<float> sumTempBuf,
                                                                          LocalTensor<float> postBrcb,
                                                                          LocalTensor<float> comBrcb,
                                                                          int64_t batchIndex, int64_t dLoop,
                                                                          int64_t dealNum)
{
    LocalTensor<float> outBuf;
    if constexpr (sizeof(T1) == 2) {
        outBuf = outCastBuf_.Get<float>();
    } else {
        outBuf = outQue_.AllocTensor<T1>();
    }
    DataCopyInX(batchIndex, dLoop, dealNum);
    LocalTensor<T1> xUb = inputQue_.DeQue<T1>();
    LocalTensor<float> inputCastBuf;
    if constexpr (sizeof(T1) == 2) {
        inputCastBuf = inputCastBuf_.Get<float>();
        Cast(inputCastBuf, xUb, RoundMode::CAST_NONE, dealNum);
        PipeBarrier<PIPE_V>();
        DoMal<float>(inputCastBuf, postBrcb, outBuf, HeadCount(), dealNum);
    } else {
        DoMal<float>(xUb, postBrcb, outBuf, HeadCount(), dealNum);
    }
    inputQue_.FreeTensor(xUb);

    DataCopyInResidual(batchIndex, dLoop, dealNum);
    LocalTensor<T1> residualUb = inputQue_.DeQue<T1>();
    if constexpr (sizeof(T1) == 2) {
#pragma unroll
        for (int32_t i = 0; i < HeadCount(); i++) {
            Cast(inputCastBuf[i * RoundUp<float>(dealNum)], residualUb[i * RoundUp<T1>(dealNum)], RoundMode::CAST_NONE,
                 dealNum);
        }
        PipeBarrier<PIPE_V>();
    }

#pragma unroll
    for (int64_t hcIndex = 0; hcIndex < HeadCount(); hcIndex++) {
        if constexpr (sizeof(T1) == 2) {
            DoMal<float>(inputCastBuf[hcIndex * RoundUp<float>(dealNum)], comBrcb[hcIndex * 64], sumTempBuf,
                         HeadCount(), dealNum);
        } else {
            DoMal<float>(residualUb[hcIndex * RoundUp<T1>(dealNum)], comBrcb[hcIndex * 64], sumTempBuf, HeadCount(),
                         dealNum);
        }
        Add(outBuf, outBuf, sumTempBuf, static_cast<int32_t>(HeadCount() * RoundUp<float>(dealNum)));
        PipeBarrier<PIPE_V>();
    }
    inputQue_.FreeTensor(residualUb);
    if constexpr (sizeof(T1) == 2) {
        LocalTensor<T1> outSumBuf = outQue_.AllocTensor<T1>();
        uint32_t outAlign = RoundUp<float>(dealNum);
        uint32_t inputAlign = RoundUp<T1>(dealNum);
        if (dealNum == outAlign && dealNum == inputAlign) {
            Cast(outSumBuf, outBuf, RoundMode::CAST_RINT, HeadCount() * dealNum);
        } else {
#pragma unroll
            for (int32_t i = 0; i < HeadCount(); i++) {
                Cast(outSumBuf[i * inputAlign], outBuf[i * outAlign], RoundMode::CAST_RINT, dealNum);
            }
        }
        PipeBarrier<PIPE_V>();
        outQue_.EnQue<T1>(outSumBuf);
        DataCopyOut(batchIndex, dLoop, dealNum);
    } else {
        outQue_.EnQue<T1>(outBuf);
        DataCopyOut(batchIndex, dLoop, dealNum);
    }
}

template <typename T1, typename T2, int32_t FIXED_HEADS>
__aicore__ inline void HcPostKernelDSplit<T1, T2, FIXED_HEADS>::DoProcess(int64_t batchSize)
{
    LocalTensor<float> sumTempBuf = tempSumBuf_.Get<float>();
    if constexpr (sizeof(T1) == 2 && sizeof(T2) == 2) {
        // Offset the sum source by eight bank groups relative to the output source.
        sumTempBuf = sumTempBuf[SUM_BUFFER_OFFSET];
    }
    for (int64_t batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        DataCopyInPost(batchIndex);
        LocalTensor<T2> postUb = postQue_.DeQue<T2>();
        LocalTensor<float> postBrcb = postBrcbBuf_.Get<float>();
        DoBrcb<T2>(postUb, postBrcb, postCastBuf_, HeadCount());
        LocalTensor<float> comBrcb = combBrcbBuf_.Get<float>();
        DataCopyInComb(batchIndex);
        LocalTensor<T2> combUb = combQue_.DeQue<T2>();
        LocalTensor<float> combCastBuf;
        if constexpr (sizeof(T2) == 2) {
            combCastBuf = combCastBuf_.Get<float>();
#pragma unroll
            for (int32_t i = 0; i < HeadCount(); i++) {
                Cast(combCastBuf[i * RoundUp<float>(HeadCount())], combUb[i * hcParamAlign_], RoundMode::CAST_NONE,
                     HeadCount());
            }
            PipeBarrier<PIPE_V>();
        }
        // Coefficients are shared by all D tiles in this batch.
#pragma unroll
        for (int64_t hcIndex = 0; hcIndex < HeadCount(); hcIndex++) {
            if constexpr (sizeof(T2) == 2) {
                Brcb(comBrcb[hcIndex * 64], combCastBuf[hcIndex * RoundUp<float>(HeadCount())], 1,
                     {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
            } else {
                Brcb(comBrcb[hcIndex * 64], combUb[hcIndex * hcParamAlign_], 1,
                     {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
            }
        }
        PipeBarrier<PIPE_V>();
        for (int64_t dLoop = 0; dLoop < dSplitTime_; dLoop++) {
            DoCompute(sumTempBuf, postBrcb, comBrcb, batchIndex, dLoop, dOnceDealing_);
        }
        if (dLastDealing_ != 0) {
            DoCompute(sumTempBuf, postBrcb, comBrcb, batchIndex, dSplitTime_, dLastDealing_);
        }
        combQue_.FreeTensor(combUb);
        postQue_.FreeTensor(postUb);
    }
}

} // namespace HcPost
#endif
