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
 * \file rotate_interleaved_split_s_pad.h
 * \brief
 */
#ifndef ROTATE_INTERLEAVED_SPLIT_S_PAD_H
#define ROTATE_INTERLEAVED_SPLIT_S_PAD_H
#include "rotate_interleaved_common.h"

namespace RotateInterleavedN {
using namespace AscendC;

template <typename T>
class InterleavedSplitSPad {
public:
    __aicore__ inline InterleavedSplitSPad(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                const RopeRegbaseTilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

protected:
    GlobalTensor<T> xGm;
    GlobalTensor<T> cosGm;
    GlobalTensor<T> sinGm;
    GlobalTensor<T> yGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueCos;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueY;
    TBuf<TPosition::VECCALC> tmpFp32Buf1;
    TBuf<TPosition::VECCALC> tmpFp32Buf2;
    TBuf<TPosition::VECCALC> tmpFp32Buf3;
    TBuf<TPosition::VECCALC> gatherOffsetBuf;
    const RopeRegbaseTilingData* tiling_;

    // tilingdata
    uint64_t batchSize;
    uint64_t seqLen;
    uint64_t numHeads;
    uint64_t headDim;
    uint64_t frontCoreNum;
    uint64_t tailCoreNum;
    uint64_t coreCalcNum;
    uint64_t coreCalcTail;
    uint64_t ubCalcNum;
    uint64_t ubCalcLoop;
    uint64_t ubCalcTail;
    uint64_t ubCalcTailNum;
    uint64_t ubCalcTailLoop;
    uint64_t ubCalcTailTail;
    uint64_t bufferNdSizeAll;
    uint64_t allHeadDim;
    uint64_t start;

    // init tmp data
    uint32_t alignLen;
    uint32_t headDimAlign;
    uint32_t allHeadDimAlign;
    uint32_t blockIdx;
    uint32_t ubCalcSeq;
    uint32_t ubCalcSeqTail;
    uint32_t ubCalcSeqLoop;
    uint64_t ioOffset;
    uint64_t ioOffsetAll;
    uint64_t triOffset;
    uint64_t bufferBsndSize;
    uint64_t bufferSdSize;
    uint64_t bufferNdSize;
    uint64_t bufferLenSize;
    uint64_t gatherOffsetLenSize;
    uint32_t blockNum = BLOCK_SIZE / sizeof(T);

    __aicore__ inline void InitData(const RopeRegbaseTilingData *tiling);
    __aicore__ inline void CopyInX(LocalTensor<T> &x, uint32_t loopIdx, uint32_t calcLen);
    __aicore__ inline void CopyInCos(LocalTensor<T> &cos, uint32_t loopIdx, uint32_t calcLen);
    __aicore__ inline void CopyInSin(LocalTensor<T> &sin, uint32_t loopIdx, uint32_t calcLen);
    __aicore__ inline void CopyOut(uint32_t loopIdx, uint32_t calcLen);
    __aicore__ inline void Compute(uint32_t loopIdx, LocalTensor<uint32_t> &gatherOffsetCast, uint32_t calcLen);
    __aicore__ inline void ComputeCastFp32(uint32_t loopIdx, LocalTensor<uint32_t> &gatherOffsetCast, uint32_t calcLen);
};

template <typename T>
__aicore__ inline void InterleavedSplitSPad<T>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                     const RopeRegbaseTilingData *tiling, TPipe *pipe)
{
    InitData(tiling);

    blockIdx = GetBlockIdx();
    bufferSdSize = seqLen * headDim;
    bufferNdSize = numHeads * headDim;
    bufferNdSizeAll = numHeads * allHeadDim;

    if (blockIdx < frontCoreNum) {
        ubCalcSeq = ubCalcNum;
        ubCalcSeqTail = ubCalcTail;
        ubCalcSeqLoop = ubCalcLoop;
        ioOffset = blockIdx * coreCalcNum * bufferNdSize;
        ioOffsetAll = blockIdx * coreCalcNum * bufferNdSizeAll;
        triOffset = blockIdx * coreCalcNum * headDim;
    } else if (coreCalcTail != 0) {
        ubCalcSeq = ubCalcTailNum;
        ubCalcSeqTail = ubCalcTailTail;
        ubCalcSeqLoop = ubCalcTailLoop;
        ioOffset = frontCoreNum * coreCalcNum * bufferNdSize + (blockIdx - frontCoreNum) * coreCalcTail * bufferNdSize;
        ioOffsetAll = frontCoreNum * coreCalcNum * bufferNdSizeAll + (blockIdx - frontCoreNum) * coreCalcTail * bufferNdSizeAll;
        triOffset = frontCoreNum * coreCalcNum * headDim + (blockIdx - frontCoreNum) * coreCalcTail * headDim;
    }

    bufferBsndSize = batchSize * seqLen * bufferNdSizeAll;
    xGm.SetGlobalBuffer((__gm__ T *)x + ioOffsetAll, bufferBsndSize);
    yGm.SetGlobalBuffer((__gm__ T *)y + ioOffsetAll, bufferBsndSize);
    cosGm.SetGlobalBuffer((__gm__ T *)cos + triOffset, bufferSdSize);
    sinGm.SetGlobalBuffer((__gm__ T *)sin + triOffset, bufferSdSize);

    bufferLenSize = batchSize * ubCalcSeq * numHeads * headDimAlign * sizeof(T);
    pipe->InitBuffer(inQueX, BUFFER_NUM, bufferLenSize);
    pipe->InitBuffer(inQueCos, BUFFER_NUM, bufferLenSize);
    pipe->InitBuffer(outQueY, BUFFER_NUM, bufferLenSize);

    if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
        bufferLenSize = batchSize * ubCalcSeq * numHeads * headDimAlign * sizeof(float);
        pipe->InitBuffer(tmpFp32Buf1, bufferLenSize);
        pipe->InitBuffer(tmpFp32Buf2, bufferLenSize);
        pipe->InitBuffer(tmpFp32Buf3, bufferLenSize);
    }

    gatherOffsetLenSize = numHeads * headDimAlign * sizeof(int32_t);
    pipe->InitBuffer(gatherOffsetBuf, gatherOffsetLenSize);
}

template <typename T>
__aicore__ inline void InterleavedSplitSPad<T>::InitData(const RopeRegbaseTilingData *tiling)
{
    tiling_ = tiling;
    batchSize = tiling_->batchSize;
    seqLen = tiling_->seqLen;
    numHeads = tiling_->numHeads;
    headDim = tiling_->headDim;
    frontCoreNum = tiling_->frontCoreNum;
    tailCoreNum = tiling_->tailCoreNum;
    coreCalcNum = tiling_->coreCalcNum;
    coreCalcTail = tiling_->coreCalcTail;
    ubCalcNum = tiling_->ubCalcNum;
    ubCalcLoop = tiling_->ubCalcLoop;
    ubCalcTail = tiling_->ubCalcTail;
    ubCalcTailNum = tiling_->ubCalcTailNum;
    ubCalcTailLoop = tiling_->ubCalcTailLoop;
    ubCalcTailTail = tiling_->ubCalcTailTail;
    allHeadDim = tiling_->allHeadDim;
    start = tiling_->start;

    alignLen = (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) ? ALIGN_16 : ALIGN_32;
    headDimAlign = (headDim + alignLen - 1) / alignLen * alignLen;
    allHeadDimAlign = (allHeadDim + alignLen - 1) / alignLen * alignLen;
}

template <typename T>
__aicore__ inline void InterleavedSplitSPad<T>::CopyInX(LocalTensor<T> &x, uint32_t loopIdx, uint32_t calcLen)
{
    DataCopyExtParams dataCopyParams;
    for (uint32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        dataCopyParams.blockCount = calcLen * numHeads;
        dataCopyParams.blockLen = headDim * sizeof(T);
        dataCopyParams.srcStride = (allHeadDim -  headDim)* sizeof(T);
        dataCopyParams.dstStride = 0;
        DataCopyPad(x[batchIdx * calcLen * numHeads * headDimAlign],
                    xGm[batchIdx * seqLen * bufferNdSizeAll + loopIdx * ubCalcSeq * bufferNdSizeAll + start], dataCopyParams,
                    {false, 0, 0, 0});
    }
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
}

template <typename T>
__aicore__ inline void InterleavedSplitSPad<T>::CopyInCos(LocalTensor<T> &cos, uint32_t loopIdx, uint32_t calcLen)
{
    DataCopyExtParams dataCopyTriParams;
    dataCopyTriParams.blockCount = calcLen;
    dataCopyTriParams.blockLen = headDim * sizeof(T);
    dataCopyTriParams.srcStride = 0;
    dataCopyTriParams.dstStride = static_cast<uint16_t>((numHeads - 1) * headDimAlign / blockNum);
    DataCopyPad(cos, cosGm[loopIdx * ubCalcSeq * headDim], dataCopyTriParams, {false, 0, 0, 0});
    event_t eventId2MTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId2MTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventId2MTE2ToV);
    BroadCastTriToBsnd(cos, batchSize, calcLen, numHeads, headDimAlign);
}

template <typename T>
__aicore__ inline void InterleavedSplitSPad<T>::CopyInSin(LocalTensor<T> &sin, uint32_t loopIdx, uint32_t calcLen)
{
    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    DataCopyExtParams dataCopyTriParams;
    dataCopyTriParams.blockCount = calcLen;
    dataCopyTriParams.blockLen = headDim * sizeof(T);
    dataCopyTriParams.srcStride = 0;
    dataCopyTriParams.dstStride = static_cast<uint16_t>((numHeads - 1) * headDimAlign / blockNum);
    DataCopyPad(sin, sinGm[loopIdx * ubCalcSeq * headDim], dataCopyTriParams, {false, 0, 0, 0});
    event_t eventId3MTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId3MTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventId3MTE2ToV);
    BroadCastTriToBsnd(sin, batchSize, calcLen, numHeads, headDimAlign);
}

template <typename T>
__aicore__ inline void InterleavedSplitSPad<T>::CopyOut(uint32_t loopIdx, uint32_t calcLen)
{
    LocalTensor<T> y = outQueY.DeQue<T>();
    DataCopyExtParams dataCopyParams;
    for (uint32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        dataCopyParams.blockCount = calcLen * numHeads;
        dataCopyParams.blockLen = headDim * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (allHeadDim -  headDim)* sizeof(T);
        DataCopyPad(yGm[batchIdx * seqLen * bufferNdSizeAll + loopIdx * ubCalcSeq * bufferNdSizeAll + start],
                    y[batchIdx * calcLen * numHeads * headDimAlign], dataCopyParams);
    }
    outQueY.FreeTensor(y);
}

template <typename T>
__aicore__ inline void InterleavedSplitSPad<T>::Process()
{
    LocalTensor<int32_t> gatherOffset = gatherOffsetBuf.Get<int32_t>();
    SetGatherSrcOffset(gatherOffset, headDimAlign * numHeads, static_cast<int32_t>(sizeof(float)));
    LocalTensor<uint32_t> gatherOffsetCast = gatherOffset.ReinterpretCast<uint32_t>();
    if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
        for (uint32_t loop = 0; loop < (ubCalcSeqTail == 0 ? ubCalcSeqLoop : ubCalcSeqLoop - 1); ++loop) {
            ComputeCastFp32(loop, gatherOffsetCast, ubCalcSeq);
            CopyOut(loop, ubCalcSeq);
        }
        if (ubCalcSeqTail != 0) {
            ComputeCastFp32(ubCalcSeqLoop - 1, gatherOffsetCast, ubCalcSeqTail);
            CopyOut(ubCalcSeqLoop - 1, ubCalcSeqTail);
        }
    } else {
        for (uint32_t loop = 0; loop < (ubCalcSeqTail == 0 ? ubCalcSeqLoop : ubCalcSeqLoop - 1); ++loop) {
            Compute(loop, gatherOffsetCast, ubCalcSeq);
            CopyOut(loop, ubCalcSeq);
        }
        if (ubCalcSeqTail != 0) {
            Compute(ubCalcSeqLoop - 1, gatherOffsetCast, ubCalcSeqTail);
            CopyOut(ubCalcSeqLoop - 1, ubCalcSeqTail);
        }
    }
}

template <typename T>
__aicore__ inline void InterleavedSplitSPad<T>::Compute(uint32_t loopIdx, LocalTensor<uint32_t> &gatherOffsetCast,
                                                        uint32_t calcLen)
{
    uint64_t totalCount = calcLen * batchSize * numHeads * headDimAlign;

    LocalTensor<T> x = inQueX.AllocTensor<T>();
    CopyInX(x, loopIdx, calcLen);

    LocalTensor<T> cos = inQueCos.AllocTensor<T>();
    CopyInCos(cos, loopIdx, calcLen);

    LocalTensor<T> y = outQueY.AllocTensor<T>();
    Mul(y, x, cos, totalCount);
    for (uint32_t i = 0; i < batchSize * calcLen; ++i) {
        Gather(x[i * numHeads * headDimAlign], x[i * numHeads * headDimAlign], gatherOffsetCast, 0,
               numHeads * headDimAlign);
    }

    CopyInSin(cos, loopIdx, calcLen);

    Mul(x, x, cos, totalCount);
    inQueCos.FreeTensor(cos);
    InterleavedInversion(x, totalCount);
    Add(y, y, x, totalCount);
    inQueX.FreeTensor(x);
    outQueY.EnQue(y);
}

template <typename T>
__aicore__ inline void
InterleavedSplitSPad<T>::ComputeCastFp32(uint32_t loopIdx, LocalTensor<uint32_t> &gatherOffsetCast, uint32_t calcLen)
{
    uint64_t totalCount = calcLen * batchSize * numHeads * headDimAlign;

    LocalTensor<T> xTensor = inQueX.AllocTensor<T>();
    CopyInX(xTensor, loopIdx, calcLen);
    LocalTensor<float> tmp32SPadBuf1 = tmpFp32Buf1.Get<float>();
    Cast(tmp32SPadBuf1, xTensor, RoundMode::CAST_NONE, totalCount);
    inQueX.FreeTensor(xTensor);

    LocalTensor<T> cos = inQueCos.AllocTensor<T>();
    CopyInCos(cos, loopIdx, calcLen);
    LocalTensor<float> tmp32SPadBuf2 = tmpFp32Buf2.Get<float>();
    Cast(tmp32SPadBuf2, cos, RoundMode::CAST_NONE, totalCount);

    LocalTensor<float> tmp32Buf3 = tmpFp32Buf3.Get<float>();
    Mul(tmp32Buf3, tmp32SPadBuf1, tmp32SPadBuf2, totalCount);

    for (uint32_t i = 0; i < batchSize * calcLen; ++i) {
        Gather(tmp32SPadBuf1[i * numHeads * headDimAlign], tmp32SPadBuf1[i * numHeads * headDimAlign], gatherOffsetCast,
               0, numHeads * headDimAlign);
    }

    CopyInSin(cos, loopIdx, calcLen);
    Cast(tmp32SPadBuf2, cos, RoundMode::CAST_NONE, totalCount);
    inQueCos.FreeTensor(cos);

    Mul(tmp32SPadBuf1, tmp32SPadBuf1, tmp32SPadBuf2, totalCount);
    InterleavedInversion(tmp32SPadBuf1, totalCount);
    Add(tmp32Buf3, tmp32Buf3, tmp32SPadBuf1, totalCount);

    LocalTensor<T> y = outQueY.AllocTensor<T>();
    Cast(y, tmp32Buf3, RoundMode::CAST_RINT, totalCount);
    outQueY.EnQue(y);
}

} // namespace RotateInterleavedN

#endif // ROTATE_INTERLEAVED_SPLIT_S_PAD_H
