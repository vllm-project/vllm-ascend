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
 * \file rotate_interleaved_split_bs_pad.h
 * \brief
 */
#ifndef ROTATE_INTERLEAVED_SPLIT_BS_PAD_H
#define ROTATE_INTERLEAVED_SPLIT_BS_PAD_H
#include "rotate_interleaved_common.h"

namespace RotateInterleavedN {
using namespace AscendC;

template <typename T>
class InterleavedSplitBSPad {
public:
    __aicore__ inline InterleavedSplitBSPad(){};
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
    uint64_t allHeadDim;
    uint64_t start;
    uint64_t ioOffsetAll;
    uint64_t bufferNdSizeAll;

    // tilingdata
    uint64_t batchSize;
    uint64_t seqLen;
    uint64_t numHeads;
    uint64_t headDim;
    uint64_t frontCoreNum;
    uint64_t tailCoreNum;
    uint64_t coreCalcNum;
    uint64_t coreCalcTail;
    uint64_t ubCalcBNum;
    uint64_t ubCalcBLoop;
    uint64_t ubCalcBTail;

    // init tmp data
    uint32_t alignLen;
    uint32_t headDimAlign;
    uint32_t blockIdx;
    uint32_t ubCalcSeq;
    uint32_t ubCalcSeqTail;
    uint32_t ubCalcSeqLoop;
    uint64_t ioOffset;
    uint64_t triOffset;
    uint64_t bufferBsndSize;
    uint64_t bufferSdSize;
    uint64_t bufferNdSize;
    uint64_t bufferLenSize;
    uint64_t gatherOffsetLenSize;
    uint32_t blockNum = BLOCK_SIZE / sizeof(T);

    __aicore__ inline void InitData(const RopeRegbaseTilingData *tiling);
    __aicore__ inline void CopyInX(LocalTensor<T> &x, uint32_t seqIdx, uint32_t batchIdx, uint32_t calcLen);
    __aicore__ inline void CopyInCos(LocalTensor<T> &cos, uint32_t seqIdx, uint32_t calcLen);
    __aicore__ inline void CopyInSin(LocalTensor<T> &sin, uint32_t seqIdx, uint32_t calcLen);
    __aicore__ inline void CopyOut(uint32_t seqIdx, uint32_t batchIdx, uint32_t calcLen);
    __aicore__ inline void Compute(uint32_t seqIdx, uint32_t batchIdx, LocalTensor<uint32_t> &gatherOffsetCast,
                                   uint32_t calcLen);
    __aicore__ inline void ComputeCastFp32(uint32_t seqIdx, uint32_t batchIdx, LocalTensor<uint32_t> &gatherOffsetCast,
                                           uint32_t calcLen);
};

template <typename T>
__aicore__ inline void InterleavedSplitBSPad<T>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                      const RopeRegbaseTilingData *tiling, TPipe *pipe)
{
    InitData(tiling);

    blockIdx = GetBlockIdx();
    bufferSdSize = seqLen * headDim;
    bufferNdSize = numHeads * headDim;
    bufferNdSizeAll = numHeads * allHeadDim;

    if (blockIdx < frontCoreNum) {
        ubCalcSeqLoop = coreCalcNum;
        ioOffset = blockIdx * coreCalcNum * bufferNdSize;
        ioOffsetAll = blockIdx * coreCalcNum * bufferNdSizeAll;
        triOffset = blockIdx * coreCalcNum * headDim;
    } else if (coreCalcTail != 0) {
        ubCalcSeqLoop = coreCalcTail;
        ioOffset = frontCoreNum * coreCalcNum * bufferNdSize + (blockIdx - frontCoreNum) * coreCalcTail * bufferNdSize;
        ioOffsetAll = frontCoreNum * coreCalcNum * bufferNdSizeAll + (blockIdx - frontCoreNum) * coreCalcTail * bufferNdSizeAll;
        triOffset = frontCoreNum * coreCalcNum * headDim + (blockIdx - frontCoreNum) * coreCalcTail * headDim;
    }

    bufferBsndSize = batchSize * seqLen * bufferNdSizeAll;
    xGm.SetGlobalBuffer((__gm__ T *)x + ioOffsetAll, bufferBsndSize);
    yGm.SetGlobalBuffer((__gm__ T *)y + ioOffsetAll, bufferBsndSize);
    cosGm.SetGlobalBuffer((__gm__ T *)cos + triOffset, bufferSdSize);
    sinGm.SetGlobalBuffer((__gm__ T *)sin + triOffset, bufferSdSize);

    bufferLenSize = ubCalcBNum * numHeads * headDimAlign * sizeof(T);
    pipe->InitBuffer(inQueX, BUFFER_NUM, bufferLenSize);
    pipe->InitBuffer(inQueCos, BUFFER_NUM, bufferLenSize);
    pipe->InitBuffer(outQueY, BUFFER_NUM, bufferLenSize);

    if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
        bufferLenSize = ubCalcBNum * numHeads * headDimAlign * sizeof(float);
        pipe->InitBuffer(tmpFp32Buf1, bufferLenSize);
        pipe->InitBuffer(tmpFp32Buf2, bufferLenSize);
        pipe->InitBuffer(tmpFp32Buf3, bufferLenSize);
    }

    gatherOffsetLenSize = numHeads * headDimAlign * sizeof(int32_t);
    pipe->InitBuffer(gatherOffsetBuf, gatherOffsetLenSize);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSPad<T>::InitData(const RopeRegbaseTilingData *tiling)
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
    ubCalcBNum = tiling_->ubCalcBNum;
    ubCalcBLoop = tiling_->ubCalcBLoop;
    ubCalcBTail = tiling_->ubCalcBTail;
    allHeadDim = tiling_->allHeadDim;
    start = tiling_->start;

    alignLen = (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) ? ALIGN_16 : ALIGN_32;
    headDimAlign = (headDim + alignLen - 1) / alignLen * alignLen;
}

template <typename T>
__aicore__ inline void InterleavedSplitBSPad<T>::CopyInX(LocalTensor<T> &x, uint32_t seqIdx, uint32_t batchIdx,
                                                         uint32_t calcLen)
{
    DataCopyExtParams dataCopyParams;
    uint64_t startOffset = batchIdx * ubCalcBNum * seqLen * bufferNdSizeAll + seqIdx * bufferNdSizeAll;

    for (uint32_t loopIdx = 0; loopIdx < calcLen; ++loopIdx) {
        dataCopyParams.blockCount = numHeads;
        dataCopyParams.blockLen = headDim * sizeof(T);
        dataCopyParams.srcStride = (allHeadDim - headDim) * sizeof(T);
        dataCopyParams.dstStride = 0;
        DataCopyPad(x[loopIdx * numHeads * headDimAlign], xGm[startOffset + loopIdx * seqLen * bufferNdSizeAll + start],
                    dataCopyParams, {false, 0, 0, 0});
    }

    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSPad<T>::CopyInCos(LocalTensor<T> &cos, uint32_t seqIdx, uint32_t calcLen)
{
    DataCopyExtParams bsPadDataCopyTriParams;
    bsPadDataCopyTriParams.blockCount = 1;
    bsPadDataCopyTriParams.blockLen = headDim * sizeof(T);
    bsPadDataCopyTriParams.srcStride = 0;
    bsPadDataCopyTriParams.dstStride = 0;
    DataCopyPad(cos, cosGm[seqIdx * headDim], bsPadDataCopyTriParams, {false, 0, 0, 0});
    event_t eventId2MTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId2MTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventId2MTE2ToV);
    BroadCastTriToB1nd(cos, calcLen, numHeads, headDimAlign);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSPad<T>::CopyInSin(LocalTensor<T> &sin, uint32_t seqIdx, uint32_t calcLen)
{
    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    DataCopyExtParams bsPadDataCopyTriParams;
    bsPadDataCopyTriParams.blockCount = 1;
    bsPadDataCopyTriParams.blockLen = headDim * sizeof(T);
    bsPadDataCopyTriParams.srcStride = 0;
    bsPadDataCopyTriParams.dstStride = 0;
    DataCopyPad(sin, sinGm[seqIdx * headDim], bsPadDataCopyTriParams, {false, 0, 0, 0});
    event_t eventId3MTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId3MTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventId3MTE2ToV);
    BroadCastTriToB1nd(sin, calcLen, numHeads, headDimAlign);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSPad<T>::CopyOut(uint32_t seqIdx, uint32_t batchIdx, uint32_t calcLen)
{
    LocalTensor<T> y = outQueY.DeQue<T>();
    DataCopyExtParams dataCopyParams;
    uint64_t startOffset = batchIdx * ubCalcBNum * seqLen * bufferNdSizeAll + seqIdx * bufferNdSizeAll;
    for (uint32_t loopIdx = 0; loopIdx < calcLen; ++loopIdx) {
        dataCopyParams.blockCount = numHeads;
        dataCopyParams.blockLen = headDim * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride =(allHeadDim - headDim) * sizeof(T);
        DataCopyPad(yGm[startOffset + loopIdx * seqLen * bufferNdSizeAll + start], y[loopIdx * numHeads * headDimAlign],
                    dataCopyParams);
    }
    outQueY.FreeTensor(y);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSPad<T>::Process()
{
    LocalTensor<int32_t> gatherOffset = gatherOffsetBuf.Get<int32_t>();
    SetGatherSrcOffset(gatherOffset, headDimAlign * numHeads, static_cast<int32_t>(sizeof(float)));
    LocalTensor<uint32_t> gatherOffsetCast = gatherOffset.ReinterpretCast<uint32_t>();

    if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
        for (uint32_t loop = 0; loop < ubCalcSeqLoop; ++loop) {
            for (uint32_t j = 0; j < (ubCalcBTail == 0 ? ubCalcBLoop : ubCalcBLoop - 1); ++j) {
                ComputeCastFp32(loop, j, gatherOffsetCast, ubCalcBNum);
                CopyOut(loop, j, ubCalcBNum);
            }
            if (ubCalcBTail != 0) {
                ComputeCastFp32(loop, ubCalcBLoop - 1, gatherOffsetCast, ubCalcBTail);
                CopyOut(loop, ubCalcBLoop - 1, ubCalcBTail);
            }
        }
    } else {
        for (uint32_t loop = 0; loop < ubCalcSeqLoop; ++loop) {
            for (uint32_t j = 0; j < (ubCalcBTail == 0 ? ubCalcBLoop : ubCalcBLoop - 1); ++j) {
                Compute(loop, j, gatherOffsetCast, ubCalcBNum);
                CopyOut(loop, j, ubCalcBNum);
            }
            if (ubCalcBTail != 0) {
                Compute(loop, ubCalcBLoop - 1, gatherOffsetCast, ubCalcBTail);
                CopyOut(loop, ubCalcBLoop - 1, ubCalcBTail);
            }
        }
    }
}

template <typename T>
__aicore__ inline void InterleavedSplitBSPad<T>::Compute(uint32_t seqIdx, uint32_t batchIdx,
                                                         LocalTensor<uint32_t> &gatherOffsetCast, uint32_t calcLen)
{
    uint64_t calcTotalNum = calcLen * numHeads * headDimAlign;

    LocalTensor<T> x = inQueX.AllocTensor<T>();
    CopyInX(x, seqIdx, batchIdx, calcLen);
    LocalTensor<T> cos = inQueCos.AllocTensor<T>();
    CopyInCos(cos, seqIdx, calcLen);

    LocalTensor<T> yTensor = outQueY.AllocTensor<T>();
    Mul(yTensor, x, cos, calcTotalNum);
    for (uint32_t i = 0; i < calcLen; ++i) {
        Gather(x[i * numHeads * headDimAlign], x[i * numHeads * headDimAlign], gatherOffsetCast, 0,
               numHeads * headDimAlign);
    }

    CopyInSin(cos, seqIdx, calcLen);
    Mul(x, x, cos, calcTotalNum);

    inQueCos.FreeTensor(cos);
    InterleavedInversion(x, calcTotalNum);
    Add(yTensor, yTensor, x, calcTotalNum);

    inQueX.FreeTensor(x);
    outQueY.EnQue(yTensor);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSPad<T>::ComputeCastFp32(uint32_t seqIdx, uint32_t batchIdx,
                                                                 LocalTensor<uint32_t> &gatherOffsetCast,
                                                                 uint32_t calcLen)
{
    uint64_t calcTotalNum = calcLen * numHeads * headDimAlign;

    LocalTensor<T> xTensor = inQueX.AllocTensor<T>();
    CopyInX(xTensor, seqIdx, batchIdx, calcLen);
    LocalTensor<float> tmp32BsPadBuf1 = tmpFp32Buf1.Get<float>();
    Cast(tmp32BsPadBuf1, xTensor, RoundMode::CAST_NONE, calcTotalNum);
    inQueX.FreeTensor(xTensor);

    LocalTensor<T> cos = inQueCos.AllocTensor<T>();
    CopyInCos(cos, seqIdx, calcLen);
    LocalTensor<float> tmp32Buf2 = tmpFp32Buf2.Get<float>();
    Cast(tmp32Buf2, cos, RoundMode::CAST_NONE, calcTotalNum);

    LocalTensor<float> tmp32Buf3 = tmpFp32Buf3.Get<float>();
    Mul(tmp32Buf3, tmp32BsPadBuf1, tmp32Buf2, calcTotalNum);

    for (uint32_t i = 0; i < calcLen; ++i) {
        Gather(tmp32BsPadBuf1[i * numHeads * headDimAlign], tmp32BsPadBuf1[i * numHeads * headDimAlign],
               gatherOffsetCast, 0, numHeads * headDimAlign);
    }

    CopyInSin(cos, seqIdx, calcLen);
    Cast(tmp32Buf2, cos, RoundMode::CAST_NONE, calcTotalNum);
    inQueCos.FreeTensor(cos);

    Mul(tmp32BsPadBuf1, tmp32BsPadBuf1, tmp32Buf2, calcTotalNum);
    InterleavedInversion(tmp32BsPadBuf1, calcTotalNum);
    Add(tmp32Buf3, tmp32Buf3, tmp32BsPadBuf1, calcTotalNum);

    LocalTensor<T> y = outQueY.AllocTensor<T>();
    Cast(y, tmp32Buf3, RoundMode::CAST_RINT, calcTotalNum);
    outQueY.EnQue(y);
}

} // namespace RotateInterleavedN

#endif // ROTATE_INTERLEAVED_SPLIT_BS_PAD_H
