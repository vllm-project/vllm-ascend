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
 * \file rotate_interleaved_split_bsn_pad.h
 * \brief
 */
#ifndef ROTATE_INTERLEAVED_SPLIT_BSN_PAD_H
#define ROTATE_INTERLEAVED_SPLIT_BSN_PAD_H
#include "rotate_interleaved_common.h"

namespace RotateInterleavedN {
using namespace AscendC;

template <typename T>
class InterleavedSplitBSNPad {
public:
    __aicore__ inline InterleavedSplitBSNPad(){};
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
    uint64_t ubCalcNNum;
    uint64_t ubCalcNLoop;
    uint64_t ubCalcNTail;
    uint64_t allHeadDim;
    uint64_t start;
    uint64_t ioOffsetAll;
    uint64_t bufferNdSizeAll;

    // init tmp data
    uint32_t alignLen;
    uint32_t headDimAlign;
    uint32_t blockIdx;
    uint32_t ubCalcSeqLoop;
    uint64_t ioOffset;
    uint64_t triOffset;
    uint64_t bufferBsndSize;
    uint64_t bufferSdSize;
    uint64_t bufferNdSize;
    uint64_t bufferLenSize;
    uint64_t gatherOffsetLenSize;

    __aicore__ inline void InitData(const RopeRegbaseTilingData *tiling);
    __aicore__ inline void CopyInX(LocalTensor<T> &x, uint32_t batchIdx, uint32_t seqIdx, uint32_t numHeadsIdx,
                                   uint32_t calcLen);
    __aicore__ inline void CopyInCos(LocalTensor<T> &cos, uint32_t seqIdx, uint32_t calcLen);
    __aicore__ inline void CopyInSin(LocalTensor<T> &sin, uint32_t seqIdx, uint32_t calcLen);
    __aicore__ inline void CopyOut(uint32_t batchIdx, uint32_t seqIdx, uint32_t numHeadsIdx, uint32_t calcLen);
    __aicore__ inline void Compute(uint32_t batchIdx, uint32_t seqIdx, uint32_t numHeadsIdx,
                                   LocalTensor<uint32_t> &gatherOffsetCast, uint32_t calcLen);
    __aicore__ inline void ComputeCastFp32(uint32_t batchIdx, uint32_t seqIdx, uint32_t numHeadsIdx,
                                           LocalTensor<uint32_t> &gatherOffsetCast, uint32_t calcLen);
};

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
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

    bufferLenSize = ubCalcNNum * headDimAlign * sizeof(T);
    pipe->InitBuffer(inQueX, BUFFER_NUM, bufferLenSize);
    pipe->InitBuffer(inQueCos, BUFFER_NUM, bufferLenSize);
    pipe->InitBuffer(outQueY, BUFFER_NUM, bufferLenSize);

    if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
        bufferLenSize = ubCalcNNum * headDimAlign * sizeof(float);
        pipe->InitBuffer(tmpFp32Buf1, bufferLenSize);
        pipe->InitBuffer(tmpFp32Buf2, bufferLenSize);
        pipe->InitBuffer(tmpFp32Buf3, bufferLenSize);
    }

    gatherOffsetLenSize = ubCalcNNum * headDimAlign * sizeof(int32_t);
    pipe->InitBuffer(gatherOffsetBuf, gatherOffsetLenSize);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::InitData(const RopeRegbaseTilingData *tiling)
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
    ubCalcNNum = tiling_->ubCalcNNum;
    ubCalcNLoop = tiling_->ubCalcNLoop;
    ubCalcNTail = tiling_->ubCalcNTail;
    allHeadDim = tiling_->allHeadDim;
    start = tiling_->start;

    alignLen = (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) ? ALIGN_16 : ALIGN_32;
    headDimAlign = (headDim + alignLen - 1) / alignLen * alignLen;
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::CopyInX(LocalTensor<T> &xTensor, uint32_t batchIdx, uint32_t seqIdx,
                                                          uint32_t numHeadsIdx, uint32_t calcLen)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = calcLen;
    dataCopyParams.blockLen = headDim * sizeof(T);
    dataCopyParams.srcStride = (allHeadDim - headDim) * sizeof(T);
    dataCopyParams.dstStride = 0;
    DataCopyPad(xTensor,
                xGm[batchIdx * seqLen * bufferNdSizeAll + seqIdx * bufferNdSizeAll + numHeadsIdx * ubCalcNNum * allHeadDim + start],
                dataCopyParams, {false, 0, 0, 0});
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::CopyInCos(LocalTensor<T> &cos, uint32_t seqIdx, uint32_t calcLen)
{
    DataCopyExtParams bsnPadDataCopyTriParams;
    bsnPadDataCopyTriParams.blockCount = 1;
    bsnPadDataCopyTriParams.blockLen = headDim * sizeof(T);
    bsnPadDataCopyTriParams.srcStride = 0;
    bsnPadDataCopyTriParams.dstStride = 0;
    DataCopyPad(cos, cosGm[seqIdx * headDim], bsnPadDataCopyTriParams, {false, 0, 0, 0});
    event_t eventId2MTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId2MTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventId2MTE2ToV);
    BroadCastTriToB1nd(cos, 1, calcLen, headDimAlign);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::CopyInSin(LocalTensor<T> &sinTensor, uint32_t seqIdx,
                                                            uint32_t calcLen)
{
    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);

    DataCopyExtParams bsnPadDataCopyTriParams;
    bsnPadDataCopyTriParams.blockCount = 1;
    bsnPadDataCopyTriParams.blockLen = headDim * sizeof(T);
    bsnPadDataCopyTriParams.srcStride = 0;
    bsnPadDataCopyTriParams.dstStride = 0;
    DataCopyPad(sinTensor, sinGm[seqIdx * headDim], bsnPadDataCopyTriParams, {false, 0, 0, 0});
    event_t eventId3MTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId3MTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventId3MTE2ToV);
    BroadCastTriToB1nd(sinTensor, 1, calcLen, headDimAlign);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::CopyOut(uint32_t batchIdx, uint32_t seqIdx, uint32_t numHeadsIdx,
                                                          uint32_t calcLen)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = calcLen;
    dataCopyParams.blockLen = headDim * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (allHeadDim - headDim) * sizeof(T);
    LocalTensor<T> y = outQueY.DeQue<T>();
    DataCopyPad(yGm[batchIdx * seqLen * bufferNdSizeAll + seqIdx * bufferNdSizeAll + numHeadsIdx * ubCalcNNum * allHeadDim + start], y,
                dataCopyParams);
    outQueY.FreeTensor(y);
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::Process()
{
    LocalTensor<int32_t> gatherOffset = gatherOffsetBuf.Get<int32_t>();
    SetGatherSrcOffset(gatherOffset, ubCalcNNum * headDimAlign, static_cast<int32_t>(sizeof(float)));
    LocalTensor<uint32_t> gatherOffsetCast = gatherOffset.ReinterpretCast<uint32_t>();

    if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            for (uint32_t j = 0; j < ubCalcSeqLoop; ++j) {
                for (uint32_t z = 0; z < (ubCalcNTail == 0 ? ubCalcNLoop : ubCalcNLoop - 1); ++z) {
                    ComputeCastFp32(batch, j, z, gatherOffsetCast, ubCalcNNum);
                    CopyOut(batch, j, z, ubCalcNNum);
                }
                if (ubCalcNTail != 0) {
                    ComputeCastFp32(batch, j, ubCalcNLoop - 1, gatherOffsetCast, ubCalcNTail);
                    CopyOut(batch, j, ubCalcNLoop - 1, ubCalcNTail);
                }
            }
        }
    } else {
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            for (uint32_t j = 0; j < ubCalcSeqLoop; ++j) {
                for (uint32_t z = 0; z < (ubCalcNTail == 0 ? ubCalcNLoop : ubCalcNLoop - 1); ++z) {
                    Compute(batch, j, z, gatherOffsetCast, ubCalcNNum);
                    CopyOut(batch, j, z, ubCalcNNum);
                }
                if (ubCalcNTail != 0) {
                    Compute(batch, j, ubCalcNLoop - 1, gatherOffsetCast, ubCalcNTail);
                    CopyOut(batch, j, ubCalcNLoop - 1, ubCalcNTail);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void InterleavedSplitBSNPad<T>::Compute(uint32_t batchIdx, uint32_t seqIdx, uint32_t numHeadsIdx,
                                                          LocalTensor<uint32_t> &gatherOffsetCast, uint32_t calcLen)
{
    uint64_t calcTotalNum = calcLen * headDimAlign;

    LocalTensor<T> xTensor = inQueX.AllocTensor<T>();
    CopyInX(xTensor, batchIdx, seqIdx, numHeadsIdx, calcLen);

    LocalTensor<T> cos = inQueCos.AllocTensor<T>();
    CopyInCos(cos, seqIdx, calcLen);

    LocalTensor<T> yTensor = outQueY.AllocTensor<T>();
    Mul(yTensor, xTensor, cos, calcTotalNum);

    Gather(xTensor, xTensor, gatherOffsetCast, 0, calcTotalNum);

    CopyInSin(cos, seqIdx, calcLen);

    Mul(xTensor, xTensor, cos, calcTotalNum);
    inQueCos.FreeTensor(cos);
    InterleavedInversion(xTensor, calcTotalNum);
    Add(yTensor, yTensor, xTensor, calcTotalNum);
    inQueX.FreeTensor(xTensor);
    outQueY.EnQue(yTensor);
}

template <typename T>
__aicore__ inline void
InterleavedSplitBSNPad<T>::ComputeCastFp32(uint32_t batchIdx, uint32_t seqIdx, uint32_t numHeadsIdx,
                                           LocalTensor<uint32_t> &gatherOffsetCast, uint32_t calcLen)
{
    uint64_t totalCount = calcLen * headDimAlign;

    LocalTensor<T> xTensor = inQueX.AllocTensor<T>();
    CopyInX(xTensor, batchIdx, seqIdx, numHeadsIdx, calcLen);
    LocalTensor<float> tmp32Buf1 = tmpFp32Buf1.Get<float>();
    Cast(tmp32Buf1, xTensor, RoundMode::CAST_NONE, totalCount);
    inQueX.FreeTensor(xTensor);

    LocalTensor<T> cos = inQueCos.AllocTensor<T>();
    CopyInCos(cos, seqIdx, calcLen);
    LocalTensor<float> tmp32Buf2 = tmpFp32Buf2.Get<float>();
    Cast(tmp32Buf2, cos, RoundMode::CAST_NONE, totalCount);

    LocalTensor<float> tmp32Buf3 = tmpFp32Buf3.Get<float>();
    Mul(tmp32Buf3, tmp32Buf1, tmp32Buf2, totalCount);

    Gather(tmp32Buf1, tmp32Buf1, gatherOffsetCast, 0, totalCount);

    CopyInSin(cos, seqIdx, calcLen);
    Cast(tmp32Buf2, cos, RoundMode::CAST_NONE, totalCount);
    inQueCos.FreeTensor(cos);

    Mul(tmp32Buf1, tmp32Buf1, tmp32Buf2, totalCount);
    InterleavedInversion(tmp32Buf1, totalCount);
    Add(tmp32Buf3, tmp32Buf3, tmp32Buf1, totalCount);

    LocalTensor<T> y = outQueY.AllocTensor<T>();
    Cast(y, tmp32Buf3, RoundMode::CAST_RINT, totalCount);
    outQueY.EnQue(y);
}

} // namespace RotateInterleavedN

#endif // ROTATE_INTERLEAVED_SPLIT_BSN_PAD_H
