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
 * \file rotate_half_bf16.h
 * \brief
 */
#ifndef ROTATE_HALF_BF16_H
#define ROTATE_HALF_BF16_H

#include "rotate_half_base.h"

namespace RotateHalfN {
using namespace AscendC;

template <typename OriT, typename CmpT>
class RotateHalfBf16 : public RotateHalfBase<OriT, CmpT> {
public:
    __aicore__ inline RotateHalfBf16(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                const RotaryPositionEmbeddingTilingData &tilingData);
    __aicore__ inline void Process();

protected:
    TPipe pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueX;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueCos;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueSin;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueueY;
    TBuf<TPosition::VECCALC> xBuf;
    TBuf<TPosition::VECCALC> xNewBuf;
    TBuf<TPosition::VECCALC> cosBuf;
    TBuf<TPosition::VECCALC> sinBuf;
    GlobalTensor<OriT> xGm;
    GlobalTensor<OriT> cosGm;
    GlobalTensor<OriT> sinGm;
    GlobalTensor<OriT> yGm;

    __aicore__ inline void NormalProcessBf16();
    __aicore__ inline void RB1sdProcessBf16();
    __aicore__ inline void BndProcessBf16();
    __aicore__ inline void SingleStepProcess(uint32_t progress, uint32_t sLines, uint64_t copyLength,
                                             uint64_t calcLength);
    __aicore__ inline void RB1sdSingleStepProcessBf16(uint32_t progress, uint64_t sLines, uint64_t xBatchStartOffset,
                                                      uint64_t rBatchStartOffset, uint64_t copyLength,
                                                      uint64_t calcLength);
    __aicore__ inline void Compute(LocalTensor<CmpT> &cos, LocalTensor<CmpT> &sin, uint32_t sLines,
                                   uint32_t calcLength);
    __aicore__ inline void CopyInR(uint64_t rStartOffset, uint16_t sLines, uint32_t copyLength);
    __aicore__ inline void CopyInX(uint64_t xStartOffset, uint16_t sLines, uint32_t copyLength);
    __aicore__ inline void CopyOut(uint64_t yOffset, uint16_t sLines, uint32_t copyLength);
};

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBf16<OriT, CmpT>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                        const RotaryPositionEmbeddingTilingData &tilingData)
{
    this->BaseMemberInit(tilingData);

    xGm.SetGlobalBuffer((__gm__ OriT *)x + this->xOffset + this->xCoreOffset * this->coreRelativeIdx,
                        this->xAllocLength);
    yGm.SetGlobalBuffer((__gm__ OriT *)y + this->xOffset + this->xCoreOffset * this->coreRelativeIdx,
                        this->xAllocLength);
    cosGm.SetGlobalBuffer((__gm__ OriT *)cos + this->rOffset + this->rCoreOffset * this->coreRelativeIdx,
                          this->rAllocLength);
    sinGm.SetGlobalBuffer((__gm__ OriT *)sin + this->rOffset + this->rCoreOffset * this->coreRelativeIdx,
                          this->rAllocLength);

    pipe.InitBuffer(inQueueX, DOUBLE_BUFFER, this->storePadDataLength * sizeof(OriT));
    pipe.InitBuffer(outQueueY, DOUBLE_BUFFER, this->storePadDataLength * sizeof(OriT));
    pipe.InitBuffer(inQueueCos, DOUBLE_BUFFER, this->storePadDataLength * sizeof(OriT));
    pipe.InitBuffer(inQueueSin, DOUBLE_BUFFER, this->storePadDataLength * sizeof(OriT));

    pipe.InitBuffer(xBuf, this->storePadDataLength * sizeof(CmpT));
    pipe.InitBuffer(cosBuf, this->storePadDataLength * sizeof(CmpT));
    pipe.InitBuffer(sinBuf, this->storePadDataLength * sizeof(CmpT));
    pipe.InitBuffer(xNewBuf, this->storePadDataLength * sizeof(CmpT));
}

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBf16<OriT, CmpT>::Process()
{
    if (this->layout == LAYOUT_BNSD || this->layout == LAYOUT_BSND || this->layout == LAYOUT_SBND ||
        this->layout == LAYOUT_NO_BROADCAST) {
        NormalProcessBf16();
    } else if (this->layout == LAYOUT_R_B1SD) {
        RB1sdProcessBf16();
    } else if (this->layout == LAYOUT_BND) {
        BndProcessBf16();
    }
}

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBf16<OriT, CmpT>::NormalProcessBf16()
{
    for (uint32_t progress = 0; progress < this->ubLoop; progress++) {
        SingleStepProcess(progress, this->storeSLines, this->storeDataLength, this->storePadDataLength);
    }
    if (this->ubLast > 0) {
        SingleStepProcess(this->ubLoop, this->ubLast, this->ubLastDataLength, this->ubLastPadDataLength);
    }
}

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBf16<OriT, CmpT>::RB1sdProcessBf16()
{
    uint64_t totalSdLength = this->totalSLines * this->dLength;
    uint64_t totalNsdLength = totalSdLength * this->bcSecondDim;
    uint64_t xBatchOffset, rBatchOffset;
    for (uint32_t loop = 0; loop < this->bcFirstDim; loop++) {
        xBatchOffset = loop * totalNsdLength;
        rBatchOffset = loop * totalSdLength;
        for (uint32_t progress = 0; progress < this->ubLoop; progress++) {
            RB1sdSingleStepProcessBf16(progress, this->storeSLines, xBatchOffset, rBatchOffset, this->storeDataLength,
                                       this->storePadDataLength);
        }
        if (this->ubLast > 0) {
            RB1sdSingleStepProcessBf16(this->ubLoop, this->ubLast, xBatchOffset, rBatchOffset, this->ubLastDataLength,
                                       this->ubLastPadDataLength);
        }
    }
}

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBf16<OriT, CmpT>::BndProcessBf16()
{
    CopyInR(0, 1, this->dLength);
    LocalTensor<OriT> cosLocal = inQueueCos.DeQue<OriT>();
    LocalTensor<OriT> sinLocal = inQueueSin.DeQue<OriT>();
    LocalTensor<CmpT> cosFp32 = cosBuf.Get<CmpT>();
    LocalTensor<CmpT> sinFp32 = sinBuf.Get<CmpT>();

    Cast(cosFp32, cosLocal, RoundMode::CAST_NONE, this->dPadLength);
    Cast(sinFp32, sinLocal, RoundMode::CAST_NONE, this->dPadLength);
    inQueueCos.FreeTensor<OriT>(cosLocal);
    inQueueSin.FreeTensor<OriT>(sinLocal);
    Muls(sinFp32, sinFp32, (CmpT)(-1.0), this->halfDPadLength);
    uint32_t broadcastLines = this->ubLoop > 0 ? this->storeSLines - 1 : this->ubLast - 1;
    if (broadcastLines > 0) {
        this->RBroadCast(cosFp32, sinFp32, broadcastLines);
    }
    uint64_t xOffset;
    for (uint32_t progress = 0; progress < this->ubLoop; progress++) {
        xOffset = progress * this->storeDataLength;
        CopyInX(xOffset, this->storeSLines, this->storeDataLength);
        Compute(cosFp32, sinFp32, this->storeSLines, this->storePadDataLength);
        CopyOut(xOffset, this->storeSLines, this->storeDataLength);
    }
    if (this->ubLast > 0) {
        xOffset = this->ubLoop * this->storeDataLength;
        CopyInX(xOffset, this->ubLast, this->ubLastDataLength);
        Compute(cosFp32, sinFp32, this->ubLast, this->ubLastPadDataLength);
        CopyOut(xOffset, this->ubLast, this->ubLastDataLength);
    }
}

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBf16<OriT, CmpT>::SingleStepProcess(uint32_t progress, uint32_t sLines,
                                                                     uint64_t copyLength, uint64_t calcLength)
{
    uint64_t xOffset, rOffset, bnLoopXStartOffset, progressOffset, batchOffset;
    rOffset = progress * this->storeDataLength;
    CopyInR(rOffset, sLines, copyLength);
    LocalTensor<OriT> cosLocal = inQueueCos.DeQue<OriT>();
    LocalTensor<OriT> sinLocal = inQueueSin.DeQue<OriT>();
    LocalTensor<CmpT> cosFp32 = cosBuf.Get<CmpT>();
    LocalTensor<CmpT> sinFp32 = sinBuf.Get<CmpT>();

    Cast(cosFp32, cosLocal, RoundMode::CAST_NONE, calcLength);
    Cast(sinFp32, sinLocal, RoundMode::CAST_NONE, calcLength);
    inQueueCos.FreeTensor<OriT>(cosLocal);
    inQueueSin.FreeTensor<OriT>(sinLocal);
    this->SinCompute(sinFp32, sLines);

    if (this->layout == LAYOUT_BNSD) {
        uint64_t totalSdSize = this->totalSLines * this->dLength;
        bnLoopXStartOffset = progress * this->storeDataLength;
        for (uint32_t bnLoop = 0; bnLoop < this->bnSize; bnLoop++) {
            xOffset = bnLoopXStartOffset + bnLoop * totalSdSize;
            CopyInX(xOffset, sLines, copyLength);
            Compute(cosFp32, sinFp32, sLines, calcLength);
            CopyOut(xOffset, sLines, copyLength);
        }
    } else if (this->layout == LAYOUT_BSND) {
        uint64_t totalSndSize = this->totalSLines * this->ndSize;
        progressOffset = progress * this->bcSecondDim * this->storeDataLength;
        for (uint32_t batchLoop = 0; batchLoop < this->bcFirstDim; batchLoop++) {
            batchOffset = batchLoop * totalSndSize;
            for (uint32_t nLoop = 0; nLoop < this->bcSecondDim; nLoop++) {
                xOffset = nLoop * this->dLength + batchOffset + progressOffset;
                CopyInX(xOffset, sLines, copyLength);
                Compute(cosFp32, sinFp32, sLines, calcLength);
                CopyOut(xOffset, sLines, copyLength);
            }
        }
    } else if (this->layout == LAYOUT_SBND) {
        bnLoopXStartOffset = progress * this->storeDataLength * this->bnSize;
        for (uint32_t bnLoop = 0; bnLoop < this->bnSize; bnLoop++) {
            xOffset = bnLoopXStartOffset + bnLoop * this->dLength;
            CopyInX(xOffset, sLines, copyLength);
            Compute(cosFp32, sinFp32, sLines, calcLength);
            CopyOut(xOffset, sLines, copyLength);
        }
    } else if (this->layout == LAYOUT_NO_BROADCAST) {
        CopyInX(rOffset, sLines, copyLength);
        Compute(cosFp32, sinFp32, sLines, calcLength);
        CopyOut(rOffset, sLines, copyLength);
    }
}

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBf16<OriT, CmpT>::RB1sdSingleStepProcessBf16(uint32_t progress, uint64_t sLines,
                                                                              uint64_t xBatchStartOffset,
                                                                              uint64_t rBatchStartOffset,
                                                                              uint64_t copyLength, uint64_t calcLength)
{
    CopyInR(progress * this->storeDataLength + rBatchStartOffset, sLines, copyLength);
    LocalTensor<OriT> cosLocal = inQueueCos.DeQue<OriT>();
    LocalTensor<OriT> sinLocal = inQueueSin.DeQue<OriT>();
    LocalTensor<CmpT> cosFp32 = cosBuf.Get<CmpT>();
    LocalTensor<CmpT> sinFp32 = sinBuf.Get<CmpT>();

    Cast(cosFp32, cosLocal, RoundMode::CAST_NONE, calcLength);
    Cast(sinFp32, sinLocal, RoundMode::CAST_NONE, calcLength);
    inQueueCos.FreeTensor<OriT>(cosLocal);
    inQueueSin.FreeTensor<OriT>(sinLocal);
    this->SinCompute(sinFp32, sLines);

    uint64_t xOffset, progressXOffset;
    progressXOffset = progress * this->storeDataLength + xBatchStartOffset;
    for (uint32_t nLoop = 0; nLoop < this->bcSecondDim; nLoop++) {
        xOffset = nLoop * this->totalSLines * this->dLength + progressXOffset;
        CopyInX(xOffset, sLines, copyLength);
        Compute(cosFp32, sinFp32, sLines, calcLength);
        CopyOut(xOffset, sLines, copyLength);
    }
}

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBf16<OriT, CmpT>::CopyInR(uint64_t rStartOffset, uint16_t sLines, uint32_t copyLength)
{
    LocalTensor<OriT> sinLocal = inQueueSin.AllocTensor<OriT>();
    LocalTensor<OriT> cosLocal = inQueueCos.AllocTensor<OriT>();
    if (this->isAligned == true) {
        DataCopy(sinLocal, sinGm[rStartOffset], copyLength);
        DataCopy(cosLocal, cosGm[rStartOffset], copyLength);
    } else {
        DataCopyExtParams copyParams{(uint16_t)(2 * sLines), // blockCount
                                     this->halfDBytes,       // blockLen
                                     0,                      // srcStride(bytes)
                                     0,                      // dstStride(block)
                                     0};
        DataCopyPad(sinLocal, sinGm[rStartOffset], copyParams, this->noPadParams);
        DataCopyPad(cosLocal, cosGm[rStartOffset], copyParams, this->noPadParams);
    }
    inQueueSin.EnQue(sinLocal);
    inQueueCos.EnQue(cosLocal);
}

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBf16<OriT, CmpT>::CopyInX(uint64_t xStartOffset, uint16_t storeSLines,
                                                           uint32_t copyLength)
{
    LocalTensor<OriT> xLocal = inQueueX.AllocTensor<OriT>();
    DataCopyExtParams copyParams;

    if (this->isAligned == true) {
        if (this->layout == LAYOUT_BNSD || this->layout == LAYOUT_NO_BROADCAST || this->layout == LAYOUT_BND ||
            this->layout == LAYOUT_R_B1SD) {
            DataCopy(xLocal, xGm[xStartOffset], copyLength);
        } else if (this->layout == LAYOUT_BSND) {
            copyParams.blockCount = storeSLines;
            copyParams.blockLen = this->dBytes;
            copyParams.srcStride = (this->bcSecondDim - 1) * this->dBytes;
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, xGm[xStartOffset], copyParams, this->noPadParams);
        } else if (this->layout == LAYOUT_SBND) {
            copyParams.blockCount = storeSLines;
            copyParams.blockLen = this->dBytes;
            copyParams.srcStride = (this->bnSize - 1) * this->dBytes;
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, xGm[xStartOffset], copyParams, this->noPadParams);
        }
    } else {
        if (this->layout == LAYOUT_BNSD || this->layout == LAYOUT_NO_BROADCAST || this->layout == LAYOUT_BND ||
            this->layout == LAYOUT_R_B1SD) {
            copyParams.blockCount = (uint16_t)(2 * storeSLines);
            copyParams.blockLen = this->halfDBytes;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, xGm[xStartOffset], copyParams, this->noPadParams);
        } else if (this->layout == LAYOUT_BSND) {
            copyParams.blockCount = storeSLines;
            copyParams.blockLen = this->halfDBytes;
            copyParams.srcStride = (2 * this->bcSecondDim - 1) * this->halfDBytes;
            copyParams.dstStride = this->halfDPadBlocks;
            DataCopyPad(xLocal, xGm[xStartOffset], copyParams, this->noPadParams);
            DataCopyPad(xLocal[this->halfDPadLength], xGm[xStartOffset + this->halfDLength], copyParams,
                        this->noPadParams);
        } else if (this->layout == LAYOUT_SBND) {
            copyParams.blockCount = storeSLines;
            copyParams.blockLen = this->halfDBytes;
            copyParams.srcStride = (this->bnSize * 2 - 1) * this->halfDBytes;
            copyParams.dstStride = this->halfDPadBlocks;
            DataCopyPad(xLocal, xGm[xStartOffset], copyParams, this->noPadParams);
            DataCopyPad(xLocal[this->halfDPadLength], xGm[xStartOffset + this->halfDLength], copyParams,
                        this->noPadParams);
        }
    }
    inQueueX.EnQue(xLocal);
}

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBf16<OriT, CmpT>::CopyOut(uint64_t yOffset, uint16_t storeSLines, uint32_t copyLength)
{
    LocalTensor<OriT> yLocal = outQueueY.DeQue<OriT>();
    DataCopyExtParams copyParams;

    if (this->isAligned == true) {
        copyParams.blockCount = storeSLines;
        copyParams.blockLen = this->dBytes;
        copyParams.srcStride = 0;
        if (this->layout == LAYOUT_BNSD || this->layout == LAYOUT_NO_BROADCAST || this->layout == LAYOUT_BND ||
            this->layout == LAYOUT_R_B1SD) {
            DataCopy(yGm[yOffset], yLocal, copyLength);
        } else if (this->layout == LAYOUT_BSND) {
            copyParams.dstStride = (this->bcSecondDim - 1) * this->dBytes;
            DataCopyPad(yGm[yOffset], yLocal, copyParams);
        } else if (this->layout == LAYOUT_SBND) {
            copyParams.dstStride = (this->bnSize - 1) * this->dBytes;
            DataCopyPad(yGm[yOffset], yLocal, copyParams);
        }
    } else {
        if (this->layout == LAYOUT_BNSD || this->layout == LAYOUT_NO_BROADCAST || this->layout == LAYOUT_BND ||
            this->layout == LAYOUT_R_B1SD) {
            copyParams.blockCount = (uint16_t)(2 * storeSLines);
            copyParams.blockLen = this->halfDBytes;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(yGm[yOffset], yLocal, copyParams);
        } else if (this->layout == LAYOUT_BSND) {
            copyParams.blockCount = storeSLines;
            copyParams.blockLen = this->halfDBytes;
            copyParams.srcStride = this->halfDPadBlocks;
            copyParams.dstStride = (2 * this->bcSecondDim - 1) * this->halfDBytes;
            DataCopyPad(yGm[yOffset], yLocal, copyParams);
            DataCopyPad(yGm[yOffset + this->halfDLength], yLocal[this->halfDPadLength], copyParams);
        } else if (this->layout == LAYOUT_SBND) {
            copyParams.blockCount = storeSLines;
            copyParams.blockLen = this->halfDBytes;
            copyParams.srcStride = this->halfDPadBlocks;
            copyParams.dstStride = (2 * this->bnSize - 1) * this->halfDBytes;
            DataCopyPad(yGm[yOffset], yLocal, copyParams);
            DataCopyPad(yGm[yOffset + this->halfDLength], yLocal[this->halfDPadLength], copyParams);
        }
    }
    outQueueY.FreeTensor(yLocal);
}

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBf16<OriT, CmpT>::Compute(LocalTensor<CmpT> &cos, LocalTensor<CmpT> &sin,
                                                           uint32_t sLines, uint32_t calcLength)
{
    LocalTensor<OriT> xLocal = inQueueX.DeQue<OriT>();
    LocalTensor<OriT> yLocal = outQueueY.AllocTensor<OriT>();
    LocalTensor<CmpT> xFp32 = xBuf.Get<CmpT>();
    LocalTensor<CmpT> xNewFp32 = xNewBuf.Get<CmpT>();

    Cast(xFp32, xLocal, RoundMode::CAST_NONE, calcLength);
    inQueueX.FreeTensor(xLocal);

    this->XNewCopy(xFp32, xNewFp32, sLines);
    this->ComputeInner(xFp32, xNewFp32, cos, sin, calcLength);
    Cast(yLocal, xNewFp32, RoundMode::CAST_RINT, calcLength);
    outQueueY.EnQue(yLocal);
}
} // namespace RotateHalfN
#endif // ROTATE_HALF_BF16_H
