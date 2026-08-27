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
 * \file rotate_half.h
 * \brief
 */
#ifndef ROTATE_HALF_H
#define ROTATE_HALF_H

#include "rotate_half_base.h"

namespace RotateHalfN {
using namespace AscendC;

template <typename T>
class RotateHalf : public RotateHalfBase<T, T> {
public:
    __aicore__ inline RotateHalf(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                const RotaryPositionEmbeddingTilingData &tilingData);
    __aicore__ inline void Process();

protected:
    TPipe pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueX;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueCos;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueueSin;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueueY;
    GlobalTensor<T> xGm;
    GlobalTensor<T> cosGm;
    GlobalTensor<T> sinGm;
    GlobalTensor<T> yGm;

    __aicore__ inline void NormalProcess();
    __aicore__ inline void RB1sdProcess();
    __aicore__ inline void BndProcess();
    __aicore__ inline void SingleStepProcess(uint32_t progress, uint32_t sLines, uint64_t copyLength,
                                             uint64_t calcLength);
    __aicore__ inline void RB1sdSingleStepProcess(uint32_t progress, uint32_t sLines, uint64_t xBatchStartOffset,
                                                  uint64_t rBatchStartOffset, uint64_t copyLength, uint64_t calcLength);
    __aicore__ inline void Compute(LocalTensor<T> &cos, LocalTensor<T> &sin, uint32_t sLines, uint32_t calcLength);
    __aicore__ inline void CopyInR(uint64_t rStartOffset, uint16_t sLines, uint32_t copyLength);
    __aicore__ inline void CopyInX(uint64_t xStartOffset, uint16_t sLines, uint32_t copyLength);
    __aicore__ inline void CopyOut(uint64_t yStartOffset, uint16_t sLines, uint32_t copyLength);
};

template <typename T>
__aicore__ inline void RotateHalf<T>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                           const RotaryPositionEmbeddingTilingData &tilingData)
{
    this->BaseMemberInit(tilingData);

    xGm.SetGlobalBuffer((__gm__ T *)x + this->xOffset + this->xCoreOffset * this->coreRelativeIdx, this->xAllocLength);
    yGm.SetGlobalBuffer((__gm__ T *)y + this->xOffset + this->xCoreOffset * this->coreRelativeIdx, this->xAllocLength);
    cosGm.SetGlobalBuffer((__gm__ T *)cos + this->rOffset + this->rCoreOffset * this->coreRelativeIdx,
                          this->rAllocLength);
    sinGm.SetGlobalBuffer((__gm__ T *)sin + this->rOffset + this->rCoreOffset * this->coreRelativeIdx,
                          this->rAllocLength);

    pipe.InitBuffer(inQueueX, DOUBLE_BUFFER, this->storePadDataLength * sizeof(T));
    pipe.InitBuffer(outQueueY, DOUBLE_BUFFER, this->storePadDataLength * sizeof(T));
    pipe.InitBuffer(inQueueCos, DOUBLE_BUFFER, this->storePadDataLength * sizeof(T));
    pipe.InitBuffer(inQueueSin, DOUBLE_BUFFER, this->storePadDataLength * sizeof(T));
}

template <typename T>
__aicore__ inline void RotateHalf<T>::Process()
{
    if (this->layout == LAYOUT_BNSD || this->layout == LAYOUT_BSND || this->layout == LAYOUT_SBND ||
        this->layout == LAYOUT_NO_BROADCAST) {
        NormalProcess();
    } else if (this->layout == LAYOUT_R_B1SD) {
        RB1sdProcess();
    } else if (this->layout == LAYOUT_BND) {
        BndProcess();
    }
}

template <typename T>
__aicore__ inline void RotateHalf<T>::NormalProcess()
{
    for (uint32_t progress = 0; progress < this->ubLoop; progress++) {
        SingleStepProcess(progress, this->storeSLines, this->storeDataLength, this->storePadDataLength);
    }
    if (this->ubLast > 0) {
        SingleStepProcess(this->ubLoop, this->ubLast, this->ubLastDataLength, this->ubLastPadDataLength);
    }
}

template <typename T>
__aicore__ inline void RotateHalf<T>::RB1sdProcess()
{
    uint64_t totalSdLength = this->totalSLines * this->dLength;
    uint64_t totalNsdLength = totalSdLength * this->bcSecondDim;
    uint64_t xBatchOffset, rBatchOffset;
    for (uint32_t batchLoop = 0; batchLoop < this->bcFirstDim; batchLoop++) {
        xBatchOffset = batchLoop * totalNsdLength;
        rBatchOffset = batchLoop * totalSdLength;
        for (uint32_t progress = 0; progress < this->ubLoop; progress++) {
            RB1sdSingleStepProcess(progress, this->storeSLines, xBatchOffset, rBatchOffset, this->storeDataLength,
                                   this->storePadDataLength);
        }
        if (this->ubLast > 0) {
            RB1sdSingleStepProcess(this->ubLoop, this->ubLast, xBatchOffset, rBatchOffset, this->ubLastDataLength,
                                   this->ubLastPadDataLength);
        }
    }
}

template <typename T>
__aicore__ inline void RotateHalf<T>::BndProcess()
{
    CopyInR(0, 1, this->dLength);
    LocalTensor<T> cosLocal = inQueueCos.DeQue<T>();
    LocalTensor<T> sinLocal = inQueueSin.DeQue<T>();
    Muls(sinLocal, sinLocal, (T)(-1.0), this->halfDPadLength);
    uint32_t broadcastLines = this->ubLoop > 0 ? this->storeSLines - 1 : this->ubLast - 1;
    if (broadcastLines > 0) {
        this->RBroadCast(cosLocal, sinLocal, broadcastLines);
    }
    uint64_t xOffset;
    for (uint32_t progress = 0; progress < this->ubLoop; progress++) {
        xOffset = progress * this->storeDataLength;
        CopyInX(xOffset, this->storeSLines, this->storeDataLength);
        Compute(cosLocal, sinLocal, this->storeSLines, this->storePadDataLength);
        CopyOut(xOffset, this->storeSLines, this->storeDataLength);
    }
    if (this->ubLast > 0) {
        xOffset = this->ubLoop * this->storeDataLength;
        CopyInX(xOffset, this->ubLast, this->ubLastDataLength);
        Compute(cosLocal, sinLocal, this->ubLast, this->ubLastPadDataLength);
        CopyOut(xOffset, this->ubLast, this->ubLastDataLength);
    }
    inQueueCos.FreeTensor(cosLocal);
    inQueueSin.FreeTensor(sinLocal);
}

template <typename T>
__aicore__ inline void RotateHalf<T>::SingleStepProcess(uint32_t progress, uint32_t sLines, uint64_t copyLength,
                                                        uint64_t calcLength)
{
    uint64_t xOffset, rOffset, bnLoopXStartOffset, progressOffset, batchOffset;
    rOffset = progress * this->storeDataLength;
    CopyInR(rOffset, sLines, copyLength);
    LocalTensor<T> cosLocal = inQueueCos.DeQue<T>();
    LocalTensor<T> sinLocal = inQueueSin.DeQue<T>();
    this->SinCompute(sinLocal, sLines);

    if (this->layout == LAYOUT_BNSD) {
        uint64_t totalSdSize = this->totalSLines * this->dLength;
        bnLoopXStartOffset = progress * this->storeDataLength;
        for (uint32_t bnLoop = 0; bnLoop < this->bnSize; bnLoop++) {
            xOffset = bnLoopXStartOffset + bnLoop * totalSdSize;
            CopyInX(xOffset, sLines, copyLength);
            Compute(cosLocal, sinLocal, sLines, calcLength);
            CopyOut(xOffset, sLines, copyLength);
        }
    } else if (this->layout == LAYOUT_BSND) {
        uint64_t totalSndSize = this->totalSLines * this->ndSize;
        progressOffset = progress * this->bcSecondDim * this->storeDataLength;
        for (uint32_t bLoop = 0; bLoop < this->bcFirstDim; bLoop++) {
            batchOffset = bLoop * totalSndSize;
            for (uint32_t nLoop = 0; nLoop < this->bcSecondDim; nLoop++) {
                xOffset = nLoop * this->dLength + batchOffset + progressOffset;
                CopyInX(xOffset, sLines, copyLength);
                Compute(cosLocal, sinLocal, sLines, calcLength);
                CopyOut(xOffset, sLines, copyLength);
            }
        }
    } else if (this->layout == LAYOUT_SBND) {
        bnLoopXStartOffset = progress * this->storeDataLength * this->bnSize;
        for (uint32_t bnLoop = 0; bnLoop < this->bnSize; bnLoop++) {
            xOffset = bnLoopXStartOffset + bnLoop * this->dLength;
            CopyInX(xOffset, sLines, copyLength);
            Compute(cosLocal, sinLocal, sLines, calcLength);
            CopyOut(xOffset, sLines, copyLength);
        }
    } else if (this->layout == LAYOUT_NO_BROADCAST) {
        CopyInX(rOffset, sLines, copyLength);
        Compute(cosLocal, sinLocal, sLines, calcLength);
        CopyOut(rOffset, sLines, copyLength);
    }
    inQueueCos.FreeTensor<T>(cosLocal);
    inQueueSin.FreeTensor<T>(sinLocal);
}

template <typename T>
__aicore__ inline void RotateHalf<T>::RB1sdSingleStepProcess(uint32_t progress, uint32_t sLines,
                                                             uint64_t xBatchStartOffset, uint64_t rBatchStartOffset,
                                                             uint64_t copyLength, uint64_t calcLength)
{
    CopyInR(progress * this->storeDataLength + rBatchStartOffset, sLines, copyLength);
    LocalTensor<T> cosLocal = inQueueCos.DeQue<T>();
    LocalTensor<T> sinLocal = inQueueSin.DeQue<T>();
    this->SinCompute(sinLocal, sLines);

    uint64_t xOffset, progressXOffset;
    progressXOffset = progress * this->storeDataLength + xBatchStartOffset;
    for (uint32_t nLoop = 0; nLoop < this->bcSecondDim; nLoop++) {
        xOffset = nLoop * this->totalSLines * this->dLength + progressXOffset;
        CopyInX(xOffset, sLines, copyLength);
        Compute(cosLocal, sinLocal, sLines, calcLength);
        CopyOut(xOffset, sLines, copyLength);
    }
    inQueueCos.FreeTensor<T>(cosLocal);
    inQueueSin.FreeTensor<T>(sinLocal);
}

template <typename T>
__aicore__ inline void RotateHalf<T>::CopyInR(uint64_t rStartOffset, uint16_t sLines, uint32_t copyLength)
{
    LocalTensor<T> cosLocal = inQueueCos.AllocTensor<T>();
    LocalTensor<T> sinLocal = inQueueSin.AllocTensor<T>();
    if (this->isAligned == true) {
        DataCopy(cosLocal, cosGm[rStartOffset], copyLength);
        DataCopy(sinLocal, sinGm[rStartOffset], copyLength);
    } else {
        DataCopyExtParams copyParams{(uint16_t)(2 * sLines), // blockCount
                                     this->halfDBytes,       // blockLen
                                     0,                      // srcStride(bytes)
                                     0,                      // dstStride(block)
                                     0};
        DataCopyPad(cosLocal, cosGm[rStartOffset], copyParams, this->noPadParams);
        DataCopyPad(sinLocal, sinGm[rStartOffset], copyParams, this->noPadParams);
    }
    inQueueCos.EnQue(cosLocal);
    inQueueSin.EnQue(sinLocal);
}

template <typename T>
__aicore__ inline void RotateHalf<T>::CopyInX(uint64_t xStartOffset, uint16_t sLines, uint32_t copyLength)
{
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopyExtParams copyParams;

    if (this->isAligned == true) {
        if (this->layout == LAYOUT_BNSD || this->layout == LAYOUT_NO_BROADCAST || this->layout == LAYOUT_BND ||
            this->layout == LAYOUT_R_B1SD) {
            DataCopy(xLocal, xGm[xStartOffset], copyLength);
        } else if (this->layout == LAYOUT_BSND) {
            copyParams.blockCount = sLines;
            copyParams.blockLen = this->dBytes;
            copyParams.srcStride = (this->bcSecondDim - 1) * this->dBytes;
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, xGm[xStartOffset], copyParams, this->noPadParams);
        } else if (this->layout == LAYOUT_SBND) {
            copyParams.blockCount = sLines;
            copyParams.blockLen = this->dBytes;
            copyParams.srcStride = (this->bnSize - 1) * this->dBytes;
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, xGm[xStartOffset], copyParams, this->noPadParams);
        }
    } else {
        if (this->layout == LAYOUT_BNSD || this->layout == LAYOUT_NO_BROADCAST || this->layout == LAYOUT_BND ||
            this->layout == LAYOUT_R_B1SD) {
            copyParams.blockCount = (uint16_t)(2 * sLines);
            copyParams.blockLen = this->halfDBytes;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, xGm[xStartOffset], copyParams, this->noPadParams);
        } else if (this->layout == LAYOUT_BSND) {
            copyParams.blockCount = sLines;
            copyParams.blockLen = this->halfDBytes;
            copyParams.srcStride = (2 * this->bcSecondDim - 1) * this->halfDBytes;
            copyParams.dstStride = this->halfDPadBlocks;
            DataCopyPad(xLocal, xGm[xStartOffset], copyParams, this->noPadParams);
            DataCopyPad(xLocal[this->halfDPadLength], xGm[xStartOffset + this->halfDLength], copyParams,
                        this->noPadParams);
        } else if (this->layout == LAYOUT_SBND) {
            copyParams.blockCount = sLines;
            copyParams.blockLen = this->halfDBytes;
            copyParams.srcStride = (2 * this->bnSize - 1) * this->halfDBytes;
            copyParams.dstStride = this->halfDPadBlocks;
            DataCopyPad(xLocal, xGm[xStartOffset], copyParams, this->noPadParams);
            DataCopyPad(xLocal[this->halfDPadLength], xGm[xStartOffset + this->halfDLength], copyParams,
                        this->noPadParams);
        }
    }
    inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void RotateHalf<T>::CopyOut(uint64_t yStartOffset, uint16_t sLines, uint32_t copyLength)
{
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopyExtParams copyParams;

    if (this->isAligned == true) {
        copyParams.blockCount = sLines;
        copyParams.blockLen = this->dBytes;
        copyParams.srcStride = 0;
        if (this->layout == LAYOUT_BNSD || this->layout == LAYOUT_NO_BROADCAST || this->layout == LAYOUT_BND ||
            this->layout == LAYOUT_R_B1SD) {
            DataCopy(yGm[yStartOffset], yLocal, copyLength);
        } else if (this->layout == LAYOUT_BSND) {
            copyParams.dstStride = (this->bcSecondDim - 1) * this->dBytes;
            DataCopyPad(yGm[yStartOffset], yLocal, copyParams);
        } else if (this->layout == LAYOUT_SBND) {
            copyParams.dstStride = (this->bnSize - 1) * this->dBytes;
            DataCopyPad(yGm[yStartOffset], yLocal, copyParams);
        }
    } else {
        if (this->layout == LAYOUT_BNSD || this->layout == LAYOUT_NO_BROADCAST || this->layout == LAYOUT_BND ||
            this->layout == LAYOUT_R_B1SD) {
            copyParams.blockCount = (uint16_t)(2 * sLines);
            copyParams.blockLen = this->halfDBytes;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(yGm[yStartOffset], yLocal, copyParams);
        } else if (this->layout == LAYOUT_BSND) {
            copyParams.blockCount = sLines;
            copyParams.blockLen = this->halfDBytes;
            copyParams.srcStride = this->halfDPadBlocks;
            copyParams.dstStride = (2 * this->bcSecondDim - 1) * this->halfDBytes;
            DataCopyPad(yGm[yStartOffset], yLocal, copyParams);
            DataCopyPad(yGm[yStartOffset + this->halfDLength], yLocal[this->halfDPadLength], copyParams);
        } else if (this->layout == LAYOUT_SBND) {
            copyParams.blockCount = sLines;
            copyParams.blockLen = this->halfDBytes;
            copyParams.srcStride = this->halfDPadBlocks;
            copyParams.dstStride = (2 * this->bnSize - 1) * this->halfDBytes;
            DataCopyPad(yGm[yStartOffset], yLocal, copyParams);
            DataCopyPad(yGm[yStartOffset + this->halfDLength], yLocal[this->halfDPadLength], copyParams);
        }
    }
    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void RotateHalf<T>::Compute(LocalTensor<T> &cos, LocalTensor<T> &sin, uint32_t sLines,
                                              uint32_t calcLength)
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    this->XNewCopy(xLocal, yLocal, sLines);
    this->ComputeInner(xLocal, yLocal, cos, sin, calcLength);
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor<T>(xLocal);
}

} // namespace RotateHalfN
#endif // ROTATE_HALF_H
