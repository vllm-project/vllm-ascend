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
 * \file rotate_half_base.h
 * \brief
 */

#ifndef ROTATE_HALF_BASE_H
#define ROTATE_HALF_BASE_H

#include "kernel_operator.h"

namespace RotateHalfN {
using namespace AscendC;

constexpr uint8_t REPEAT_MAX = 255;
constexpr int32_t SINGLE_BUFFER = 1;
constexpr int32_t DOUBLE_BUFFER = 2;
constexpr uint32_t BYTE_OF_BLOCK = 32;
constexpr uint32_t BYTE_OF_REPEAT = 256;

constexpr uint16_t LAYOUT_BNSD = 1;
constexpr uint16_t LAYOUT_BSND = 2;
constexpr uint16_t LAYOUT_SBND = 3;
constexpr uint16_t LAYOUT_NO_BROADCAST = 4;
constexpr uint16_t LAYOUT_BND = 5;
constexpr uint16_t LAYOUT_R_B1SD = 6;

template <typename OriT, typename CmpT>
class RotateHalfBase {
public:
    __aicore__ inline RotateHalfBase(){};
    __aicore__ inline void BaseMemberInit(const RotaryPositionEmbeddingTilingData &tilingData);

protected:
    __aicore__ inline void GetTilingData(const RotateHalfParams &tiling);
    __aicore__ inline void SinCompute(LocalTensor<CmpT> &sin, uint32_t sLines);
    __aicore__ inline void ComputeInner(LocalTensor<CmpT> &x, LocalTensor<CmpT> &xNew, LocalTensor<CmpT> &cos,
                                        LocalTensor<CmpT> &sin, uint32_t calcLength);
    __aicore__ inline void RBroadCast(LocalTensor<CmpT> &cos, LocalTensor<CmpT> &sin, uint32_t broadcastLines);
    __aicore__ inline void XNewCopy(LocalTensor<CmpT> &x, LocalTensor<CmpT> &xNew, uint16_t sLines);

    bool isAligned;
    uint16_t layout;
    uint64_t gmLength;
    uint64_t bcFirstDim;
    uint64_t bcSecondDim;
    uint64_t dLength;
    uint64_t dPadLength;
    uint64_t halfDLength;
    uint64_t halfDPadLength;
    uint64_t totalSLines;
    uint64_t storeSLines;
    uint64_t storeDataLength;
    uint64_t storePadDataLength;
    uint64_t ubLoop;
    uint64_t ubLast;
    uint64_t formerCoreNum;
    uint64_t tailCoreNum;
    uint64_t formerSLines;
    uint64_t tailSLines;
    uint64_t coreSLines;
    uint64_t xDataLength;
    uint64_t rDataLength;
    uint64_t ubLastDataLength;
    uint64_t ubLastPadDataLength;
    uint64_t xOffset;
    uint64_t rOffset;

    uint8_t repeatStride;
    uint32_t dataEachRepeat;
    uint32_t bnSize;
    uint32_t ndSize;
    uint32_t bndSize;
    uint32_t dBytes;
    uint32_t halfDBytes;
    uint32_t halfDPadBlocks;
    uint32_t innerHalfLoop;
    uint32_t innerHalfLast;
    uint64_t xAllocLength;
    uint64_t rAllocLength;
    uint64_t xCoreOffset;
    uint64_t rCoreOffset;
    uint64_t coreRelativeIdx;

    DataCopyPadExtParams<OriT> noPadParams{false, 0, 0, 0};
};

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBase<OriT, CmpT>::BaseMemberInit(const RotaryPositionEmbeddingTilingData &tilingData)
{
    const RotateHalfParams &tiling = tilingData.rotateHalfParams;
    GetTilingData(tiling);

    // intermediate variable
    repeatStride = dPadLength * sizeof(CmpT) / BYTE_OF_BLOCK;
    dataEachRepeat = BYTE_OF_REPEAT / sizeof(CmpT);
    innerHalfLoop = halfDLength / dataEachRepeat;
    innerHalfLast = halfDLength % dataEachRepeat;
    bnSize = bcFirstDim * bcSecondDim;
    ndSize = bcSecondDim * dLength;
    bndSize = bnSize * dLength;
    dBytes = dLength * sizeof(OriT);
    halfDBytes = halfDLength * sizeof(OriT);
    halfDPadBlocks = halfDPadLength * sizeof(OriT) / BYTE_OF_BLOCK;

    // gm and ub space params
    xAllocLength = gmLength;
    rAllocLength = rDataLength;
    rCoreOffset = rDataLength;
    if (layout == LAYOUT_BNSD || layout == LAYOUT_NO_BROADCAST) {
        xCoreOffset = coreSLines * dLength;
    } else if (layout == LAYOUT_BSND) {
        xCoreOffset = coreSLines * ndSize;
    } else if (layout == LAYOUT_SBND) {
        xCoreOffset = coreSLines * bndSize;
    } else if (layout == LAYOUT_BND) {
        xCoreOffset = xDataLength;
        xAllocLength = xDataLength;
        rCoreOffset = 0;
        rOffset = 0;
    } else if (layout == LAYOUT_R_B1SD) {
        xCoreOffset = coreSLines * dLength;
        rCoreOffset = xCoreOffset;
        rAllocLength = bcFirstDim * totalSLines * dLength;
    }
}

template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBase<OriT, CmpT>::GetTilingData(const RotateHalfParams &tiling)
{
    isAligned = tiling.isAligned == 1;
    layout = tiling.tilingMode;
    gmLength = tiling.gmLength;
    bcFirstDim = tiling.broadcastFirstDim;
    bcSecondDim = tiling.broadcastSecondDim;
    dLength = tiling.dLength;
    dPadLength = tiling.dPadLength;
    halfDLength = tiling.halfDLength;
    halfDPadLength = tiling.halfDPadLength;
    totalSLines = tiling.totalSLines;
    storeSLines = tiling.storeSLines;
    storeDataLength = tiling.storeDataLength;
    storePadDataLength = tiling.storePadDataLength;
    formerCoreNum = tiling.formerCoreNum;
    tailCoreNum = tiling.tailCoreNum;
    formerSLines = tiling.formerSLines;
    tailSLines = tiling.tailSLines;

    if (GetBlockIdx() < formerCoreNum) {
        coreRelativeIdx = GetBlockIdx();
        coreSLines = formerSLines;
        ubLoop = tiling.formerUbLoop;
        ubLast = tiling.formerUbLast;
        xDataLength = tiling.formerXDataLength;
        rDataLength = tiling.formerRDataLength;
        ubLastDataLength = tiling.formerUbLastDataLength;
        ubLastPadDataLength = tiling.formerUbLastPadDataLength;
        xOffset = 0;
        rOffset = 0;
    } else {
        coreRelativeIdx = GetBlockIdx() - formerCoreNum;
        coreSLines = tailSLines;
        ubLoop = tiling.tailUbLoop;
        ubLast = tiling.tailUbLast;
        xDataLength = tiling.tailXDataLength;
        rDataLength = tiling.tailRDataLength;
        ubLastDataLength = tiling.tailUbLastDataLength;
        ubLastPadDataLength = tiling.tailUbLastPadDataLength;
        xOffset = tiling.formerXCoreOffset;
        rOffset = tiling.formerRCoreOffset;
    }
}

/* sin_l = -1 * sin_l */
template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBase<OriT, CmpT>::SinCompute(LocalTensor<CmpT> &sin, uint32_t sLines)
{
    uint32_t repeatOffset, innerOffset;
    uint32_t repeatLoop = sLines / REPEAT_MAX;
    uint8_t repeatLast = sLines % REPEAT_MAX;
    UnaryRepeatParams repeatParams{1, 1, repeatStride, repeatStride};

    for (uint32_t i = 0; i < repeatLoop; i++) {
        repeatOffset = i * REPEAT_MAX * dPadLength;
        for (uint32_t j = 0; j < innerHalfLoop; j++) {
            innerOffset = j * dataEachRepeat + repeatOffset;
            Muls(sin[innerOffset], sin[innerOffset], (CmpT)(-1.0), dataEachRepeat, REPEAT_MAX, repeatParams);
        }
        if (innerHalfLast > 0) {
            innerOffset = innerHalfLoop * dataEachRepeat + repeatOffset;
            Muls(sin[innerOffset], sin[innerOffset], (CmpT)(-1.0), innerHalfLast, REPEAT_MAX, repeatParams);
        }
    }
    if (repeatLast > 0) {
        repeatOffset = repeatLoop * REPEAT_MAX * dPadLength;
        for (uint32_t j = 0; j < innerHalfLoop; j++) {
            innerOffset = j * dataEachRepeat + repeatOffset;
            Muls(sin[innerOffset], sin[innerOffset], (CmpT)(-1.0), dataEachRepeat, repeatLast, repeatParams);
        }
        if (innerHalfLast > 0) {
            innerOffset = innerHalfLoop * dataEachRepeat + repeatOffset;
            Muls(sin[innerOffset], sin[innerOffset], (CmpT)(-1.0), innerHalfLast, repeatLast, repeatParams);
        }
    }
}

/* x = x * cos, xNew = xNew * sin, y = x + xNew */
template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBase<OriT, CmpT>::ComputeInner(LocalTensor<CmpT> &x, LocalTensor<CmpT> &xNew,
                                                                LocalTensor<CmpT> &cos, LocalTensor<CmpT> &sin,
                                                                uint32_t calcLength)
{
    Mul(x, x, cos, calcLength);
    Mul(xNew, xNew, sin, calcLength);
    Add(xNew, xNew, x, calcLength);
}

/* broadcast cos, sin from (1, D) to (storeSLines, D) or (ubLast, D) shape */
template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBase<OriT, CmpT>::RBroadCast(LocalTensor<CmpT> &cos, LocalTensor<CmpT> &sin,
                                                              uint32_t broadcastLines)
{
    uint32_t repeatOffset, copySrcOffset, copyDstOffset;
    uint32_t innerLoop = dPadLength / dataEachRepeat;
    uint32_t innerLast = dPadLength % dataEachRepeat;
    uint32_t repeatLoop = broadcastLines / REPEAT_MAX;
    uint8_t repeatLast = broadcastLines % REPEAT_MAX;
    CopyRepeatParams repParams{1, 1, repeatStride, 0};

    for (uint32_t i = 0; i < repeatLoop; i++) {
        repeatOffset = i * REPEAT_MAX * dPadLength;
        for (uint32_t j = 0; j < innerLoop; j++) {
            copySrcOffset = j * dataEachRepeat;
            copyDstOffset = copySrcOffset + repeatOffset + dPadLength;
            Copy(cos[copyDstOffset], cos[copySrcOffset], dataEachRepeat, REPEAT_MAX, repParams);
            Copy(sin[copyDstOffset], sin[copySrcOffset], dataEachRepeat, REPEAT_MAX, repParams);
        }
        if (innerLast > 0) {
            copySrcOffset = innerLoop * dataEachRepeat;
            copyDstOffset = copySrcOffset + repeatOffset + dPadLength;
            Copy(cos[copyDstOffset], cos[copySrcOffset], innerLast, REPEAT_MAX, repParams);
            Copy(sin[copyDstOffset], sin[copySrcOffset], innerLast, REPEAT_MAX, repParams);
        }
    }
    if (repeatLast > 0) {
        repeatOffset = repeatLoop * REPEAT_MAX * dPadLength;
        for (uint32_t j = 0; j < innerLoop; j++) {
            copySrcOffset = j * dataEachRepeat;
            copyDstOffset = copySrcOffset + repeatOffset + dPadLength;
            Copy(cos[copyDstOffset], cos[copySrcOffset], dataEachRepeat, repeatLast, repParams);
            Copy(sin[copyDstOffset], sin[copySrcOffset], dataEachRepeat, repeatLast, repParams);
        }
        if (innerLast > 0) {
            copySrcOffset = innerLoop * dataEachRepeat;
            copyDstOffset = copySrcOffset + repeatOffset + dPadLength;
            Copy(cos[copyDstOffset], cos[copySrcOffset], innerLast, repeatLast, repParams);
            Copy(sin[copyDstOffset], sin[copySrcOffset], innerLast, repeatLast, repParams);
        }
    }
}

/* copy x to xNew: x_l --> xNew_r, x_r --> xNew_l */
template <typename OriT, typename CmpT>
__aicore__ inline void RotateHalfBase<OriT, CmpT>::XNewCopy(LocalTensor<CmpT> &x, LocalTensor<CmpT> &xNew,
                                                            uint16_t sLines)
{
    uint16_t stride = this->halfDPadLength * sizeof(CmpT) / BYTE_OF_BLOCK;
    DataCopyParams copyParams{sLines, stride, stride, stride};
    DataCopy(xNew, x[this->halfDPadLength], copyParams);
    DataCopy(xNew[this->halfDPadLength], x, copyParams);
}

} // namespace RotateHalfN
#endif // ROTATE_HALF_BASE_H