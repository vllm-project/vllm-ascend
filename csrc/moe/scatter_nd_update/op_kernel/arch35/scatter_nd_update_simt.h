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
 * \file scatter_nd_add.h
 * \brief
 */

#ifndef SCATTER_ND_UPDATE_SMIT_H
#define SCATTER_ND_UPDATE_SMIT_H

#include "kernel_operator.h"
#include "scatter_nd_update_common.h"

namespace ScatterNdUpdate {
using namespace AscendC;

__aicore__ inline uint32_t ROUND_UP32(uint32_t x)
{
    if (x % UB_AGLIN_VALUE != 0) {
        return (x / UB_AGLIN_VALUE + 1) * UB_AGLIN_VALUE;
    }
    return x;
}

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtCompute(
    __ubuf__ INDICES_T* idxLocalAddr, __ubuf__ PARAMS_T* xLocalAddr, __gm__ PARAMS_T* outputGmAddr,
    const __ubuf__ TYPE_T* strideListAddr, const __ubuf__ TYPE_T* outputShapeAddr, uint32_t currUbTilingSize,
    TYPE_T xOffSet, TYPE_T sliceSize, uint32_t rankSize, OFFSET_T indiceOffSet, TYPE_T magic, TYPE_T shift)
{
    for (uint32_t index = threadIdx.x; index < currUbTilingSize; index += blockDim.x) {
        TYPE_T globalIdx = xOffSet + index;
        TYPE_T quotient = Simt::UintDiv(globalIdx, magic, shift);
        TYPE_T currIndiceIdx = quotient * rankSize;
        TYPE_T scatterAxisIdx = globalIdx - quotient * sliceSize;
        OFFSET_T idx = 0;
        bool outOfBound = false;
        for (TYPE_T dim = 0; dim < rankSize; ++dim) {
            INDICES_T indiceVal = idxLocalAddr[currIndiceIdx + dim - indiceOffSet];
            outOfBound |= (indiceVal < static_cast<INDICES_T>(0) ||
                           static_cast<TYPE_T>(indiceVal) > outputShapeAddr[dim]);
            idx += static_cast<OFFSET_T>(indiceVal) * strideListAddr[dim];
        }
        if (!outOfBound) {
            uint64_t dst = static_cast<uint64_t>(idx + scatterAxisIdx);
            outputGmAddr[dst] = xLocalAddr[index];
        }
    }
}

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtComputeDimensionOne(
    __gm__ INDICES_T* idxGmAddr, __ubuf__ PARAMS_T* xLocalAddr, __gm__ PARAMS_T* outputGmAddr,
    uint32_t currUbTilingSize, TYPE_T xOffSet, TYPE_T sliceSize, uint32_t rankSize, TYPE_T sliceStride,
    OFFSET_T indiceOffSet, int64_t varInAxis, TYPE_T magic, TYPE_T shift)
{
    for (uint32_t index = threadIdx.x; index < currUbTilingSize; index += blockDim.x) {
        TYPE_T globalIdx = xOffSet + index;
        TYPE_T quotient = Simt::UintDiv(globalIdx, magic, shift);
        TYPE_T currIndiceIdx = quotient * rankSize;
        TYPE_T scatterAxisIdx = globalIdx - quotient * sliceSize;
        OFFSET_T idx = static_cast<OFFSET_T>(idxGmAddr[currIndiceIdx]);

        if (idx >= 0 && idx < varInAxis) {
            uint64_t dst = static_cast<uint64_t>(idx * sliceStride + scatterAxisIdx);
            outputGmAddr[dst] = xLocalAddr[index];
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T = INDICES_T>
class ScatterNdUpdateSimt {
public:
    __aicore__ inline ScatterNdUpdateSimt(const ScatterNdUpdateRegBaseTilingData& tilingData, TPipe& pipe)
        : pipe_(pipe), tiling_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeData();
    __aicore__ inline void CopyIn(LocalTensor<INDICES_T>& idxLocal, LocalTensor<PARAMS_T>& xLocal);
    __aicore__ inline void SimdFree(LocalTensor<INDICES_T>& idxLocal, LocalTensor<PARAMS_T>& xLocal);
    __aicore__ inline void ComputeDimensionOther();
    __aicore__ inline void ComputeDimensionOne();
    __aicore__ inline void CopyInUpdate(LocalTensor<PARAMS_T>& xLocal);

private:
    TPipe& pipe_;
    const ScatterNdUpdateRegBaseTilingData& tiling_;
    GlobalTensor<INDICES_T> idxGm;
    GlobalTensor<PARAMS_T> xGm;
    GlobalTensor<PARAMS_T> outputGm;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueIdx, inQueX;
    TBuf<TPosition::VECCALC> strideListBuf;
    TBuf<TPosition::VECCALC> outputShapeBuf;

    uint32_t blockIdx;
    TYPE_T blockTilingSize = 0;
    TYPE_T currBlockTilingSize = 0;

    uint32_t ubTilingSize = 0;
    uint32_t currUbTilingSize = 0;

    TYPE_T xBlockOffSet = 0;
    TYPE_T xOffSet = 0;
    OFFSET_T indiceBlockOffSet = 0;
    OFFSET_T indiceOffSet = 0;
    uint32_t currIdxTilingSize = 0;
    TYPE_T ubLoopCnt = 0;
};

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimt<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::Init(GM_ADDR x, GM_ADDR indices,
                                                                                        GM_ADDR updates, GM_ADDR y,
                                                                                        GM_ADDR workspace)
{
    if (tiling_.sliceSize == 0) {
        return;
    }

    blockIdx = GetBlockIdx();

    this->xBlockOffSet = tiling_.blockTilingSize * blockIdx;
    this->indiceBlockOffSet = static_cast<OFFSET_T>(this->xBlockOffSet / tiling_.sliceSize * tiling_.rankSize);

    if (blockIdx == tiling_.blockNum - 1) {
        this->currBlockTilingSize = tiling_.tailBlockTilingSize;
    } else {
        this->currBlockTilingSize = tiling_.blockTilingSize;
    }

    this->ubTilingSize = tiling_.ubTilingSize;
    if (this->currBlockTilingSize <= tiling_.ubTilingSize) {
        this->ubTilingSize = this->currBlockTilingSize;
    }

    auto indiceUbTilingSize = (this->ubTilingSize + tiling_.sliceSize - 1) / tiling_.sliceSize * tiling_.rankSize;
    indiceUbTilingSize += 2;
    idxGm.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    xGm.SetGlobalBuffer((__gm__ PARAMS_T*)updates);
    outputGm.SetGlobalBuffer((__gm__ PARAMS_T*)y);

    pipe_.InitBuffer(inQueX, DOUBLE_BUFFER, ROUND_UP32(this->ubTilingSize * sizeof(PARAMS_T)));
    if (tiling_.rankSize >= INDICE_RANK_TWO) {
        pipe_.InitBuffer(inQueIdx, DOUBLE_BUFFER, ROUND_UP32(indiceUbTilingSize * sizeof(INDICES_T)));
        pipe_.InitBuffer(strideListBuf, MAX_SHAPE_RANK * sizeof(TYPE_T));
        pipe_.InitBuffer(outputShapeBuf, MAX_SHAPE_RANK * sizeof(TYPE_T));
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimt<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::ComputeDimensionOther()
{
    LocalTensor<INDICES_T> idxLocal = inQueIdx.AllocTensor<INDICES_T>();
    LocalTensor<PARAMS_T> xLocal = inQueX.AllocTensor<PARAMS_T>();
    CopyIn(idxLocal, xLocal);
    uint32_t currUbTilingSize = this->currUbTilingSize;
    TYPE_T xOffSet = this->xOffSet;
    TYPE_T sliceSize = tiling_.sliceSize;
    uint32_t rankSize = tiling_.rankSize;
    OFFSET_T indiceOffSet = this->indiceOffSet;

    LocalTensor<TYPE_T> strideList = strideListBuf.Get<TYPE_T>();
    LocalTensor<TYPE_T> outputShape = outputShapeBuf.Get<TYPE_T>();
    for (uint32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        strideList(i) = tiling_.strideList[i];
    }
    for (uint32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        outputShape(i) = tiling_.outPutShape[i];
    }
    DataSyncBarrier<MemDsbT::UB>();
    TYPE_T magic = 0;
    TYPE_T shift = 0;
    GetUintDivMagicAndShift(magic, shift, sliceSize);
    asc_vf_call<SimtCompute<INDICES_T, PARAMS_T, TYPE_T, OFFSET_T>>(
        dim3(THREAD_NUM), (__ubuf__ INDICES_T*)idxLocal.GetPhyAddr(), (__ubuf__ PARAMS_T*)xLocal.GetPhyAddr(),
        (__gm__ PARAMS_T*)(outputGm.GetPhyAddr()), (__ubuf__ TYPE_T*)strideList.GetPhyAddr(),
        (__ubuf__ TYPE_T*)outputShape.GetPhyAddr(), currUbTilingSize, xOffSet, sliceSize, rankSize, indiceOffSet, magic,
        shift);
    SimdFree(idxLocal, xLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimt<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::ComputeDimensionOne()
{
    LocalTensor<PARAMS_T> xLocal = inQueX.AllocTensor<PARAMS_T>();
    CopyInUpdate(xLocal);
    uint32_t currUbTilingSize = this->currUbTilingSize;
    TYPE_T xOffSet = this->xOffSet;
    TYPE_T sliceSize = tiling_.sliceSize;
    uint32_t rankSize = tiling_.rankSize;
    OFFSET_T indiceOffSet = this->indiceOffSet;
    int64_t varInAxis = tiling_.varInAxis;
    TYPE_T sliceStride = tiling_.strideList[0];

    TYPE_T magic = 0;
    TYPE_T shift = 0;
    GetUintDivMagicAndShift(magic, shift, sliceSize);
    asc_vf_call<SimtComputeDimensionOne<INDICES_T, PARAMS_T, TYPE_T, OFFSET_T>>(
        dim3(THREAD_NUM), (__gm__ INDICES_T*)(idxGm.GetPhyAddr()), (__ubuf__ PARAMS_T*)xLocal.GetPhyAddr(),
        (__gm__ PARAMS_T*)(outputGm.GetPhyAddr()), currUbTilingSize, xOffSet, sliceSize, rankSize, sliceStride,
        indiceOffSet, varInAxis, magic, shift);
    inQueX.FreeTensor(xLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimt<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::Process()
{
    if (tiling_.sliceSize == 0) {
        return;
    }
    if (blockIdx < tiling_.blockNum) {
        this->ubLoopCnt = (this->currBlockTilingSize + this->ubTilingSize - 1) / this->ubTilingSize;
        for (TYPE_T idx = 0; idx < this->ubLoopCnt - 1; idx++) {
            this->currUbTilingSize = this->ubTilingSize;
            this->xOffSet = this->xBlockOffSet + idx * this->ubTilingSize;
            ComputeData();
        }
        this->xOffSet = this->xBlockOffSet + (this->ubLoopCnt - 1) * this->ubTilingSize;
        this->currUbTilingSize = this->currBlockTilingSize - this->ubTilingSize * (this->ubLoopCnt - 1);
        ComputeData();
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimt<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::ComputeData()
{
    auto currEnd = this->xOffSet + this->currUbTilingSize;
    auto indiceBegin = static_cast<OFFSET_T>(this->xOffSet / tiling_.sliceSize * tiling_.rankSize);
    auto indiceEnd = static_cast<OFFSET_T>((currEnd + tiling_.sliceSize - 1) / tiling_.sliceSize * tiling_.rankSize);
    this->currIdxTilingSize = indiceEnd - indiceBegin;
    this->indiceOffSet = indiceBegin;
    if (tiling_.rankSize >= INDICE_RANK_TWO) {
        ComputeDimensionOther();
    } else {
        ComputeDimensionOne();
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimt<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::CopyIn(
    LocalTensor<INDICES_T>& idxLocal, LocalTensor<PARAMS_T>& xLocal)
{
    DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(this->currIdxTilingSize * sizeof(INDICES_T)), 0, 0, 0};
    DataCopyPadExtParams<INDICES_T> idxPadParams{false, 0, 0, 0};
    DataCopyPad(idxLocal, idxGm[this->indiceOffSet], idxCopyParams, idxPadParams);

    DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(this->currUbTilingSize * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyPadExtParams<PARAMS_T> xPadParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[this->xOffSet], xCopyParams, xPadParams);

    inQueIdx.EnQue(idxLocal);
    inQueX.EnQue(xLocal);
    inQueIdx.DeQue<INDICES_T>();
    inQueX.DeQue<PARAMS_T>();
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimt<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::SimdFree(
    LocalTensor<INDICES_T>& idxLocal, LocalTensor<PARAMS_T>& xLocal)
{
    inQueIdx.FreeTensor(idxLocal);
    inQueX.FreeTensor(xLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimt<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::CopyInUpdate(
    LocalTensor<PARAMS_T>& xLocal)
{
    DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(this->currUbTilingSize * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyPadExtParams<PARAMS_T> xPadParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm[this->xOffSet], xCopyParams, xPadParams);

    inQueX.EnQue(xLocal);
    inQueX.DeQue<PARAMS_T>();
}

} // namespace ScatterNdUpdate

#endif // SCATTER_ND_UPDATE_SMIT_H