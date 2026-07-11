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
 * \file scatter_nd_update.h
 * \brief
 */

#ifndef SCATTER_ND_UPDATE_SMIT_SORT_H
#define SCATTER_ND_UPDATE_SMIT_SORT_H

#include "kernel_operator.h"
#include "scatter_nd_update_common.h"

namespace ScatterNdUpdate {
using namespace AscendC;

template <typename OFFSET_T, typename PARAMS_T, typename TYPE_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SortSimtCompute(
    __gm__ PARAMS_T* updateAddr, __gm__ PARAMS_T* varAddr, const __ubuf__ OFFSET_T* shiftSortLocalAddr,
    const __ubuf__ uint32_t* updatesOriginIdexLocalAddr, const __ubuf__ int32_t* uniqueIdCountLocalAddr,
    TYPE_T sliceSize, uint32_t uniqueIdNum, TYPE_T indiceBlockOffSet, int64_t varSize, TYPE_T magic, TYPE_T shift)
{
    for (TYPE_T index = threadIdx.x; index < uniqueIdNum * sliceSize; index += blockDim.x) {
        TYPE_T rowIndex = Simt::UintDiv(index, magic, shift);
        TYPE_T sliceIndex = index - rowIndex * sliceSize;

        int32_t indiceIndex = uniqueIdCountLocalAddr[rowIndex];

        OFFSET_T varIndex = shiftSortLocalAddr[indiceIndex];

        if (varIndex >= 0 && varIndex < varSize) {
            OFFSET_T updateIndex = updatesOriginIdexLocalAddr[indiceIndex];

            int64_t varOffset = varIndex + sliceIndex;
            int64_t updateOffset = (indiceBlockOffSet + updateIndex) * sliceSize + sliceIndex;
            varAddr[varOffset] = updateAddr[updateOffset];
        }
    }
}

template <typename OFFSET_T, typename PARAMS_T, typename TYPE_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void NoSortSimtCompute(
    __gm__ PARAMS_T* updateAddr, __gm__ PARAMS_T* varAddr, const __ubuf__ OFFSET_T* varIndexLocalAddr, TYPE_T sliceSize,
    uint32_t currUbTilingSize, TYPE_T indiceBlockOffSet, int64_t varSize, TYPE_T magic, TYPE_T shift)
{
    for (TYPE_T index = threadIdx.x; index < currUbTilingSize * sliceSize; index += blockDim.x) {
        TYPE_T rowIndex = Simt::UintDiv(index, magic, shift);
        TYPE_T sliceIndex = index - rowIndex * sliceSize;

        OFFSET_T varIndex = varIndexLocalAddr[rowIndex];

        if (varIndex >= 0 && varIndex < varSize) {
            int64_t varOffset = varIndex + sliceIndex;
            int64_t updateOffset = (indiceBlockOffSet + rowIndex) * sliceSize + sliceIndex;
            varAddr[varOffset] = updateAddr[updateOffset];
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T = INDICES_T>
class ScatterNdUpdateSimtSort : public ScatterNdUpdateBase<PARAMS_T, INDICES_T, OFFSET_T> {
public:
    __aicore__ inline ScatterNdUpdateSimtSort(const ScatterNdUpdateRegBaseTilingData& tilingData, TPipe& pipe)
        : pipe_(pipe), tiling_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData();
    __aicore__ inline void InitStride();
    __aicore__ inline void SimdFree(LocalTensor<uint32_t>& updatesOriginIndexLocal,
                                    LocalTensor<int32_t>& uniqueIdCountLocal);
    __aicore__ inline void Compute();
    __aicore__ inline void ComputeSortDimensionOne(LocalTensor<INDICES_T> indiceLocal);
    __aicore__ inline void ComputeSortDimensionOther(LocalTensor<INDICES_T> indiceLocal);
    __aicore__ inline void SortDataMove(LocalTensor<OFFSET_T> varIndexLocal);
    __aicore__ inline void NoSortDataMove(LocalTensor<OFFSET_T> varIndexLocal);

private:
    TPipe& pipe_;
    const ScatterNdUpdateRegBaseTilingData& tiling_;
    GlobalTensor<INDICES_T> idxGm;
    GlobalTensor<PARAMS_T> updateGm;
    GlobalTensor<PARAMS_T> varGm;

    TQue<QuePosition::VECIN, 1> indicesQueue_;

    TYPE_T indiceBlockOffSet = 0;
    TYPE_T currBlockTilingSize = 0;

    uint32_t ubTilingSize = 0;
    uint32_t currUbTilingSize = 0;

    TYPE_T ubLoopCnt = 0;
    uint32_t blockIdx;
};

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimtSort<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::Init(GM_ADDR x, GM_ADDR indices,
                                                                                            GM_ADDR updates, GM_ADDR y,
                                                                                            GM_ADDR workspace)
{
    if (tiling_.sliceSize == 0) {
        return;
    }

    blockIdx = GetBlockIdx();

    this->indiceBlockOffSet = tiling_.blockTilingSize * blockIdx;

    if (blockIdx == tiling_.blockNum - 1) {
        this->currBlockTilingSize = tiling_.tailBlockTilingSize;
    } else {
        this->currBlockTilingSize = tiling_.blockTilingSize;
    }

    this->ubTilingSize = tiling_.ubTilingSize;
    if (this->currBlockTilingSize <= tiling_.ubTilingSize) {
        this->ubTilingSize = this->currBlockTilingSize;
    }

    idxGm.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    updateGm.SetGlobalBuffer((__gm__ PARAMS_T*)updates);
    varGm.SetGlobalBuffer((__gm__ PARAMS_T*)y);

    pipe_.InitBuffer(indicesQueue_, 1, ROUND_UP32(ubTilingSize * tiling_.rankSize * sizeof(INDICES_T)));
    // 计算完偏移存储indice
    pipe_.InitBuffer(this->outOfstBuf_, ROUND_UP32(ubTilingSize * sizeof(OFFSET_T)));
    pipe_.InitBuffer(this->sortIndicesQue_, ROUND_UP32(ubTilingSize * sizeof(OFFSET_T)) + UB_AGLIN_VALUE * 2);
    pipe_.InitBuffer(this->updatesOriginIdexQue_, 1, ROUND_UP32(ubTilingSize * sizeof(uint32_t)));
    pipe_.InitBuffer(this->uniqueIdCountQue_, 1, ROUND_UP32((ubTilingSize + 1) * sizeof(int32_t)));

    pipe_.InitBuffer(this->maxScoreBuf_, HASH_SCORE_BUF_SIZE * sizeof(float));
    pipe_.InitBuffer(this->strideBuf_, MAX_SHAPE_RANK * sizeof(INDICES_T));

    InitStride();
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimtSort<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::Process()
{
    // if input is empty, return directly
    if (tiling_.sliceSize == 0) {
        return;
    }
    if (blockIdx < tiling_.blockNum) {
        this->ubLoopCnt = (this->currBlockTilingSize + this->ubTilingSize - 1) / this->ubTilingSize;

        for (TYPE_T idx = 0; idx < this->ubLoopCnt - 1; idx++) {
            this->currUbTilingSize = this->ubTilingSize;
            Compute();
        }

        this->currUbTilingSize = this->currBlockTilingSize - this->ubTilingSize * (this->ubLoopCnt - 1);
        Compute();
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimtSort<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::Compute()
{
    LocalTensor<OFFSET_T> varIndexLocal;
    LocalTensor<INDICES_T> indiceLocal = indicesQueue_.AllocTensor<INDICES_T>();

    ComputeSortDimensionOther(indiceLocal);
    varIndexLocal = this->outOfstBuf_.template Get<OFFSET_T>();

    // 判断是否动态排序
    LocalTensor<float> dstLocal = this->maxScoreBuf_.template Get<float>();
    if constexpr (IsSameType<OFFSET_T, int32_t>::value) {
        IndexStatisticInt32(varIndexLocal, dstLocal, this->maxScore_, currUbTilingSize, tiling_.afterAxis);
    } else {
        IndexStatisticInt64(varIndexLocal, dstLocal, this->maxScore_, currUbTilingSize, tiling_.afterAxis);
    }

    if (this->maxScore_ > 0.01f) {
        SortDataMove(varIndexLocal);
    } else {
        NoSortDataMove(varIndexLocal);
    }
    indicesQueue_.FreeTensor(indiceLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimtSort<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::SortDataMove(
    LocalTensor<OFFSET_T> varIndexLocal)
{
    this->SortAndComputeUniqueIdx(varIndexLocal, currUbTilingSize);

    LocalTensor<OFFSET_T> sortIndiceLocal = this->sortIndicesQue_.template Get<OFFSET_T>();
    LocalTensor<uint32_t> updatesOriginIndexLocal = this->updatesOriginIdexQue_.template DeQue<uint32_t>();
    LocalTensor<int32_t> uniqueIdCountLocal = this->uniqueIdCountQue_.template DeQue<int32_t>();

    LocalTensor<OFFSET_T> shiftSortLocal = sortIndiceLocal[this->shiftOffset_];
    TYPE_T sliceSize = tiling_.sliceSize;
    uint32_t uniqueIdNum = this->uniqueIdNum_;
    TYPE_T indiceBlockOffSets = this->indiceBlockOffSet;
    int64_t varSize = tiling_.outputStorageShapeSize;
    TYPE_T magic = 0;
    TYPE_T shift = 0;

    GetUintDivMagicAndShift(magic, shift, sliceSize);
    asc_vf_call<SortSimtCompute<OFFSET_T, PARAMS_T, TYPE_T>>(
        dim3(THREAD_NUM), (__gm__ PARAMS_T*)(updateGm.GetPhyAddr()), (__gm__ PARAMS_T*)(varGm.GetPhyAddr()),
        (__ubuf__ OFFSET_T*)shiftSortLocal.GetPhyAddr(), (__ubuf__ uint32_t*)updatesOriginIndexLocal.GetPhyAddr(),
        (__ubuf__ int32_t*)uniqueIdCountLocal.GetPhyAddr(), sliceSize, uniqueIdNum, indiceBlockOffSets, varSize, magic,
        shift);
    this->indiceBlockOffSet = this->indiceBlockOffSet + currUbTilingSize;
    SimdFree(updatesOriginIndexLocal, uniqueIdCountLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimtSort<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::NoSortDataMove(
    LocalTensor<OFFSET_T> varIndexLocal)
{
    TYPE_T sliceSize = tiling_.sliceSize;
    uint32_t currUbTilingSize = this->currUbTilingSize;
    TYPE_T indiceBlockOffSets = this->indiceBlockOffSet;
    int64_t varSize = tiling_.outputStorageShapeSize;
    TYPE_T magic = 0;
    TYPE_T shift = 0;
    GetUintDivMagicAndShift(magic, shift, sliceSize);
    asc_vf_call<NoSortSimtCompute<OFFSET_T, PARAMS_T, TYPE_T>>(
        dim3(THREAD_NUM), (__gm__ PARAMS_T*)(updateGm.GetPhyAddr()), (__gm__ PARAMS_T*)(varGm.GetPhyAddr()),
        (__ubuf__ OFFSET_T*)varIndexLocal.GetPhyAddr(), sliceSize, currUbTilingSize, indiceBlockOffSets, varSize, magic,
        shift);
    this->indiceBlockOffSet = this->indiceBlockOffSet + currUbTilingSize;
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimtSort<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::ComputeSortDimensionOther(
    LocalTensor<INDICES_T> indiceLocal)
{
    this->template CopyIn<INDICES_T>(indiceLocal, idxGm[this->indiceBlockOffSet * tiling_.rankSize],
                                     currUbTilingSize * tiling_.rankSize);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    LocalTensor<OFFSET_T> outOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();
    this->ComputeOutOfset(indiceLocal, outOfstLocal, currUbTilingSize, tiling_.rankSize);

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimtSort<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::ComputeSortDimensionOne(
    LocalTensor<INDICES_T> indiceLocal)
{
    this->template CopyIn<INDICES_T>(indiceLocal, idxGm[this->indiceBlockOffSet * tiling_.rankSize],
                                     currUbTilingSize * tiling_.rankSize);

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimtSort<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::InitStride()
{
    LocalTensor<INDICES_T> strideLocal = this->strideBuf_.template Get<INDICES_T>();
    for (int32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        strideLocal(i) = tiling_.strideList[i];
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateSimtSort<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::SimdFree(
    LocalTensor<uint32_t>& updatesOriginIndexLocal, LocalTensor<int32_t>& uniqueIdCountLocal)
{
    this->updatesOriginIdexQue_.FreeTensor(updatesOriginIndexLocal);
    this->uniqueIdCountQue_.FreeTensor(uniqueIdCountLocal);
}

} // namespace ScatterNdUpdate

#endif // SCATTER_ND_UPDATE_SMIT_H
