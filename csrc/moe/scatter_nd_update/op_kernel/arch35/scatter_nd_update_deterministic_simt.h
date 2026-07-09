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
 * \file scatter_nd_update_deterministic_simt.h
 * \brief
 */

#ifndef SCATTER_ND_UPDATE_DETER_SIMT_H
#define SCATTER_ND_UPDATE_DETER_SIMT_H

#include "kernel_operator.h"
#include "scatter_nd_update_common.h"
#include "op_kernel/math_util.h"
namespace ScatterNdUpdate {
using namespace AscendC;

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtComputeData(
    __ubuf__ PARAMS_T* updateLocalAddr, __gm__ PARAMS_T* outputGmAddr, __gm__ TYPE_T* maskGmAddr,
    __gm__ TYPE_T* varIdxGmAddr, uint32_t updateCount, TYPE_T updateOffSet, TYPE_T sliceSize, uint32_t rankSize,
    int64_t varSize, TYPE_T magic, TYPE_T shift)
{
    for (uint32_t i = threadIdx.x; i < updateCount; i += blockDim.x) {
        TYPE_T globalUpdateIdx = updateOffSet + i;
        TYPE_T quotient = Simt::UintDiv(globalUpdateIdx, magic, shift);

        // indice中对应行号
        TYPE_T currIndiceIdx = quotient;
        TYPE_T scatterAxisIdx = globalUpdateIdx - quotient * sliceSize;
        OFFSET_T idx = varIdxGmAddr[currIndiceIdx];

        if (idx >= 0 && idx < varSize && maskGmAddr[idx / sliceSize] == currIndiceIdx) {
            uint64_t dst = static_cast<uint64_t>(idx + scatterAxisIdx);
            outputGmAddr[dst] = updateLocalAddr[i];
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T = INDICES_T>
class ScatterNdUpdateDeterministicSimt
    : public ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T> {
public:
    __aicore__ inline ScatterNdUpdateDeterministicSimt(const ScatterNdUpdateRegBaseTilingData& tilingData, TPipe& pipe)
        : ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>(tilingData, pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputeData();
    __aicore__ inline void CopyInUpdate(LocalTensor<PARAMS_T>& updateLocal);

private:
    TYPE_T updateOffSet = 0;
    TYPE_T indiceOffSet = 0;
    TYPE_T idxLoopSize = 0;
    TYPE_T updateLoopSize = 0;
    int32_t updateCount = 0;
};

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimt<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::Init(
    GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace)
{
    this->InitBase(x, indices, updates, y, workspace);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimt<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::ComputeData()
{
    LocalTensor<PARAMS_T> updateLocal = this->inQueX.template AllocTensor<PARAMS_T>();
    CopyInUpdate(updateLocal);
    TYPE_T sliceSize = this->tiling_.sliceSize;
    uint32_t rankSize = this->tiling_.rankSize;
    TYPE_T updateOffSet = this->updateOffSet;

    int64_t varSize = this->tiling_.outputStorageShapeSize;
    uint32_t updateCount = this->updateCount;

    TYPE_T magic = 0;
    TYPE_T shift = 0;
    GetUintDivMagicAndShift(magic, shift, sliceSize);
    asc_vf_call<SimtComputeData<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>>(
        dim3(THREAD_NUM), (__ubuf__ PARAMS_T*)updateLocal.GetPhyAddr(), (__gm__ PARAMS_T*)(this->outputGm.GetPhyAddr()),
        (__gm__ TYPE_T*)this->maskGm.GetPhyAddr(), (__gm__ TYPE_T*)this->varIdxGm.GetPhyAddr(), updateCount,
        updateOffSet, sliceSize, rankSize, varSize, magic, shift);
    this->inQueX.template FreeTensor(updateLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimt<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::Process()
{
    if (this->tiling_.sliceSize == 0) {
        return;
    }
    SyncAll();
    this->CalcMask();
    SyncAll();

    if (this->tiling_.usedCoreNumBefore <= this->blockIdx) {
        return;
    }
    if (this->blockIdx == this->tiling_.usedCoreNumBefore - 1) {
        this->currBlockHandleIdx = this->tiling_.tailCoreIndexCount;
    } else {
        this->currBlockHandleIdx = this->tiling_.eachCoreIndexCount;
    }
    this->InitUpdateBuffer();

    this->idxLoopSize = Ops::Base::CeilDiv(this->currBlockHandleIdx, static_cast<TYPE_T>(this->tiling_.indicesFactor));
    this->indiceBlockOffSet = this->blockIdx * this->tiling_.eachCoreIndexCount;
    int64_t tailLoopIndices = this->currBlockHandleIdx - this->tiling_.indicesFactor * (this->idxLoopSize - 1);
    for (TYPE_T i = 0; i < this->idxLoopSize; i++) {
        this->indiceOffSet = this->indiceBlockOffSet + i * this->tiling_.indicesFactor;
        int64_t currLoopIndicesCount = i == this->idxLoopSize - 1 ? tailLoopIndices : this->tiling_.indicesFactor;
        int64_t currLoopUpdatesCount = currLoopIndicesCount * this->tiling_.afterAxis;
        for (TYPE_T j = 0; j < this->tiling_.updateLoopSize; j++) {
            this->updateOffSet = this->indiceOffSet * this->tiling_.sliceSize + j * this->tiling_.afterAxisFactor;
            if (j == this->tiling_.updateLoopSize - 1) {
                this->updateCount = currLoopUpdatesCount - j * this->tiling_.afterAxisFactor;
            } else {
                this->updateCount = this->tiling_.afterAxisFactor;
            }
            ComputeData();
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimt<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::CopyInUpdate(
    LocalTensor<PARAMS_T>& updateLocal)
{
    DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(this->updateCount * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyPadExtParams<PARAMS_T> xPadParams{false, 0, 0, 0};
    DataCopyPad(updateLocal, this->updateGm[this->updateOffSet], xCopyParams, xPadParams);
    this->inQueX.template EnQue(updateLocal);
    this->inQueX.template DeQue<PARAMS_T>();
}

} // namespace ScatterNdUpdate

#endif // SCATTER_ND_UPDATE_DETER_SIMT_H