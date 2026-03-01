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
 * \file causal_conv1d_update_base.h
 * \brief causal_conv1d_update kernel
 */

#ifndef CAUSAL_CONV1D_UPDATE_BASE_H
#define CAUSAL_CONV1D_UPDATE_BASE_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "causal_conv1d_update_struct.h"
#include "causal_conv1d_update_tilingdata.h"
#include "platform_util.h"

namespace CausalConv1dUpdateOp {
using namespace AscendC;
using namespace Ops::Base;

template <typename T>
class CausalConv1dUpdateBase {
public:
    __aicore__ inline CausalConv1dUpdateBase(){};

protected:
    __aicore__ inline void ParseTilingData(
        const CausalConv1dUpdateTilingData* tilingData, CausalConv1dUpdateTilingData& runTilingData);
    __aicore__ inline void ParseCoreBlocks(
        const CausalConv1dUpdateTilingData& runTilingData, int32_t blockIdx, int64_t& blockLen);
    __aicore__ inline void GetXInCopyParams(int64_t xLen, AscendC::DataCopyExtParams& copyParams);
    __aicore__ inline void GetWeightInCopyParams(int64_t weightLen, AscendC::DataCopyExtParams& copyParams);
    __aicore__ inline void GetStateInCopyParams(int64_t stateLen, AscendC::DataCopyExtParams& copyParams);
    __aicore__ inline void GetBiasInCopyParams(int64_t biasLen, AscendC::DataCopyExtParams& copyParams);
    __aicore__ inline void GetYOutCopyParams(int64_t yLen, AscendC::DataCopyExtParams& copyParams);
    __aicore__ inline void GetStateOutCopyParams(int64_t stateLen, AscendC::DataCopyExtParams& copyParams);

// protected:
//     constexpr static int32_t BLOCK_SIZE = GetUbBlockSize();

};

template <typename T>
__aicore__ inline void CausalConv1dUpdateBase<T>::ParseTilingData(
    const CausalConv1dUpdateTilingData* tilingData, CausalConv1dUpdateTilingData& runTilingData)
{
    runTilingData.numCore = tilingData->numCore;
    runTilingData.blockFactor = tilingData->blockFactor;
    runTilingData.blockTailFactor = tilingData->blockTailFactor;
    runTilingData.batch = tilingData->batch;
    runTilingData.seqLen = tilingData->seqLen;
    runTilingData.dim = tilingData->dim;
    runTilingData.width = tilingData->width;
    runTilingData.stateLen = tilingData->stateLen;
    runTilingData.hasIndices = tilingData->hasIndices;
    runTilingData.hasBias = tilingData->hasBias;
    runTilingData.hasNumAccept = tilingData->hasNumAccept;
    runTilingData.hasQueryLoc = tilingData->hasQueryLoc;
    runTilingData.activationMode = tilingData->activationMode;
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateBase<T>::ParseCoreBlocks(
    const CausalConv1dUpdateTilingData& runTilingData, int32_t blockIdx, int64_t& batchNum)
{
    if (blockIdx == runTilingData.numCore - 1) {
        batchNum = runTilingData.blockTailFactor;
    } else {
        batchNum = runTilingData.blockFactor;
    }
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateBase<T>::GetXInCopyParams(int64_t xLen, DataCopyExtParams& copyParams)
{
    copyParams.blockCount = 1;
    copyParams.blockLen = xLen * sizeof(T);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateBase<T>::GetWeightInCopyParams(int64_t weightLen, DataCopyExtParams& copyParams)
{
    copyParams.blockCount = 1;
    copyParams.blockLen = weightLen * sizeof(T);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateBase<T>::GetStateInCopyParams(int64_t stateLen, DataCopyExtParams& copyParams)
{
    copyParams.blockCount = 1;
    copyParams.blockLen = stateLen * sizeof(T);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateBase<T>::GetBiasInCopyParams(int64_t biasLen, DataCopyExtParams& copyParams)
{
    copyParams.blockCount = 1;
    copyParams.blockLen = biasLen * sizeof(T);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateBase<T>::GetYOutCopyParams(int64_t yLen, DataCopyExtParams& copyParams)
{
    copyParams.blockCount = 1;
    copyParams.blockLen = yLen * sizeof(T);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
}

template <typename T>
__aicore__ inline void CausalConv1dUpdateBase<T>::GetStateOutCopyParams(int64_t stateLen, DataCopyExtParams& copyParams)
{
    copyParams.blockCount = 1;
    copyParams.blockLen = stateLen * sizeof(T);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
}

} // namespace CausalConv1dUpdateOp

#endif