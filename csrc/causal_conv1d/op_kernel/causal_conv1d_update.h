/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAUSAL_CONV1D_UPDATE_H
#define CAUSAL_CONV1D_UPDATE_H

#include "causal_conv1d.h"

namespace NsCausalConv1d {

template <typename T>
class CausalConv1dUpdate
    : public CausalConv1d<T, CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE, CAUSAL_CONV1D_TPL_WIDTH_RUNTIME,
                          CAUSAL_CONV1D_TPL_FN_PLAN_INVALID> {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR, GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                const CausalConv1dTilingData *tilingData)
    {
        (void)workspace;
        this->ResetRuntimeState(tilingData);
        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        this->weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight));
        this->biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias));
        this->convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(convStates));
        this->queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(queryStartLoc));
        this->cacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cacheIndices));
        this->numAcceptedTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(numAcceptedTokens));
        this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
        this->InitSharedBuffersAndEvents();
    }

    __aicore__ inline void Process()
    {
        const CausalConv1dTilingData *tilingData = this->GetTilingData();
        const int32_t dim = tilingData->dim;
        const int32_t baseDimCnt = static_cast<int32_t>(tilingData->baseDimCnt);
        const int32_t width = static_cast<int32_t>(tilingData->width);
        const int32_t baseDim = static_cast<int32_t>(tilingData->baseDim);
        if (baseDim <= 0 || baseDimCnt <= 0 || baseDim > MAX_BLOCK_DIM || width < 2 || width > MAX_WIDTH || dim <= 0 ||
            tilingData->batch <= 0) {
            this->ReleaseEvents();
            return;
        }

        this->ProcessDefault();
        this->ReleaseEvents();
    }
};

template <typename T>
__aicore__ inline void RunCausalConv1dUpdate(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                             GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                             GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                             const CausalConv1dTilingData *tilingData)
{
    CausalConv1dUpdate<T> op;
    op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, workspace,
            tilingData);
    op.Process();
}

} // namespace NsCausalConv1d

#endif // CAUSAL_CONV1D_UPDATE_H
