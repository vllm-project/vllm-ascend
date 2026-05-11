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

#ifndef CAUSAL_CONV1D_FN_H
#define CAUSAL_CONV1D_FN_H

#include "causal_conv1d.h"

namespace NsCausalConv1d {

template <typename T, uint32_t widthKey, uint32_t fnPlanKey>
class CausalConv1dFn : public CausalConv1d<T, CAUSAL_CONV1D_TPL_RUN_MODE_FN, widthKey, fnPlanKey> {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens, GM_ADDR y,
                                GM_ADDR workspace, const CausalConv1dTilingData *tilingData)
    {
        (void)numAcceptedTokens;
        this->ResetRuntimeState(tilingData);
        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        this->weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight));
        this->biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias));
        this->convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(convStates));
        this->queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(queryStartLoc));
        this->cacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cacheIndices));
        this->initialStateModeGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(initialStateMode));
        this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
        if (tilingData->hasInitStateWorkspace != 0) {
            const uint64_t syncElems =
                static_cast<uint64_t>(GetBlockNum()) * INIT_STATE_SYNCALL_NEED_SIZE;
            const uint64_t syncBytes = syncElems * sizeof(int32_t);
            const uint64_t workspaceElems =
                static_cast<uint64_t>(tilingData->batch) *
                static_cast<uint64_t>(tilingData->width - 1) *
                static_cast<uint64_t>(tilingData->dim);
            this->initStateSyncGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace), syncElems);
            auto *workspaceBytes = reinterpret_cast<__gm__ uint8_t *>(workspace);
            this->initStateWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspaceBytes + syncBytes),
                                                        workspaceElems);
        }
        this->InitSharedBuffersAndEvents();
    }

    __aicore__ inline void Process()
    {
        this->ProcessVarlenTokenTiled();
        this->ReleaseEvents();
    }
};

template <typename T, uint32_t widthKey, uint32_t fnPlanKey>
__aicore__ inline void RunCausalConv1dFn(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                        GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                        GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                        const CausalConv1dTilingData *tilingData)
{
    CausalConv1dFn<T, widthKey, fnPlanKey> op;
    op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, workspace,
            tilingData);
    op.Process();
}

}

#endif
