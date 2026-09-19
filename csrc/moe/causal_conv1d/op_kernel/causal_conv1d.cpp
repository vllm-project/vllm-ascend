/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d.cpp
 * \brief CausalConv1d kernel entry functions.
 */

#include "causal_conv1d_fn.h"
#include "causal_conv1d_update.h"

namespace {

template <typename T, uint32_t runModeKey, uint32_t widthKey, uint32_t fnPlanKey>
__aicore__ inline void RunCausalConv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                       GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR hasInitialState,
                                       GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                       const CausalConv1dTilingData *tilingData)
{
    if constexpr (runModeKey == CAUSAL_CONV1D_TPL_RUN_MODE_FN) {
        NsCausalConv1d::RunCausalConv1dFn<T, widthKey, fnPlanKey>(
            x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState, numAcceptedTokens, y, workspace,
            tilingData);
    } else {
        NsCausalConv1d::RunCausalConv1dUpdate<T>(
            x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState, numAcceptedTokens, y, workspace,
            tilingData);
    }
}

}

template <uint32_t runModeKey, uint32_t widthKey, uint32_t fnPlanKey>
__global__ __aicore__ void causal_conv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                         GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR hasInitialState,
                                         GM_ADDR numAcceptedTokens, GM_ADDR queryStartLocCpu, GM_ADDR cacheIndicesCpu,
                                         GM_ADDR hasInitialStateCpu, GM_ADDR numAcceptedTokensCpu, GM_ADDR y,
                                         GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CausalConv1dTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GM_ADDR userWorkspace = workspace;
    if (workspace != nullptr) {
        userWorkspace = AscendC::GetUserWorkspace(workspace);
    }

    GM_ADDR resolvedQueryStartLoc =
        (tilingData.queryStartLocUseCpu != 0) ? queryStartLocCpu : queryStartLoc;
    GM_ADDR resolvedCacheIndices =
        (tilingData.cacheIndicesUseCpu != 0) ? cacheIndicesCpu : cacheIndices;
    GM_ADDR resolvedHasInitialState =
        (tilingData.hasInitialStateUseCpu != 0) ? hasInitialStateCpu : hasInitialState;
    GM_ADDR resolvedNumAcceptedTokens =
        (tilingData.numAcceptedTokensUseCpu != 0) ? numAcceptedTokensCpu : numAcceptedTokens;

    RunCausalConv1d<DTYPE_X, runModeKey, widthKey, fnPlanKey>(
        x, weight, bias, convStates, resolvedQueryStartLoc, resolvedCacheIndices, resolvedHasInitialState,
        resolvedNumAcceptedTokens, y, userWorkspace, &tilingData);
}
