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

/*!
 * \file causal_conv1d.cpp
 * \brief
 */

#include "causal_conv1d_fn.h"
#include "causal_conv1d_update.h"

namespace {

template <typename T, uint32_t runModeKey, uint32_t widthKey, uint32_t fnPlanKey>
__aicore__ inline void RunCausalConv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                       GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                       GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                       const CausalConv1dTilingData *tilingData)
{
    if constexpr (runModeKey == CAUSAL_CONV1D_TPL_RUN_MODE_FN) {
        NsCausalConv1d::RunCausalConv1dFn<T, widthKey, fnPlanKey>(
            x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, workspace,
            tilingData);
    } else {
        NsCausalConv1d::RunCausalConv1dUpdate<T>(
            x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, workspace,
            tilingData);
    }
}

} // namespace

template <uint32_t runModeKey, uint32_t widthKey, uint32_t fnPlanKey>
__global__ __aicore__ void causal_conv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                         GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                         GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CausalConv1dTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GM_ADDR userWorkspace = nullptr;

    RunCausalConv1d<DTYPE_X, runModeKey, widthKey, fnPlanKey>(
        x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
        userWorkspace, &tilingData);
}
