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

#include "causal_conv1d.h"

namespace {

// NOTE:
// Dtype is provided via AscendC compile macros (e.g. DTYPE_X / ORIG_DTYPE_X), so tiling key does not need to carry dtype.

template <typename T>
__aicore__ inline void RunCausalConv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                      GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                      GM_ADDR numAcceptedTokens, GM_ADDR y, const CausalConv1dTilingData* tilingData)
{
    NsCausalConv1d::CausalConv1d<T> op;
    op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, tilingData);
    op.Process();
}

} // namespace

template <uint32_t schMode>
__global__ __aicore__ void causal_conv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                        GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                        GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CausalConv1dTilingData);
    GET_TILING_DATA_WITH_STRUCT(CausalConv1dTilingData, tilingData, tiling);

    #if defined(ORIG_DTYPE_X)
        #if (ORIG_DTYPE_X == DT_FLOAT16)
            RunCausalConv1d<half>(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, &tilingData);
        #elif (ORIG_DTYPE_X == DT_BF16)
            RunCausalConv1d<bfloat16_t>(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, &tilingData);
        #endif
    #else
        #if (DTYPE_X == DT_FLOAT16)
            RunCausalConv1d<half>(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, &tilingData);
        #elif (DTYPE_X == DT_BF16)
            RunCausalConv1d<bfloat16_t>(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, &tilingData);
        #endif
    #endif
}

