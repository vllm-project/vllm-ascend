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

    template <typename T>
__aicore__ inline void RunCausalConv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                      GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR hasInitialState,
                                      GM_ADDR y, const  NsCausalConv1d::CausalConv1dTilingData* tilingData)
{
    NsCausalConv1d::CausalConv1d<T> op;
    op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState, y, tilingData);
    op.Process();
}

} // namespace

template <uint32_t schMode>
__global__ __aicore__ void causal_conv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                        GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR hasInitialState,
                                        GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT( NsCausalConv1d::CausalConv1dTilingData);
    // GET_TILING_DATA_WITH_STRUCT( NsCausalConv1d::CausalConv1dTilingData, tilingData, tiling);
    GET_TILING_DATA(tilingData, tiling);
    #if defined(ORIG_DTYPE_X)
        #if (ORIG_DTYPE_X == DT_FLOAT16)
            RunCausalConv1d<half>(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState, y, &tilingData);
        #elif (ORIG_DTYPE_X == DT_BF16)
            RunCausalConv1d<bfloat16_t>(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState, y, &tilingData);
        #elif (ORIG_DTYPE_X == DT_FLOAT)
            RunCausalConv1d<float>(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState, y, &tilingData);
        #endif
    #else
        #if (DTYPE_X == DT_FLOAT16)
            RunCausalConv1d<half>(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState, y, &tilingData);
        #elif (DTYPE_X == DT_BF16)
            RunCausalConv1d<bfloat16_t>(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState, y, &tilingData);
        #elif (DTYPE_X == DT_FLOAT)
            RunCausalConv1d<float>(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState, y, &tilingData);
        #endif
    #endif
}