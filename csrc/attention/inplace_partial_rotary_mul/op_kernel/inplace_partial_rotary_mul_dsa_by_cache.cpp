/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file inplace_partial_rotary_mul_dsa_by_cache.cpp
 * \brief In-place partial rotary embedding with position-indexed rope caches.
 */
#if !defined(__DAV_C310__) && defined(DTYPE_POSITIONS)
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "inplace_partial_rotary_mul.h"

#define TILING_KEY_DSA_BY_CACHE_BRC 3001
#define TILING_KEY_DSA_BY_CACHE_NO_BRC 3002
#define TILING_KEY_DSA_BY_CACHE_BRC_FP32_ROPE 3011
#define TILING_KEY_DSA_BY_CACHE_NO_BRC_FP32_ROPE 3012

using namespace AscendC;
using namespace InplacePartialRotaryMul;

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul_dsa_by_cache(
    GM_ADDR x, GM_ADDR positions, GM_ADDR cos_sin_cache, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tilingData, tiling);
    const RopeRegbaseTilingData* __restrict__ tilingData1 = &tilingData;
    if (TILING_KEY_IS(TILING_KEY_DSA_BY_CACHE_BRC)) {
        InplacePartialRotaryMul::InplacePartialRotaryMulDsaByCacheABA<DTYPE_X, DTYPE_POSITIONS, true> op;
        op.Init(x, positions, cos_sin_cache, cos_sin_cache, y, workspace, tilingData1, &pipe);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_DSA_BY_CACHE_NO_BRC)) {
        InplacePartialRotaryMul::InplacePartialRotaryMulDsaByCacheABA<DTYPE_X, DTYPE_POSITIONS, false> op;
        op.Init(x, positions, cos_sin_cache, cos_sin_cache, y, workspace, tilingData1, &pipe);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_DSA_BY_CACHE_BRC_FP32_ROPE)) {
        InplacePartialRotaryMul::InplacePartialRotaryMulDsaByCacheABA<DTYPE_X, DTYPE_POSITIONS, true, float> op;
        op.Init(x, positions, cos_sin_cache, cos_sin_cache, y, workspace, tilingData1, &pipe);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_DSA_BY_CACHE_NO_BRC_FP32_ROPE)) {
        InplacePartialRotaryMul::InplacePartialRotaryMulDsaByCacheABA<DTYPE_X, DTYPE_POSITIONS, false, float> op;
        op.Init(x, positions, cos_sin_cache, cos_sin_cache, y, workspace, tilingData1, &pipe);
        op.Process();
        return;
    }
}
#endif
