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
 * \file reshape_and_cache.cpp
 * \brief
 */
#include "reshape_and_cache_by_group.h"

extern "C" __global__ __aicore__ void reshape_and_cache_by_group(
    GM_ADDR keyIn, GM_ADDR keyCacheIn, GM_ADDR groupLen, GM_ADDR groupKeyIdx, GM_ADDR groupKeyCacheIdx, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    REGISTER_TILING_DEFAULT(ReshapeAndCacheByGroup::ReshapeAndCacheByGroupTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        ReshapeAndCacheByGroup::ReshapeAndCacheByGroupBase<uint8_t> op;
        op.Init( &pipe, &tilingData);
        op.Process(keyIn,keyCacheIn, groupLen, groupKeyIdx, groupKeyCacheIdx);
    } else if (TILING_KEY_IS(2)) {
        ReshapeAndCacheByGroup::ReshapeAndCacheByGroupBase<half> op;
        op.Init( &pipe, &tilingData);
        op.Process(keyIn,keyCacheIn, groupLen, groupKeyIdx, groupKeyCacheIdx);
    } else if (TILING_KEY_IS(4)) {
        ReshapeAndCacheByGroup::ReshapeAndCacheByGroupBase<int32_t> op;
        op.Init( &pipe, &tilingData);
        op.Process(keyIn,keyCacheIn, groupLen, groupKeyIdx, groupKeyCacheIdx);
    }
}
