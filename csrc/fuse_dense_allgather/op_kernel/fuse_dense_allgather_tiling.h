/*
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

#ifndef FUSE_DENSE_ALLGATHER
#define FUSE_DENSE_ALLGATHER

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct Opshape {
    int32_t batchSize = 1;
    int32_t m = -1;
    int32_t n = -1;
};

struct PPTilingData {
    Opshape opShape = {};
    int32_t m0 = 1;
    int32_t n0 = 1;
    int32_t mLoop = 1;
    int32_t nLoop = 1;
    int32_t coreLoop = 1;
    int32_t swizzlCount = 1;
    uint32_t tilingKey = 0;
    int32_t blockDim = 1;
};

struct CommTilingData {
    int32_t rank = 1;
    int32_t rankSize = 1;
    int32_t pValue = 1;
    int32_t ubMoveNum = 1;
    int32_t lenPerLoop = 1;
    int32_t is91093 = 0;
};

struct FuseDenseAllgatherInfo {
    PPTilingData ppTilingData{};
    CommTilingData commTilingData{};
};

struct FuseDenseAllgatherTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    FuseDenseAllgatherInfo fuseDenseAllgatherInfo;
};

#endif  // FUSE_DENSE_ALLGATHER