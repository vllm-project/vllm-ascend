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

#ifndef MATMUL_ALLREDUCE_ADD_RMSNORM_TILING_H
#define MATMUL_ALLREDUCE_ADD_RMSNORM_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

enum QuantGranularity : int {
    QUANT_GRANULARITY_UNDEFINED = -1,
    PER_TENSOR = 0,
    PER_CHANNEL = 1,
    PER_GROUP = 2,
    QUANT_GRANULARITY_MAX = 3,
};

struct Opshape {
    int32_t batchSize = 1;
    int32_t m = -1;
    int32_t k = -1;
    int32_t n = -1;
};

struct PPTilingData {
    Opshape opShape = {};
    int32_t m0 = 1;
    int32_t k0 = 1;
    int32_t n0 = 1;
    int32_t mLoop = 1;
    int32_t kLoop = 1;
    int32_t nLoop = 1;
    int32_t coreLoop = 1;
    int32_t swizzlCount = 1;
    int32_t swizzlDirect = 0;
    uint32_t tilingKey = 0;
    int32_t blockDim = 1;
    int32_t splitK = 0;
    bool weightNz = false;
    bool isTransA = false;
    bool isTransB = false;
    bool isGatherAddOut = false;
};

struct CommTilingData {
    int32_t rank = 1;
    int32_t rankSize = 1;
    int32_t pValue = 1;
    int32_t ubMoveNum = 1;
    int32_t write2OtherRank = 0;
    int32_t withSerialMode = 0;
    int32_t tag = 0;
    int32_t commNpuSplit = 1;
    int32_t commDataSplit = 1;
    int32_t commDirect = 0;
    int32_t lenPerLoop = 1;
    int32_t is91093 = 0;
    int32_t buffer_size = 0;
};

struct RmsNormTilingData {
    RmsNormTiling tiling{};
    uint32_t loopCount;
    uint32_t calcBytes;
    float epsilon{};
};

struct QuantInfo {
    QuantGranularity dequantGranularity = QuantGranularity::QUANT_GRANULARITY_UNDEFINED;
    int32_t dequantGroupSize = -1;
    QuantGranularity quantGranularity = QuantGranularity::QUANT_GRANULARITY_UNDEFINED;
    int32_t quantGroupSize = -1;
};

struct MatmulAllreduceAddRmsnormInfo {
    PPTilingData ppTilingData{};
    CommTilingData commTilingData{};
    RmsNormTilingData rmsnormTilingData{};
    QuantInfo quantInfo{};
};

struct MatmulAllreduceAddRmsnormTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MatmulAllreduceAddRmsnormInfo matmulAllreduceAddRmsnormInfo;
};

#endif  // MATMUL_ALLREDUCE_ADD_RMSNORM_TILING_H