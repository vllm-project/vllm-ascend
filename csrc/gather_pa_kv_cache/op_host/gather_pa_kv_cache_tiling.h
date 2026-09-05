/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GATHER_PA_KV_CACHE_TILING_H
#define GATHER_PA_KV_CACHE_TILING_H

#include <cstdint>

constexpr int32_t TILING_KEY_NZ = 577;
constexpr int32_t TILING_KEY_ND_INT8 = 618;
constexpr int32_t TILING_KEY_ND_B16 = 619;

struct GatherPaKvCacheTilingData {
    int32_t blockSize;
    int32_t numTokens;
    int32_t numblkTabCol;
    int32_t tokenSizeK;
    int32_t tokenSizeV;
    int32_t typeByte;
    int32_t hasSeqStarts;
    int32_t isSeqLensCumsum;
    int64_t kCacheBlockStride;
    int64_t vCacheBlockStride;
    // vllm-ascend direct kernels do not receive the GE tiling key, so carry
    // the original 577/618/619 key in the tiling payload.
    int32_t tilingKey;
};

#endif  // GATHER_PA_KV_CACHE_TILING_H
