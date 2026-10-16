/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GMM_DEQUANT_SITU_QUANT_TILING_H_
#define GMM_DEQUANT_SITU_QUANT_TILING_H_

#include <stdint.h>

struct GmmDequantSituQuantTilingData {
    int32_t experts;
    int32_t k;
    int32_t n;
    int32_t capacity;
    int32_t groupListType;
    int32_t hasLinear;
    int32_t weightNz;
    float beta;
    float invBeta;
    float linearBeta;
    float invLinearBeta;
    uint32_t reserved;
    uint64_t packedBytes;
};

static_assert(sizeof(GmmDequantSituQuantTilingData) == 56, "Unexpected GmmDequantSituQuant tiling layout");

#endif // GMM_DEQUANT_SITU_QUANT_TILING_H_
