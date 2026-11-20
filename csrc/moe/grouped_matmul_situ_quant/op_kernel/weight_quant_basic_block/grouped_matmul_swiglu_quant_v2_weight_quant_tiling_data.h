/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_v2_weight_quant_tiling_data.h
 * \brief GMMSQ MxA8W4 TilingData structure definition
 */
#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_WEIGHT_QUANT_TILING_DATA_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_WEIGHT_QUANT_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace GMMSQArch35Tiling {
#pragma pack(push, 8)
struct GMMSQWeightQuantTilingData {
    uint8_t groupListType = 0;
    uint8_t reserve0[3] = {0};

    uint32_t groupNum = 0;
    uint32_t coreNum = 0;
    uint32_t groupSize = 0;

    uint64_t kSize = 0;
    uint64_t nSize = 0;
    // 总 M（x 的行数）。group_list 未必覆盖全部行（丢 token 场景），
    // 未被任何专家覆盖的尾部行由 controller 在末尾显式写 0。
    uint64_t mSize = 0;

    // 激活参数：仅 SiTU 通路读取；SwiGLU 通路不读取，默认 1.0 保持原行为不变
    float beta = 1.0f;
    float invBeta = 1.0f;
    float linearBeta = 1.0f;
    float invLinearBeta = 1.0f;
};
#pragma pack(pop)
} // namespace GMMSQArch35Tiling
#endif // GROUPED_MATMUL_SWIGLU_QUANT_V2_WEIGHT_QUANT_TILING_DATA_H
