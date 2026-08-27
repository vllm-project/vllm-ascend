/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file recurrent_gated_delta_rule.cpp
 * \brief
 */
#ifndef RECURRENT_GATED_DELTA_RULE_TILING_DATA_H
#define RECURRENT_GATED_DELTA_RULE_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace RecurrentGatedDeltaRule {
#pragma pack(push, 8)
struct alignas(8) RecurrentGatedDeltaRuleTilingData { // alignas(8)确保8字节对齐
    uint32_t vectorCoreNum;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t t;
    uint32_t nk;
    uint32_t dk;
    uint32_t nv;
    uint32_t dv;
    uint32_t sBlockNum;
    uint32_t b;
    uint32_t vStep;
    uint32_t stateOutBufferNum;
    uint32_t attnOutBufferNum;
    float scale;
    uint32_t hasGama;
    uint32_t hasGamaK;
    uint32_t hasAcceptedTokens;
};
#pragma pack(pop)
} // RecurrentGatedDeltaRule

#endif // RECURRENT_GATED_DELTA_RULE_TILING_DATA_H
