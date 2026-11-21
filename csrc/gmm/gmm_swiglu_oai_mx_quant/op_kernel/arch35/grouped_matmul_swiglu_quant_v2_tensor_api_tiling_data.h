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
 * \file grouped_matmul_swiglu_quant_v2_tensor_api_tiling_data.h
 * \brief Tiling data for the MX FP8 Tensor API kernel.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_TENSOR_API_TILING_DATA_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_TENSOR_API_TILING_DATA_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#include "grouped_matmul_swiglu_quant_swiglu_params.h"

namespace GroupedMatmulSwigluQuantV2TensorApi {
#pragma pack(push, 8)
struct GMMTensorApiQuantParams {
    uint32_t groupNum = 0;
    uint32_t activeType = 0;
    uint32_t aQuantMode = 0;
    uint32_t bQuantMode = 0;
    uint8_t singleX = 0;
    uint8_t singleW = 0;
    uint8_t singleY = 0;
    int8_t groupType = 0;
    uint8_t groupListType = 0;
    uint8_t hasBias = 0;
    uint16_t reserved = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMTensorApiMMTiling {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t kAL1 = 0;
    uint32_t kBL1 = 0;
    uint32_t scaleKAL1 = 0;
    uint32_t scaleKBL1 = 0;
    uint8_t isBias = 0;
    uint8_t dbL0C = 0;
    uint16_t reserved1 = 0;
    uint32_t reserved2 = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMSwigluQuantV2TensorApiTilingData {
    GMMTensorApiQuantParams gmmQuantParams;
    GMMTensorApiMMTiling mmTilingData;
    GMMSwigluQuantSwigluParams swigluParams;
};
#pragma pack(pop)
} // namespace GroupedMatmulSwigluQuantV2TensorApi

#endif // GROUPED_MATMUL_SWIGLU_QUANT_V2_TENSOR_API_TILING_DATA_H
