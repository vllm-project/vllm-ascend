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
 * \file gmm_swiglu_oai_mx_quant_tiling_data.h
 * \brief Tiling data for the GmmSwigluOaiMxQuant Tensor API kernel.
 */

#ifndef VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_TILING_DATA_H
#define VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_TILING_DATA_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#include "gmm_swiglu_oai_mx_quant_swiglu_params.h"

namespace GmmSwigluOaiMxQuantTensorApi {
#pragma pack(push, 8)
struct GmmTensorApiQuantParams {
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
struct GmmTensorApiMMTiling {
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
struct GmmSwigluOaiMxQuantTilingData {
    GmmTensorApiQuantParams gmmQuantParams;
    GmmTensorApiMMTiling mmTilingData;
    GmmSwigluOaiMxQuantSwigluParams swigluParams;
};
#pragma pack(pop)
} // namespace GmmSwigluOaiMxQuantTensorApi

#endif // VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_TILING_DATA_H
