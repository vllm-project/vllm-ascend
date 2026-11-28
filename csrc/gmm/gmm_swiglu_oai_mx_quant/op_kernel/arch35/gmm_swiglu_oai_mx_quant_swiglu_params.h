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
 * \file gmm_swiglu_oai_mx_quant_swiglu_params.h
 * \brief SwiGLU attributes appended to the Ascend 950 tiling payload.
 */

#ifndef VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_SWIGLU_PARAMS_H
#define VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_SWIGLU_PARAMS_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace GmmSwigluOaiMxQuantTensorApi {
constexpr float kDefaultClampLimit = 7.0F;
constexpr float kDefaultGluAlpha = 1.702F;
constexpr float kDefaultGluBias = 1.0F;
constexpr uint32_t kSwigluParamsBytes = 32;

#pragma pack(push, 8)
struct GmmSwigluOaiMxQuantSwigluParams {
    int64_t swigluMode = 0;
    float clampLimit = kDefaultClampLimit;
    float gluAlpha = kDefaultGluAlpha;
    float gluBias = kDefaultGluBias;
    float dstTypeMax = 0.0F;
    uint8_t scaleAlg = 0;
    uint8_t roundMode = 0;
    uint16_t mxScalePreinitialized = 0;
    uint32_t reserved1 = 0;
};
#pragma pack(pop)

static_assert(sizeof(GmmSwigluOaiMxQuantSwigluParams) == kSwigluParamsBytes,
              "SwiGLU tiling attributes must occupy 32 bytes");
} // namespace GmmSwigluOaiMxQuantTensorApi

#endif // VLLM_ASCEND_GMM_SWIGLU_OAI_MX_QUANT_SWIGLU_PARAMS_H
