/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dequant_swiglu_quant_dynamic_bf16.hpp
 * \brief
 */

#ifndef CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BF16_HPP
#define CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BF16_HPP

#include "kernel_operator.h"
#include "dequant_swiglu_quant_dynamic_base.hpp"

namespace DequantSwigluQuant {
using namespace AscendC;

TEMPLATE_DECLARE
class DequantSwigluQuantDynamicBF16 : public DequantSwigluQuantDynamicBase<TEMPLATE_ARGS> {
public:
    __aicore__ inline DequantSwigluQuantDynamicBF16(){};
    __aicore__ inline ~DequantSwigluQuantDynamicBF16(){};

    __aicore__ inline void Init(GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm,
        GM_ADDR quant_scale_gm, GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
        GM_ADDR userspace, const SwiGluTilingData* tilingData, TPipe* pipe_);
    __aicore__ inline void Process();
};

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicBF16<TEMPLATE_ARGS>::Init(
    GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm, GM_ADDR quant_scale_gm,
    GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm, GM_ADDR userspace, const SwiGluTilingData* tilingData,
    TPipe* pipe_) {
    this->InitCommon(x_gm, weight_scale_gm, activation_scale_gm, bias_gm, quant_scale_gm, quant_offset_gm, y_gm, scale_gm, userspace, tilingData, pipe_);

    if (this->numRound < this->baseRowLen) {
        this->baseRowLen = this->numRound;
    }
    this->InitUbBufferCommon(this->baseColLen, this->numRound);
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicBF16<TEMPLATE_ARGS>::Process() {
    this->BaseProcess();
}
}  // namespace DequantSwigluQuant
#endif  // CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BF16_HPP
