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
 * \file dequant_swiglu_quant_dynamic_bias_int32.hpp
 * \brief
 */

#ifndef CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BIAS_INT32_HPP
#define CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BIAS_INT32_HPP

#include "kernel_operator.h"
#include "dequant_swiglu_quant_dynamic_base.hpp"

namespace DequantSwigluQuant {
using namespace AscendC;

constexpr int64_t BLOCK_BYTES = 32;

TEMPLATE_DECLARE
class DequantSwigluQuantDynamicBiasInt32 : public DequantSwigluQuantDynamicBase<TEMPLATE_ARGS>  {
public:
    __aicore__ inline DequantSwigluQuantDynamicBiasInt32(){};
    __aicore__ inline ~DequantSwigluQuantDynamicBiasInt32(){};

    __aicore__ inline void Init(GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm,
        GM_ADDR quant_scale_gm, GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
        GM_ADDR userspace, const SwiGluTilingData* tilingData, TPipe* pipe_);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength, uint32_t realRowLen);
};

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicBiasInt32<TEMPLATE_ARGS>::Init(
    GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm, GM_ADDR quant_scale_gm,
    GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm, GM_ADDR userspace, const SwiGluTilingData* tilingData,
    TPipe* pipe_) {
    this->InitCommon(x_gm, weight_scale_gm, activation_scale_gm, bias_gm, quant_scale_gm, quant_offset_gm, y_gm, scale_gm, userspace, tilingData, pipe_);
    if (this->activateScaleIsEmpty == 0) {
        this->activationScaleGm.SetGlobalBuffer((__gm__ float*) activation_scale_gm + this->biasOffset,
            this->numRound);
    }
    this->weightScaleGm.SetGlobalBuffer((__gm__ float*)weight_scale_gm, this->colNum);
    if (this->biasIsEmpty == 0) {
        this->biasGm.SetGlobalBuffer((__gm__ BiasType*)bias_gm, this->colNum);
    }
    this->InitUbBufferCommon(this->baseColLen, this->numRound);
    InitUbBuffer(this->baseColLen, this->numRound);
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicBiasInt32<TEMPLATE_ARGS>::Process() {
    this->BaseProcess();
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicBiasInt32<TEMPLATE_ARGS>::InitUbBuffer(uint64_t tileLength,
    uint32_t realRowLen) {
    uint64_t alignTileLength = tileLength;
    if (!this->isOut32BAligned) {
        alignTileLength = this->Align(tileLength, sizeof(int8_t));
    }
    if (this->biasIsEmpty == 0) {
        this->pipe->InitBuffer(this->inBiasQueueA, 1, alignTileLength * sizeof(BiasType));
        this->pipe->InitBuffer(this->inBiasQueueB, 1, alignTileLength * sizeof(BiasType) );
    }
    this->pipe->InitBuffer(this->weightScaleQueueA, 1, alignTileLength * sizeof(float));
    this->pipe->InitBuffer(this->weightScaleQueueB, 1, alignTileLength * sizeof(float));
    if (this->activateScaleIsEmpty == 0) {
        this->pipe->InitBuffer(this->inQueueActivationScale, 1, this->baseRowLen * sizeof(float));
    }
}
}  // namespace DequantSwigluQuant
#endif  // CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BIAS_INT32_HPP
