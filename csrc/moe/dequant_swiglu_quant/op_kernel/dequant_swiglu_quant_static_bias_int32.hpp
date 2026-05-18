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
 * \file dequant_swiglu_quant_static_bias_int32.hpp
 * \brief
 */

#ifndef CANN_DEQUANT_SWIGLU_QUANT_STATIC_BIAS_INT32_HPP
#define CANN_DEQUANT_SWIGLU_QUANT_STATIC_BIAS_INT32_HPP
#include "kernel_operator.h"
#include "dequant_swiglu_quant_static_base.hpp"
namespace DequantSwigluQuant {
using namespace AscendC;

TEMPLATE_DECLARE_STATIC
class DequantSwigluQuantStaticBiasInt32 : public DequantSwigluQuantStaticBase<TEMPLATE_ARGS_STATIC> {
public:
    __aicore__ inline DequantSwigluQuantStaticBiasInt32() {}
    __aicore__ inline ~DequantSwigluQuantStaticBiasInt32() {}
    __aicore__ inline void Init(GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm,
        GM_ADDR quant_scale_gm, GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
        const SwiGluTilingData* tilingData, TPipe* pipe_);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitUbBuffer();
private:
};

TEMPLATE_DECLARE_STATIC
__aicore__ inline void DequantSwigluQuantStaticBiasInt32<TEMPLATE_ARGS_STATIC>::Init(GM_ADDR x_gm, GM_ADDR weight_scale_gm,
    GM_ADDR activation_scale_gm, GM_ADDR bias_gm,
    GM_ADDR quant_scale_gm, GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
    const SwiGluTilingData* tilingData, TPipe* pipe_)
{
    this->pipe = pipe_;
    this->InitCommon(x_gm, weight_scale_gm, activation_scale_gm, bias_gm, quant_scale_gm, quant_offset_gm, y_gm, scale_gm, tilingData, pipe_);

    if (this->activateScaleIsEmpty == 0) {
        this->activationScaleGm.SetGlobalBuffer((__gm__ float*) activation_scale_gm + this->inputCopyOffset,
            this->curCoreRowNum);
    }
    this->weightScaleGm.SetGlobalBuffer((__gm__ float*) weight_scale_gm, this->colNum);
    if (tilingData->biasIsEmpty == 0) {
        this->biasGm.SetGlobalBuffer((__gm__ BiasType*)bias_gm, this->colNum);
    }
    this->InitUbBufferCommon();
    this->InitUbBuffer();
}

TEMPLATE_DECLARE_STATIC
__aicore__ inline void DequantSwigluQuantStaticBiasInt32<TEMPLATE_ARGS_STATIC>::Process()
{
    if (this->blockIdx >= this->usedCoreNum) {
        return;
    }
    this->processCompute();
}
TEMPLATE_DECLARE_STATIC
__aicore__ inline void DequantSwigluQuantStaticBiasInt32<TEMPLATE_ARGS_STATIC>::InitUbBuffer()
{
    int64_t alignColNumber = this->curColNum == this->Align(this->curColNum, sizeof(InType))
                                 ? this->curColNum
                                 : this->Align(this->curColNum, sizeof(OutType));

    this->pipe->InitBuffer(this->inQueueWeightScale, bufferNum, alignColNumber * sizeof(float) * NUM2);
    if (this->activateScaleIsEmpty == 0) {
        this->pipe->InitBuffer(this->inQueueActivationScale, bufferNum, this->curCoreRowNum * sizeof(float));
    }
    if (this->biasIsEmpty == 0) {
        this->pipe->InitBuffer(this->inQueueBias, bufferNum, alignColNumber * sizeof(float) * NUM2);
    }
}

}
#endif  // CANN_DEQUANT_SWIGLU_QUANT_STATIC_BIAS_INT32_HPP
