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
 * \file dequant_swiglu_quant.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#if (ORIG_DTYPE_X == DT_INT32) || (ORIG_DTYPE_X == DT_BF16)
  #include "dequant_swiglu_quant.h"
  #include "dequant_swiglu_quant_cut_group.h"
#endif

#include "dequant_swiglu_quant_static_bf16.hpp"
#include "dequant_swiglu_quant_static_bias_int32.hpp"
#include "dequant_swiglu_quant_static_bias_float.hpp"
#include "dequant_swiglu_quant_dynamic_bf16.hpp"
#include "dequant_swiglu_quant_dynamic_bias_int32.hpp"
#include "dequant_swiglu_quant_dynamic_bias_float.hpp"
#include "dequant_swiglu_quant_dynamic_performance.hpp"

using namespace AscendC;

// DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP32_QS HAS_GROUP(100000000) + QS_OFFSET(100) * QS_FP32(0)
// DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP16_QS HAS_GROUP(100000000) + QS_OFFSET(100) * QS_FP16(1)
// DEQUANT_SWIGLU_QUANT_WITH_GROUP_BF16_QS HAS_GROUP(100000000) + QS_OFFSET(100) * QS_BF16(2)
// DEQUANT_SWIGLU_QUANT_WITHOUT_GROUP_FP32_QS NO_GROUP(200000000) + QS_OFFSET(100) * QS_FP32(0)
// DEQUANT_SWIGLU_QUANT_WITHOUT_GROUP_FP16_QS NO_GROUP(200000000) + QS_OFFSET(100) * QS_FP16(1)
// DEQUANT_SWIGLU_QUANT_WITHOUT_GROUP_BF16_QS NO_GROUP(200000000) + QS_OFFSET(100) * QS_BF16(2)
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_BF16_BIAS_FP32_QS 100000000
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_NO_BIAS_FP32_QS DEQUANT_SWIGLU_QUANT_WITH_GROUP_BF16_BIAS_FP32_QS
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_BF16_BIAS_FP16_QS 100000100
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_NO_BIAS_FP16_QS DEQUANT_SWIGLU_QUANT_WITH_GROUP_BF16_BIAS_FP16_QS
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_BF16_BIAS_BF16_QS 100000200
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_NO_BIAS_BF16_QS DEQUANT_SWIGLU_QUANT_WITH_GROUP_BF16_BIAS_BF16_QS
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP16_BIAS_FP32_QS 100001000
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP16_BIAS_FP16_QS 100001100
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP16_BIAS_BF16_QS 100001200
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP32_BIAS_FP32_QS 100002000
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP32_BIAS_FP16_QS 100002100
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP32_BIAS_BF16_QS 100002200
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_INT32_BIAS_FP32_QS 100003000
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_INT32_BIAS_FP16_QS 100003100
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_INT32_BIAS_BF16_QS 100003200
#define DEQUANT_SWIGLU_QUANT_WITHOUT_GROUP_FP32_QS 200000000
#define DEQUANT_SWIGLU_QUANT_WITHOUT_GROUP_FP16_QS 200000100
#define DEQUANT_SWIGLU_QUANT_WITHOUT_GROUP_BF16_QS 200000200
// cut by groupnum
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP32_QS_GR 110000000
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP16_QS_GR 110000100
#define DEQUANT_SWIGLU_QUANT_WITH_GROUP_BF16_QS_GR 110000200

extern "C" __global__ __aicore__ void dequant_swiglu_quant(GM_ADDR xGM, GM_ADDR weightSscaleGM,
                                                           GM_ADDR activationScaleGM, GM_ADDR biasGM,
                                                           GM_ADDR quantScaleGM, GM_ADDR quantOffsetGM,
                                                           GM_ADDR groupIndex, GM_ADDR yGM, GM_ADDR scaleGM,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
  if (workspace == nullptr) {
    return;
  }

  GM_ADDR userspace = GetUserWorkspace(workspace);
  if (userspace == nullptr) {
    return;
  }
  TPipe pipe;
#if (ORIG_DTYPE_X == DT_INT32)
  if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_BF16_BIAS_FP32_QS)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<bfloat16_t, float, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
#endif
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_BF16_BIAS_FP16_QS)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<bfloat16_t, half, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
#endif
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_BF16_BIAS_BF16_QS)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<bfloat16_t, bfloat16_t, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
#endif
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP16_BIAS_FP32_QS)) {
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<half, float, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP16_BIAS_FP16_QS)) {
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<half, half, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP16_BIAS_BF16_QS)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<half, bfloat16_t, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
#endif
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP32_BIAS_FP32_QS)) {
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<float, float, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP32_BIAS_FP16_QS)) {
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<float, half, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP32_BIAS_BF16_QS)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<float, bfloat16_t, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
#endif
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_INT32_BIAS_FP32_QS)) {
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<int32_t, float, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_INT32_BIAS_FP16_QS)) {
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<int32_t, half, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_INT32_BIAS_BF16_QS)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<int32_t, bfloat16_t, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
#endif
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITHOUT_GROUP_FP32_QS)) {
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    // DTYPE_GROUP_INDEX == float mean have no groupIndex
    DequantSwigluQuantOps::DequantSwigluQuantBase<float, float, float, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, nullptr, quantScaleGM, nullptr, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITHOUT_GROUP_FP16_QS)) {
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    // DTYPE_GROUP_INDEX == float mean have no groupIndex
    DequantSwigluQuantOps::DequantSwigluQuantBase<float, half, float, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, nullptr, quantScaleGM, nullptr, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITHOUT_GROUP_BF16_QS)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    // DTYPE_GROUP_INDEX == float mean have no groupIndex
    DequantSwigluQuantOps::DequantSwigluQuantBase<float, bfloat16_t, float, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, nullptr, quantScaleGM, nullptr, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
#endif
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP32_QS_GR)) {
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantGroupOps::DequantSwigluQuantGroup<bfloat16_t, float, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_FP16_QS_GR)) {
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantGroupOps::DequantSwigluQuantGroup<bfloat16_t, half, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_BF16_QS_GR)) {
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantGroupOps::DequantSwigluQuantGroup<bfloat16_t, bfloat16_t, int64_t, int32_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM, tilingData);
    op.Process();
#endif
  } else if (TILING_KEY_IS(10004)) {
    // ORIG_DTYPE_BIAS == DT_INT32
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantStaticBiasInt32<int32_t, float, int32_t, int8_t, 1, 1> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, tilingData,
            &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(10005)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantStaticBiasInt32<int32_t, float, int32_t, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, tilingData,
            &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(30001)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantDynamicBiasInt32<int32_t, float, int32_t, int8_t, 1, 1> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace,
            tilingData, &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(30005)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantDynamicBiasInt32<int32_t, float, int32_t, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace,
            tilingData, &(pipe));
    op.Process();
  }
  // ORIG_DTYPE_BIAS == DT_FLOAT16
  else if (TILING_KEY_IS(10006)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantStaticBiasFloat<int32_t, float, half, int8_t, 1, 1> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, tilingData,
            &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(10007)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantStaticBiasFloat<int32_t, float, half, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, tilingData,
            &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(30003)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantDynamicBiasFloat<int32_t, float, half, int8_t, 1, 1> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace,
            tilingData, &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(30007)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantDynamicBiasFloat<int32_t, float, half, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace,
            tilingData, &(pipe));
    op.Process();
  }
  // ORIG_DTYPE_BIAS == DT_FLOAT
  else if (TILING_KEY_IS(10008)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantStaticBiasFloat<int32_t, float, float, int8_t, 1, 1> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, tilingData,
            &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(10009)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantStaticBiasFloat<int32_t, float, float, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, tilingData,
            &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(30002)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantDynamicBiasFloat<int32_t, float, float, int8_t, 1, 1> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace,
            tilingData, &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(30013)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantDynamicPerformance<int32_t, float, float, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace,
            tilingData, &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(30006)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantDynamicBiasFloat<int32_t, float, float, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace,
            tilingData, &(pipe));
    op.Process();
  }
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
  // ORIG_DTYPE_BIAS == DT_BF16
  else if (TILING_KEY_IS(10010)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantStaticBiasFloat<int32_t, float, bfloat16_t, int8_t, 1, 1> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, tilingData,
            &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(10011)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantStaticBiasFloat<int32_t, float, bfloat16_t, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, tilingData,
            &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(30004)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantDynamicBiasFloat<int32_t, float, bfloat16_t, int8_t, 1, 1> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace,
            tilingData, &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(30008)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantDynamicBiasFloat<int32_t, float, bfloat16_t, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace,
            tilingData, &(pipe));
    op.Process();
  }
#endif
#endif
#if (ORIG_DTYPE_X == DT_FLOAT16)
  if (TILING_KEY_IS(10000)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantStaticBF16<half, float, half, int8_t, 1, 1> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, tilingData,
            &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(10002)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantStaticBF16<half, float, half, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, tilingData,
            &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(30009)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantDynamicBF16<half, float, half, int8_t, 1, 1> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace,
            tilingData, &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(30010)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantDynamicBF16<half, float, half, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace,
            tilingData, &(pipe));
    op.Process();
  }
#endif
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)) && (ORIG_DTYPE_X == DT_BF16)
  if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_NO_BIAS_FP32_QS)) {
    // New tiling branch for BF16
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<float, float, int64_t, bfloat16_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, nullptr, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
  }  else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_NO_BIAS_FP16_QS)) {
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<float, half, int64_t, bfloat16_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, nullptr, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
  } else if (TILING_KEY_IS(DEQUANT_SWIGLU_QUANT_WITH_GROUP_NO_BIAS_BF16_QS)) {
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantBaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantBaseTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuantOps::DequantSwigluQuantBase<float, bfloat16_t, int64_t, bfloat16_t> op(&pipe);
    op.Init(xGM, weightSscaleGM, activationScaleGM, nullptr, quantScaleGM, quantOffsetGM, groupIndex, yGM, scaleGM,
            tilingData);
    op.Process();
  } else if (TILING_KEY_IS(10001)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantStaticBF16<bfloat16_t, float, bfloat16_t, int8_t, 1, 1> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, tilingData,
            &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(10003)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantStaticBF16<bfloat16_t, float, bfloat16_t, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, tilingData,
            &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(30011)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantDynamicBF16<bfloat16_t, float, bfloat16_t, int8_t, 1, 1> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace,
            tilingData, &(pipe));
    op.Process();
  } else if (TILING_KEY_IS(30012)) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tilingDataIn, tiling);
    const SwiGluTilingData* __restrict__ tilingData = &tilingDataIn;
    DequantSwigluQuant::DequantSwigluQuantDynamicBF16<bfloat16_t, float, bfloat16_t, int8_t, 1, 0> op;
    op.Init(xGM, weightSscaleGM, activationScaleGM, biasGM, quantScaleGM, quantOffsetGM, yGM, scaleGM, userspace,
            tilingData, &(pipe));
    op.Process();
  }
#endif
}