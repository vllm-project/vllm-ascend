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
 * \file swi_glu_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SWIGLU_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SWIGLU_H_

#include <cstdint>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../tiling_base/tiling_templates_registry.h"
#include "swi_glu_grad_regbase_tiling.h"

using namespace Ops::NN::Optiling;

namespace optiling {
const int64_t STATIC_FLOAT16_X = 10000;
const int64_t STATIC_BFLOAT16_X = 10001;
const int64_t STATIC_FLOAT16_XD = 10002;
const int64_t STATIC_BFLOAT16_XD = 10003;
const int64_t STATIC_INT_X_INT_BIAS_QUANT_ONE = 10004;
const int64_t STATIC_INT_X_INT_BIAS_QUANT_D = 10005;
const int64_t STATIC_INT_X_FLOAT16_BIAS_QUANT_ONE = 10006;
const int64_t STATIC_INT_X_FLOAT16_BIAS_QUANT_D = 10007;
const int64_t STATIC_INT_X_FLOAT32_BIAS_QUANT_ONE = 10008;
const int64_t STATIC_INT_X_FLOAT32_BIAS_QUANT_D = 10009;
const int64_t STATIC_INT_X_BFLOAT16_BIAS_QUANT_ONE = 10010;
const int64_t STATIC_INT_X_BFLOAT16_BIAS_QUANT_D = 10011;

const int64_t DYNAMIC_FLOAT16_X = 30009;
const int64_t DYNAMIC_BFLOAT16_X = 30011;
const int64_t DYNAMIC_FLOAT16_XD = 30010;
const int64_t DYNAMIC_BFLOAT16_XD = 30012;
const int64_t DYNAMIC_INT_X_INT_BIAS_QUANT_ONE = 30001;
const int64_t DYNAMIC_INT_X_INT_BIAS_QUANT_D = 30005;
const int64_t DYNAMIC_INT_X_FLOAT16_BIAS_QUANT_ONE = 30003;
const int64_t DYNAMIC_INT_X_FLOAT16_BIAS_QUANT_D = 30007;
const int64_t DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_ONE = 30002;
const int64_t DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_D = 30006;
const int64_t DYNAMIC_INT_X_BFLOAT16_BIAS_QUANT_ONE = 30004;
const int64_t DYNAMIC_INT_X_BFLOAT16_BIAS_QUANT_D = 30008;

BEGIN_TILING_DATA_DEF(SwiGluTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, is32BAligned);
    TILING_DATA_FIELD_DEF(uint32_t, isDoubleBuffer);
    TILING_DATA_FIELD_DEF(uint64_t, rowLen);
    TILING_DATA_FIELD_DEF(uint64_t, colLen);
    TILING_DATA_FIELD_DEF(uint32_t, baseRowLen);
    TILING_DATA_FIELD_DEF(uint32_t, baseColLen);
    TILING_DATA_FIELD_DEF(uint32_t, activateLeft);
    TILING_DATA_FIELD_DEF(uint32_t, biasIsEmpty);
    TILING_DATA_FIELD_DEF(uint32_t, quantScaleIsEmpty);
    TILING_DATA_FIELD_DEF(uint32_t, activateScaleIsEmpty);
    TILING_DATA_FIELD_DEF(uint64_t, swiColLen);
    TILING_DATA_FIELD_DEF(uint64_t, perRowLen);
    TILING_DATA_FIELD_DEF(uint64_t, modRowLen);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwiGlu, SwiGluTilingData)
REGISTER_TILING_DATA_CLASS(SwiGluGrad, SwiGluTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant, SwiGluTilingData)

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_SWIGLU_H_
