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
 * \file mc2_moe_utils.h
 * \brief
 */
#ifndef OPS_COMMON_INC_MC2_MOE_UTIL_H
#define OPS_COMMON_INC_MC2_MOE_UTIL_H

#include "mc2_log.h"

namespace Mc2Moe {
const size_t SUPPORT_DIM_NUM = 3;      // E, C, H, x weight current only support 3-dim
const size_t BIAS_SUPPORT_DIM_NUM = 2; // bias can be 2 or 3 dims

// 对于 moe gather, x = [E, C, H], w = [E, H, M], weight 可能会转置，其 dim 设为局部变量
// 对于 moe scatter, x = [E, C, M], w = [E, M, H], weight 可能会转置，其 dim 设为局部变量
const size_t DIM_E = 0;
const size_t X_DIM_C = 1;
const size_t X_DIM_H = 2;
const size_t X_DIM_M = 2;

// 专家个数 E 需要在 [min, max] 之间，不是维度值
const int64_t VALUE_E_MIN = 2;
const int64_t VALUE_E_MAX = 512;
// hidden size H 需要在 [min, max] 之间，不是维度值
const int64_t VALUE_H_MIN = 1;
const int64_t VALUE_H_MAX = 65535;
const int64_t VALUE_C_MIN = 1;

struct OutShapeInfo {
    int64_t e;
    int64_t c;
    int64_t h;
};

// 当前 ep tp 仅支持 2 4 8 16
bool EpTpSizeCheck(const int64_t epSize, const int64_t tpSize);

bool DimNumCheck(const char *nodeName, const gert::Shape *xShape, const gert::Shape *weightShape);

bool GroupCheck(const char *nodeName, const char *groupEp, const char *groupTp);

void DynamicShapeCheck(const gert::Shape *xShape, const gert::Shape *weightShape, const size_t wDim,
                       OutShapeInfo &outShapeInfo);

void EmptyShapeCheck(const gert::Shape *xShape, const gert::Shape *weightShape, const size_t wDim,
                     OutShapeInfo &outShapeInfo);

bool CheckBiasDtype(const char *nodeName, const ge::DataType xType, const ge::DataType biasType);

bool CheckTensorDtype(const char *nodeName, const ge::DataType xType, const ge::DataType weightType,
                      const ge::DataType biasType);

void SetShape(gert::Shape *shape, const OutShapeInfo &outShapeInfo);

void TransDimHMIdx(const bool isTransW, size_t &wDimH, size_t &wDimM);

void PrintTensorShape(const char *nodeName, const gert::Shape *shape, const char *shapeName);

} // namespace Mc2Moe

#endif // OPS_COMMON_INC_MC2_MOE_UTIL_H
