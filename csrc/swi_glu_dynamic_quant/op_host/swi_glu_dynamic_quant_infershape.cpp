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
 * \file swi_glu_dynamic_quant_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace {
constexpr size_t GLU_IN_X = 0;
constexpr size_t GLU_OUT_Y = 0;
constexpr size_t GLU_OUT_Y2 = 1;
constexpr size_t GLU_ATTR_DIM = 0;
constexpr size_t ATTR_DST_TYPE = 3;
const size_t SPLIT_NUM = 2;
}  // namespace

namespace ops {
static ge::graphStatus InferShapeForSwiGluDynamicQuant(gert::InferShapeContext* context) {
    auto x_shape = context->GetInputShape(GLU_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto y_shape = context->GetOutputShape(GLU_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    auto scale_shape = context->GetOutputShape(GLU_OUT_Y2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale_shape);
    int64_t in_dim = x_shape->GetDimNum();
    int64_t split_dim = (static_cast<int64_t>(in_dim) - 1);
    if (split_dim < 0 || split_dim >= static_cast<int64_t>(x_shape->GetDimNum())) {
      D_OP_LOGE("SwiGluDynamicQuant", "The value of attr [dim] must be in the range [-%zu, %zu], but got [%ld].",
                  x_shape->GetDimNum(), x_shape->GetDimNum() - 1, split_dim);
      return GRAPH_FAILED;
    }
    *y_shape = *x_shape;
    // dynamic shape
    if (x_shape->GetDim(split_dim) == -1) {
        return ge::GRAPH_SUCCESS;
    }
    if (x_shape->GetDim(split_dim) < 0 || x_shape->GetDim(split_dim) % SPLIT_NUM != 0) {
        D_OP_LOGE("SwiGluDynamicQuant", "The shape [%s] is not divisible by 2.", Ops::Base::ToString(*x_shape).c_str());
        return ge::GRAPH_FAILED;
    }
    y_shape->SetDim(split_dim, x_shape->GetDim(split_dim) / SPLIT_NUM);
    scale_shape->SetDimNum(0);
    for (int i = 0; i< in_dim - 1; i++){
      scale_shape->AppendDim(x_shape->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSwiGluDynamicQuant(gert::InferDataTypeContext *context) {
    OP_LOGD(context, "====Enter SwiGluDynamicQuant inferDataType impl.=====");
    auto dstTypePtr = context->GetAttrs()->GetInt(ATTR_DST_TYPE);
    ge::DataType dstType = static_cast<ge::DataType>(*dstTypePtr);
    context->SetOutputDataType(0, dstType);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    OP_LOGD(context, "====Enter SwiGluDynamicQuant inferDataType impl end.=====");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwiGluDynamicQuant).InferShape(InferShapeForSwiGluDynamicQuant).InferDataType(InferDataTypeForSwiGluDynamicQuant);
}  // namespace ops
