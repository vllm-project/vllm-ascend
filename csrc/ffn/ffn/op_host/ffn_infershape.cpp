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
 * \file ffn_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

#include "ffn_layout.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShapeFFN(gert::InferShapeContext *context)
{
    // 布局识别需要激活类型（swiglu 的隐藏宽是 w1 边长的一半）
    bool isSwiglu = false;
    const auto *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const char *act = attrs->GetAttrPointer<char>(0);
        if (act != nullptr) {
            constexpr char swigluName[] = "swiglu";
            size_t i = 0;
            while (i < sizeof(swigluName) - 1 &&
                   tolower(static_cast<unsigned char>(act[i])) == swigluName[i]) {
                ++i;
            }
            isSwiglu = (i == sizeof(swigluName) - 1) && (act[i] == '\0');
        }
    }
    auto in_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
    // 输出最后一维 = weight2 的 N：
    //  - canonical [K, N]：N 在最后一维（arch22 原语义）
    //  - linear [N, K]（out, in）：N 在倒数第二维（arch35 与 PyTorch Linear 对齐）
    auto weight2_shape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, weight2_shape);
    auto weight1_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, weight1_shape);
    size_t xDimNum = in_shape->GetDimNum();
    size_t w1DimNum = weight1_shape->GetDimNum();
    size_t w2DimNum = weight2_shape->GetDimNum();
    if (xDimNum > 0 && w1DimNum >= 2 && w2DimNum >= 2) {
        int64_t xK = in_shape->GetDim(xDimNum - 1);
        int64_t w1d0 = weight1_shape->GetDim(w1DimNum - 2);
        int64_t w1d1 = weight1_shape->GetDim(w1DimNum - 1);
        int64_t w2d0 = weight2_shape->GetDim(w2DimNum - 2);
        int64_t w2d1 = weight2_shape->GetDim(w2DimNum - 1);
        // 布局识别走公共规则（ffn_layout.h）；INVALID/全方阵按 linear 计算，
        // 非法组合由 aclnn/tiling 层拦截报错。
        const bool isLinear = ffnlayout::FfnDetectLayout(w1d0, w1d1, w2d0, w2d1, xK, isSwiglu) ==
                              ffnlayout::FfnLayout::LINEAR;
        int64_t n = isLinear ? w2d0 : w2d1;
        out_shape->SetDim(xDimNum - 1, n);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFFN(gert::InferDataTypeContext *context)
{
    auto input_x_dtype = context->GetInputDataType(0);
    if (input_x_dtype == ge::DT_INT8) {
        auto attrs = context->GetAttrs();
        const int64_t *output_dtype = attrs->GetInt(2);
        if (output_dtype != nullptr && *output_dtype == 1) {
            context->SetOutputDataType(0, ge::DT_BF16);
        } else {
            context->SetOutputDataType(0, ge::DT_FLOAT16);
        }
    } else {
        context->SetOutputDataType(0, input_x_dtype);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FFN).InferShape(InferShapeFFN).InferDataType(InferDataTypeFFN);
} // namespace ops
