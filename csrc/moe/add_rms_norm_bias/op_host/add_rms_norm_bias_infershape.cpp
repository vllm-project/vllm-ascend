/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_bias_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"

static constexpr int IDX_0 = 0;
static constexpr int IDX_1 = 1;
static constexpr int IDX_2 = 2;

using namespace ge;
using namespace Ops::Base;

namespace ops {

static ge::graphStatus InferShape4AddRmsNormBias(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4AddRmsNormBias");

    // get input shapes
    const gert::Shape* x1Shape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* gammaShape = context->GetInputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    gert::Shape* rstdShape = context->GetOutputShape(IDX_1);
    gert::Shape* xShape = context->GetOutputShape(IDX_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstdShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    *yShape = *x1Shape;
    *xShape = *x1Shape;

    size_t xDimNum = x1Shape->GetDimNum();
    size_t gammaDimNum = gammaShape->GetDimNum();

    if (IsUnknownRank(*x1Shape) || IsUnknownRank(*gammaShape)) {
        SetUnknownRank(*rstdShape);
        OP_LOGD(context, "End to do InferShape4AddRmsNormBias with unknown rank.");
        return GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        xDimNum < gammaDimNum, OP_LOGE(context, "x dim num should not be smaller than gamma dim num."),
        return GRAPH_FAILED);

    rstdShape->SetDimNum(xDimNum);
    for (size_t i = 0; i < xDimNum; i++) {
        if (i < xDimNum - gammaDimNum) {
            rstdShape->SetDim(i, x1Shape->GetDim(i));
        } else {
            rstdShape->SetDim(i, 1);
        }
    }

    OP_LOGD(context, "End to do InferShape4AddRmsNormBias");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4AddRmsNormBias(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4AddRmsNormBias");
    context->SetOutputDataType(IDX_0, context->GetInputDataType(IDX_0));
    context->SetOutputDataType(IDX_1, DT_FLOAT);
    context->SetOutputDataType(IDX_2, context->GetInputDataType(IDX_0));
    OP_LOGD(context, "End to do InferDataType4AddRmsNormBias");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddRmsNormBias).InferShape(InferShape4AddRmsNormBias).InferDataType(InferDataType4AddRmsNormBias);
} // namespace ops
