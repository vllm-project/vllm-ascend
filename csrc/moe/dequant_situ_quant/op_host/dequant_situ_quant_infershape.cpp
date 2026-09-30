/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dequant_situ_quant_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "graph/utils/type_utils.h"
#include "util/shape_util.h"
#include "log/log.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_X = 0;
constexpr size_t OUTPUT_IDX_Y = 0;
constexpr size_t OUTPUT_IDX_SCALE = 1;
constexpr int64_t CONST_UNKNOW_SHAPE = -1;
constexpr int64_t NUM_TWO = 2;

graphStatus InferShape4DequantSituQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4DequantSituQuant.");

    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape* scaleShape = context->GetOutputShape(OUTPUT_IDX_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

    *yShape = *xShape;
    OP_CHECK_IF(Ops::Base::IsUnknownRank(*xShape),
                OP_LOGD(context, "End to do InferShape4DequantSituQuant, inputx is [-2]."), return GRAPH_SUCCESS);

    int64_t xShapeRank = static_cast<int64_t>(xShape->GetDimNum());

    auto inputDesc = context->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType xDataType = inputDesc->GetDataType();

    bool isInt8 = (xDataType == ge::DT_INT8);
    if (isInt8) {
        OP_CHECK_IF(xShapeRank <= 1,
                    OP_LOGE(context, "x shape rank must > 1 for int8, but is %ld", xShapeRank),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(xShapeRank != 2,
                    OP_LOGE(context, "x shape rank must be 2 for int32/bfloat16, but is %ld", xShapeRank),
                    return ge::GRAPH_FAILED);
    }

    int64_t lastDim = xShape->GetDim(xShapeRank - 1);
    int64_t outLastDim = lastDim == CONST_UNKNOW_SHAPE ? CONST_UNKNOW_SHAPE : lastDim / NUM_TWO;
    OP_CHECK_IF((lastDim != CONST_UNKNOW_SHAPE) && (lastDim % NUM_TWO != 0),
                OP_LOGE(context, "The last dim of x must be even number, but is %ld", lastDim),
                return ge::GRAPH_FAILED);

    yShape->SetDim(xShapeRank - 1, outLastDim);

    *scaleShape = *yShape;
    if (isInt8) {
        scaleShape->SetDimNum(xShapeRank - 1);
    } else {
        scaleShape->SetDimNum(1);
        scaleShape->SetDim(0, xShape->GetDim(0));
    }

    OP_LOGD(context, "End to do InferShape4DequantSituQuant");
    return GRAPH_SUCCESS;
}

graphStatus InferDtype4DequantSituQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDtype4DequantSituQuant enter");
    context->SetOutputDataType(OUTPUT_IDX_Y, DT_INT8);
    context->SetOutputDataType(OUTPUT_IDX_SCALE, DT_FLOAT);
    OP_LOGD(context, "InferDtype4DequantSituQuant end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DequantSituQuant)
    .InferShape(InferShape4DequantSituQuant)
    .InferDataType(InferDtype4DequantSituQuant);
} // namespace ops
