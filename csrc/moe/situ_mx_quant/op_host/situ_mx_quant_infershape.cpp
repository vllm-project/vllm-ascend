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
 * \file situ_mx_quant_infershape.cpp
 * \brief Shape inference for Situ + MX quantization operator
 */

#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"

using namespace ge;

namespace ops {
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr int64_t UNKNOWN_RANK_DIM = -2;
constexpr size_t INDEX_INPUT_X = 0;
constexpr size_t INDEX_OUTPUT_Y = 0;
constexpr size_t INDEX_OUTPUT_MXSCALE = 1;

constexpr size_t INDEX_ATTR_BETA = 0;
constexpr size_t INDEX_ATTR_LINEAR_BETA = 1;
constexpr size_t INDEX_ATTR_ACTIVATE_LEFT = 2;
constexpr size_t INDEX_ATTR_AXIS = 3;
constexpr size_t INDEX_ATTR_DST_TYPE = 4;

constexpr int64_t SPLIT_NUM = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t ALIGN_NUM = 2;
constexpr size_t MAX_DIM_NUM = 7;

static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

graphStatus InferShapeForSituMxQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForSituMxQuant");
    const gert::Shape* xShape = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape* mxscaleShape = context->GetOutputShape(INDEX_OUTPUT_MXSCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, mxscaleShape);

    OP_CHECK_IF(xShape->GetDimNum() < 1 || xShape->GetDimNum() > MAX_DIM_NUM,
                OP_LOGE(context->GetNodeName(), "Input x rank[%lu] should be in [1, 7].", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    if (xShape->GetDimNum() == 1 && xShape->GetDim(0) == UNKNOWN_RANK_DIM) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y, mxscale shape to (-2, )");
        *yShape = *xShape;
        *mxscaleShape = *xShape;
        return ge::GRAPH_SUCCESS;
    }

    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);

    const int64_t* axis = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);

    int64_t xRank = xShape->GetDimNum();
    int64_t lastDimIdx = xRank - 1;
    int64_t axisNorm = (*axis >= 0) ? static_cast<int64_t>(*axis) : static_cast<int64_t>(*axis + xRank);
    OP_CHECK_IF(axisNorm != lastDimIdx,
                OP_LOGE(context->GetNodeName(), "axis must be -1 (last axis), but got axis=%ld (normalized=%ld)",
                        *axis, axisNorm),
                return ge::GRAPH_FAILED);

    // Validate last dimension is divisible by 2
    if (xShape->GetDim(lastDimIdx) != UNKNOWN_DIM_VALUE_ && xShape->GetDim(lastDimIdx) % SPLIT_NUM != 0) {
        OP_LOGE(context->GetNodeName(), "The last dimension must be divisible by 2, but got [%ld].",
                xShape->GetDim(lastDimIdx));
        return ge::GRAPH_FAILED;
    }

    // Step 1: Compute y shape (Situ output)
    // y 的 shape 与 x 的 shape 维度一致，在最后一维上是 x 的一半
    *yShape = *xShape;
    if (xShape->GetDim(lastDimIdx) != UNKNOWN_DIM_VALUE_) {
        yShape->SetDim(lastDimIdx, xShape->GetDim(lastDimIdx) / SPLIT_NUM);
    }

    // Step 2: Compute mxscale shape
    // mxscale.shape = y.shape
    // mxscale.shape[axis] = CeilDiv(y.shape[axis], 2 * 32)  (even-aligned block count)
    // mxscale.shape += [2]
    *mxscaleShape = *yShape;
    int64_t yAxisSize = 0;
    if (yShape->GetDim(axisNorm) == UNKNOWN_DIM_VALUE_) {
        yAxisSize = UNKNOWN_DIM_VALUE_;
    } else {
        int64_t yDim = yShape->GetDim(axisNorm);
        yAxisSize = Ops::Base::CeilDiv(yDim, ALIGN_NUM * BLOCK_SIZE);
    }
    mxscaleShape->SetDim(axisNorm, yAxisSize);
    mxscaleShape->AppendDim(ALIGN_NUM);

    OP_LOGI(context->GetNodeName(), "End to do InferShapeForSituMxQuant");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForSituMxQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForSituMxQuant");
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int64_t* dstDtype = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstDtype);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    OP_CHECK_IF(
        std::find(Y_SUPPORT_DTYPE_SET.begin(), Y_SUPPORT_DTYPE_SET.end(), outDtype) == Y_SUPPORT_DTYPE_SET.end(),
        OP_LOGE(context->GetNodeName(),
                "dst_type is illegal, only supports 36(FLOAT8_E4M3FN) or 35(FLOAT8_E5M2). but got %d(%s) please check.",
                *dstDtype, ge::TypeUtils::DataTypeToAscendString(outDtype).GetString()),
        return ge::GRAPH_FAILED);
    context->SetOutputDataType(INDEX_OUTPUT_Y, outDtype);
    context->SetOutputDataType(INDEX_OUTPUT_MXSCALE, ge::DT_FLOAT8_E8M0);
    OP_LOGI(context->GetNodeName(), "End to do InferDataTypeForSituMxQuant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SituMxQuant).InferShape(InferShapeForSituMxQuant).InferDataType(InferDataTypeForSituMxQuant);
} // namespace ops
