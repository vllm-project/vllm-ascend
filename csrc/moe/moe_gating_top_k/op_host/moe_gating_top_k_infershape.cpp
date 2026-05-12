/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_gating_top_k_infershape.cpp
 * \brief
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "error_log.h"

#include <string>

#include <string>
#define TO_STRING(x) std::string(#x)

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t X_INDEX = 0;
static constexpr int64_t BIAS_INDEX = 1;
static constexpr int64_t Y_INDEX = 0;
static constexpr int64_t EXPERT_IDX_INDEX = 1;
static constexpr int64_t OUT_INDEX = 2;

static ge::graphStatus CheckInputShape(gert::InferShapeContext *context, const gert::Shape *xShape)
{
    int64_t XRows = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(0);
    int64_t expertNum = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(1);
    if (XRows < NEG_ONE || expertNum < NEG_ONE) {
        OP_LOGE(context, "Invalid x shape, shape is %s.", TO_STRING(*xShape).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDimsAndAttr(gert::InferShapeContext *context, const gert::Shape *xShape,
                                             const int64_t k)
{
    if (xShape->GetDimNum() == 1U) {
        if (xShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE(context, "The dynamic dim of x should be -2, current shape is %s.",
                    TO_STRING(*xShape).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (xShape->GetDimNum() != DIM_TWO) {
        OP_LOGE(context, "The dim of x should be 2 or dynamic, current shape is %s.",
                TO_STRING(*xShape).c_str());
        return ge::GRAPH_FAILED;
    }

    if (k < 0) {
        OP_LOGE(context, "k must be a non-negative number.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static void ShowInputShapeInfo(gert::InferShapeContext *context, const gert::Shape *xShape, const int64_t k)
{
    OP_LOGD(context, "x shape is: %s.", TO_STRING(*xShape).c_str());
    OP_LOGD(context, "k is: %ld.", k);
}

static void ShowOutputShapeInfo(gert::InferShapeContext *context, const gert::Shape *yShape,
                                const gert::Shape *expertIdxShape, const gert::Shape *outShape)
{
    OP_LOGD(context, "y shape is: %s after infershape.", TO_STRING(*yShape).c_str());
    OP_LOGD(context, "expert_idx shape is: %s after infershape.", TO_STRING(*expertIdxShape).c_str());
    OP_LOGD(context, "out shape is: %s after infershape.", TO_STRING(*outShape).c_str());
}

static ge::graphStatus InferShape4MoeGatingTopK(gert::InferShapeContext *context)
{
    OP_LOGD(context, "Begin to do MoeGatingTopKInfershape.");

    // 获取输入shape
    const gert::Shape *xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape *yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    gert::Shape *expertIdxShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertIdxShape);
    gert::Shape *outShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    // 获取attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *kPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, kPtr);
    const int64_t k = *kPtr;
    ShowInputShapeInfo(context, xShape, k);

    // 参数校验
    if (CheckInputDimsAndAttr(context, xShape, k) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInputShape(context, xShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t rows = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(0);
    int64_t expertNum = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(1);

    yShape->SetDimNum(DIM_TWO);
    yShape->SetDim(0U, rows);
    yShape->SetDim(1U, k);

    expertIdxShape->SetDimNum(DIM_TWO);
    expertIdxShape->SetDim(0U, rows);
    expertIdxShape->SetDim(1U, k);

    outShape->SetDimNum(DIM_TWO);
    outShape->SetDim(0U, rows);
    outShape->SetDim(1U, expertNum);

    ShowOutputShapeInfo(context, yShape, expertIdxShape, outShape);
    OP_LOGD(context, "End to do MoeGatingTopKInfershape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeGatingTopK(gert::InferDataTypeContext *context)
{
    OP_LOGD(context, "Begin to do MoeGatingTopKInferDataType.");
    auto xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(Y_INDEX, xDtype);
    context->SetOutputDataType(EXPERT_IDX_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUT_INDEX, ge::DT_FLOAT);
    OP_LOGD(context, "End to do MoeGatingTopKInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeGatingTopK).InferShape(InferShape4MoeGatingTopK).InferDataType(InferDataType4MoeGatingTopK);
} // namespace ops
