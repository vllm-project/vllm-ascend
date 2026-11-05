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
 * \file swiglu_group_quant_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "error/ops_error.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_X = 0;
constexpr size_t OUTPUT_IDX_Y = 0;
constexpr size_t OUTPUT_IDX_SCALE = 1;
constexpr int64_t NUM_TWO = 2;
constexpr size_t ATTR_INDEX_DST_TYPE = 0;
constexpr size_t ATTR_INDEX_QUANT_MODE = 1;
constexpr size_t ATTR_INDEX_UE8M0_SCALE = 4;
constexpr size_t ATTR_INDEX_OUTPUT_ORIGIN = 6;
constexpr int64_t ACTIVATE_DIM = -1;
constexpr int64_t PER_BLOCK_FP16 = 128;
constexpr int64_t PER_MX_FP16 = 32;
constexpr int64_t MX_QUANT_MODE = 2;
constexpr int64_t FP8_QUANT_MODE = 3;
constexpr int64_t MX_SCALE_ALIGN_FACTOR = 2;

graphStatus InferShape4SwigluGroupQuant(gert::InferShapeContext* context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferShape4SwigluGroupQuant.");

    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OPS_LOG_E_IF_NULL(context, xShape, ge::GRAPH_FAILED);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OPS_LOG_E_IF_NULL(context, yShape, ge::GRAPH_FAILED);
    gert::Shape* scaleShape = context->GetOutputShape(OUTPUT_IDX_SCALE);
    OPS_LOG_E_IF_NULL(context, scaleShape, ge::GRAPH_FAILED);

    int64_t xDim = xShape->GetDimNum();
    int64_t splitDim = static_cast<int64_t>(xDim) - 1;

    if (xShape->GetDim(splitDim) == -1) {
        return ge::GRAPH_SUCCESS;
    }

    if (xShape->GetDim(splitDim) < 0 || xShape->GetDim(splitDim) % NUM_TWO != 0) {
        OPS_LOG_E(context->GetNodeName(), "InferShape4SwigluGroupQuant Split Dim Invalid");
        return GRAPH_FAILED;
    }

    // infer yShape
    *yShape = *xShape;
    yShape->SetDim(splitDim, xShape->GetDim(splitDim) / NUM_TWO);

    auto attrsPtr = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrsPtr, ge::GRAPH_FAILED);
    auto quantModeAttr = attrsPtr->GetAttrPointer<int>(ATTR_INDEX_QUANT_MODE);
    bool isMxQuant = (quantModeAttr != nullptr && (*quantModeAttr == MX_QUANT_MODE)) ? true : false;
    bool isFp8Quant = (quantModeAttr != nullptr && (*quantModeAttr == FP8_QUANT_MODE)) ? true : false;

    // 设置Scale的shape
    scaleShape->SetDimNum(1);
    for (int i = 0; i < xDim - 1; i++) {
        scaleShape->AppendDim(xShape->GetDim(i));
    }
    if (isMxQuant) {
        int64_t tailDim = (xShape->GetDim(splitDim) / 2 + PER_MX_FP16 - 1) / PER_MX_FP16;
        // 额外地，mxFp8需要将最后一维reshape为(-1, 2)
        tailDim = (tailDim + MX_SCALE_ALIGN_FACTOR - 1) / MX_SCALE_ALIGN_FACTOR;
        scaleShape->AppendDim(tailDim);
        scaleShape->AppendDim(MX_SCALE_ALIGN_FACTOR);
    } else {
        int64_t tailDim = (xShape->GetDim(splitDim) / 2 + PER_BLOCK_FP16 - 1) / PER_BLOCK_FP16;
        scaleShape->AppendDim(tailDim);
    }
    OPS_LOG_D(context->GetNodeName(), "End to do InferShape4SwigluGroupQuant");
    return ge::GRAPH_SUCCESS;
}

graphStatus InferDtype4SwigluGroupQuant(gert::InferDataTypeContext* context)
{
    OPS_LOG_D(context->GetNodeName(), "InferDtype4SwigluGroupQuant enter");

    auto dstTypePtr = context->GetAttrs()->GetInt(ATTR_INDEX_DST_TYPE);
    ge::DataType dstType = static_cast<ge::DataType>(*dstTypePtr);

    context->SetOutputDataType(ATTR_INDEX_DST_TYPE, dstType);

    auto attrsPtr = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrsPtr, ge::GRAPH_FAILED);
    auto quantModeAttr = attrsPtr->GetAttrPointer<int>(ATTR_INDEX_QUANT_MODE);
    auto ue8m0ScalePtr = attrsPtr->GetAttrPointer<bool>(ATTR_INDEX_UE8M0_SCALE);
    bool isMxQuant = (quantModeAttr != nullptr && (*quantModeAttr == MX_QUANT_MODE)) ? true : false;
    bool isFp8Quant = (quantModeAttr != nullptr && (*quantModeAttr == FP8_QUANT_MODE)) ? true : false;
    if ((isFp8Quant && *ue8m0ScalePtr) || isMxQuant) {
        context->SetOutputDataType(OUTPUT_IDX_SCALE, DT_FLOAT8_E8M0);
    } else {
        context->SetOutputDataType(OUTPUT_IDX_SCALE, DT_FLOAT);
    }

    OPS_LOG_D(context->GetNodeName(), "InferDtype4SwigluGroupQuant end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwigluGroupQuant)
    .InferShape(InferShape4SwigluGroupQuant)
    .InferDataType(InferDtype4SwigluGroupQuant);
}  // namespace ops
