/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file custom_fused_infer_attention_v310_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;

namespace ops {
static constexpr uint32_t IFA_LAYOUT_DIM0 = 0;
static constexpr uint32_t IFA_QUERY_INDEX = 0;
static constexpr int32_t IFA_UNKNOWN_DIMS = -2;
static constexpr uint32_t IFA_DIM_NUMS_1 = 1;
static constexpr uint32_t IFA_LAYOUT_BSND_DIMS = 4;
static constexpr uint32_t IFA_LAYOUT_TND_DIMS = 3;
static constexpr uint32_t IFA_ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t IFA_ATTR_INPUT_LAYOUT_INDEX = 2;
static constexpr uint32_t IFA_INPUT_ACTUAL_SEQ_LENGTHS_Q_INDEX = 4;
static constexpr uint32_t IFA_INPUT_ACTUAL_SEQ_LENGTHS_KV_INDEX = 5;

static ge::graphStatus InferShapeIncreFlashAttention(gert::InferShapeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Enter CustomFusedInferAttentionV310 inferShape impl.");
    // query shape
    const gert::Shape *queryShape = context->GetInputShape(IFA_QUERY_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)

    // attentionOut
    gert::Shape *attentionOutShape = context->GetOutputShape(IFA_ATTENTION_OUT_INDEX);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)

    // Set output shape
    *attentionOutShape = *queryShape;

    // UNKNOWN DIM
    if ((queryShape->GetDimNum() == IFA_DIM_NUMS_1) && (queryShape->GetDim(IFA_LAYOUT_DIM0) == IFA_UNKNOWN_DIMS)) {
        attentionOutShape->SetDimNum(IFA_DIM_NUMS_1);
        (*attentionOutShape)[IFA_LAYOUT_DIM0] = IFA_UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    // Get attr
    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED)

    const char *inputLayoutPtr = attrs->GetAttrPointer<char>(IFA_ATTR_INPUT_LAYOUT_INDEX);
    OPS_LOG_E_IF_NULL(context, inputLayoutPtr, return ge::GRAPH_FAILED)

    if (strcmp(inputLayoutPtr, "BSND") == 0) {
        if (queryShape->GetDimNum() != IFA_LAYOUT_BSND_DIMS) {
            OPS_LOG_E(context->GetNodeName(), "Layout BSND, queryDims(%zu) must be 4!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else if (strcmp(inputLayoutPtr, "TND") == 0) {
        if (queryShape->GetDimNum() != IFA_LAYOUT_TND_DIMS) {
            OPS_LOG_E(context->GetNodeName(), "Layout TND, queryDims(%zu) must be 3!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else {
        OPS_LOG_E(context->GetNodeName(), "Invalid input layout: %s, only BSND and TND are supported", inputLayoutPtr);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context->GetNodeName(), "CustomFusedInferAttentionV310 inferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeIncreFlashAttention(gert::InferDataTypeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Enter CustomFusedInferAttentionV310 inferDataType impl.");
    context->SetOutputDataType(IFA_ATTENTION_OUT_INDEX, ge::DT_FLOAT16);
    OPS_LOG_D(context->GetNodeName(), "CustomFusedInferAttentionV310 inferDataType end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CustomFusedInferAttentionV310)
    .InferShape(InferShapeIncreFlashAttention)
    .InferDataType(InferDataTypeIncreFlashAttention)
    .InputsDataDependency({IFA_INPUT_ACTUAL_SEQ_LENGTHS_Q_INDEX})
    .InputsDataDependency({IFA_INPUT_ACTUAL_SEQ_LENGTHS_KV_INDEX});
} // namespace ops