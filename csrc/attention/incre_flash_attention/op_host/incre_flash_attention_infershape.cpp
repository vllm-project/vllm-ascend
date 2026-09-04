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
 * \file incre_flash_attention_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "log/error_code.h"

using namespace ge;

namespace ops {
static constexpr uint32_t IFA_LAYOUT_DIM0 = 0;
static constexpr uint32_t IFA_QUERY_INDEX = 0;
static constexpr int32_t IFA_UNKNOWN_DIMS = -2;
static constexpr uint32_t IFA_DIM_NUMS_1 = 1;
static constexpr uint32_t IFA_LAYOUT_SH_DIMS = 2;
static constexpr uint32_t IFA_LAYOUT_BSH_DIMS = 3;
static constexpr uint32_t IFA_LAYOUT_NSD_DIMS = 3;
static constexpr uint32_t IFA_LAYOUT_BNSD_DIMS = 4;
static constexpr uint32_t IFA_LAYOUT_BSND_DIMS = 4;
static constexpr uint32_t IFA_ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t IFA_ATTR_INPUT_LAYOUT_INDEX = 2;
static constexpr uint32_t IFA_INPUT_ACTUAL_SEQ_LENGTHS_INDEX = 5;
static constexpr uint32_t IFA_QUANT_SCALE2_INDEX = 9;

static ge::graphStatus InferShapeIncreFlashAttention(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("IncreFlashAttention", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter IncreFlashAttention inferShape impl.");
    // query shape
    const gert::Shape *queryShape = context->GetInputShape(IFA_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    // attentionOut
    gert::Shape *attentionOutShape = context->GetOutputShape(IFA_ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);

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
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char *inputLayoutPtr = attrs->GetAttrPointer<char>(IFA_ATTR_INPUT_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutPtr);

    if (strcmp(inputLayoutPtr, "SH") == 0) {
        if (queryShape->GetDimNum() != IFA_LAYOUT_SH_DIMS) {
            OP_LOGE(context->GetNodeName(), "Layout SH, queryDims(%zu) must be 2!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else if (strcmp(inputLayoutPtr, "BSH") == 0) {
        if (queryShape->GetDimNum() != IFA_LAYOUT_BSH_DIMS) {
            OP_LOGE(context->GetNodeName(), "Layout BSH, queryDims(%zu) must be 3!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else if (strcmp(inputLayoutPtr, "NSD") == 0) {
        if (queryShape->GetDimNum() != IFA_LAYOUT_NSD_DIMS) {
            OP_LOGE(context->GetNodeName(), "Layout NSD, queryDims(%zu) must be 3!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else if (strcmp(inputLayoutPtr, "BNSD") == 0) {
        if (queryShape->GetDimNum() != IFA_LAYOUT_BNSD_DIMS) {
            OP_LOGE(context->GetNodeName(), "Layout BNSD, queryDims(%zu) must be 4!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else if (strcmp(inputLayoutPtr, "BSND") == 0) {
        if (queryShape->GetDimNum() != IFA_LAYOUT_BSND_DIMS) {
            OP_LOGE(context->GetNodeName(), "Layout BSND, queryDims(%zu) must be 4!", queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE(context->GetNodeName(), "Invalid input layout: %s, not support!", inputLayoutPtr);
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "IncreFlashAttention inferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeIncreFlashAttention(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("IncreFlashAttention", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter IncreFlashAttention inferDataType impl.");
    // default set q's dtype as ifa's output dtype
    ge::DataType outputType = context->GetInputDataType(IFA_QUERY_INDEX);
    if (context->GetOptionalInputDataType(IFA_QUANT_SCALE2_INDEX) != ge::DT_UNDEFINED) {
        outputType = ge::DT_INT8;
    } else if (context->GetInputDataType(IFA_QUERY_INDEX) == ge::DT_INT8) {
        outputType = ge::DT_FLOAT16;
    }
    // attention_out, outidx:0
    context->SetOutputDataType(IFA_ATTENTION_OUT_INDEX, outputType);
    OP_LOGD(context->GetNodeName(), "IncreFlashAttention inferDataType end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IncreFlashAttention)
    .InferShape(InferShapeIncreFlashAttention)
    .InferDataType(InferDataTypeIncreFlashAttention)
    .InputsDataDependency({IFA_INPUT_ACTUAL_SEQ_LENGTHS_INDEX});
} // namespace ops