/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"


using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 4;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 0;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 2;

static ge::graphStatus InferShapeLightningIndexer(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("LightningIndexer", "InferShapeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OPS_LOG_E_IF_NULL(context, keyShape, return ge::GRAPH_FAILED);
    gert::Shape *outShape = context->GetOutputShape(0);

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OPS_LOG_E_IF_NULL(context, inputLayoutQueryPtr, return ge::GRAPH_FAILED);
    const char *inputLayoutKeyPtr = attrs->GetAttrPointer<char>(ATTR_KEY_LAYOUT_INDEX);
    OPS_LOG_E_IF_NULL(context, inputLayoutKeyPtr, return ge::GRAPH_FAILED);
    const int64_t *seleced_count = attrs->GetInt(ATTR_SPARSE_COUNT_INDEX);
    OPS_LOG_E_IF_NULL(context, seleced_count, return ge::GRAPH_FAILED);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    OPS_ERR_IF(
        inputLayoutQueryPtrStr != "TND" && inputLayoutQueryPtrStr != "BSND",
        OPS_LOG_E(context, "The attr layout_query should be TND or BSND, but got %s.", inputLayoutQueryPtrStr.c_str()),
        return ge::GRAPH_FAILED);

    outShape->SetDimNum(queryShape->GetDimNum());
    if (inputLayoutQueryPtrStr == "BSND") {
        OPS_ERR_IF(
            queryShape->GetDimNum() != 4,
            OPS_LOG_E(context, "Layout BSND, queryDims (%zu) must be 4!", queryShape->GetDimNum()),
            return ge::GRAPH_FAILED);
        outShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim B
        outShape->SetDim(1, queryShape->GetDim(1)); // 1:Dim S
        outShape->SetDim(2, keyShape->GetDim(2));   // 2:Dim N
        outShape->SetDim(3, *seleced_count);        // 3:Dim K
    } else {
        OPS_ERR_IF(
            queryShape->GetDimNum() != 3,
            OPS_LOG_E(context, "Layout TND, queryDims (%zu) must be 3!", queryShape->GetDimNum()),
            return ge::GRAPH_FAILED);
        outShape->SetDim(0, queryShape->GetDim(0));                      // 0:Dim T
        int32_t nDimIndex = (inputLayoutKeyPtrStr == "PA_BSND") ? 2 : 1; // 2:Key Dim N
        outShape->SetDim(1, keyShape->GetDim(nDimIndex));                // 1:Dim N
        outShape->SetDim(2, *seleced_count);                             // 2:Dim K
    }
    OPS_LOG_D(context->GetNodeName(), "LightningIndexer InferShape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLightningIndexer(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("LightningIndexer", "InferDataTypeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    OPS_LOG_D(context->GetNodeName(), "Enter LightningIndexer InferDataType impl.");
    // default set q's dtype as fia's output type
    ge::DataType outputType = ge::DT_INT32;
    // attention_out, outidx:0
    context->SetOutputDataType(0, outputType);
    OPS_LOG_D(context->GetNodeName(), "LightningIndexer InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LightningIndexer)
    .InferShape(InferShapeLightningIndexer)
    .InferDataType(InferDataTypeLightningIndexer);
} // namespace ops
