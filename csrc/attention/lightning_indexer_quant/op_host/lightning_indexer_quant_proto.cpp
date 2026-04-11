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
 * \file lightning_indexer_quant_proto.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "error/ops_error.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 2;
constexpr uint32_t ATTR_KV_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 4;

static ge::graphStatus InferShapeLightningIndexerQuant(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OPS_LOG_E("LightningIndexerQuant", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OPS_LOG_E_IF_NULL(context, keyShape, return ge::GRAPH_FAILED);
    gert::Shape *outShape = context->GetOutputShape(0);

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OPS_LOG_E_IF_NULL(context, inputLayoutQueryPtr, return ge::GRAPH_FAILED);
    const char *inputLayoutKeyPtr = attrs->GetAttrPointer<char>(ATTR_KV_LAYOUT_INDEX);
    OPS_LOG_E_IF_NULL(context, inputLayoutKeyPtr, return ge::GRAPH_FAILED);
    const int64_t *sparse_count = attrs->GetInt(ATTR_SPARSE_COUNT_INDEX);
    OPS_LOG_E_IF_NULL(context, sparse_count, return ge::GRAPH_FAILED);

    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    if (inputLayoutQueryPtrStr != "TND" && inputLayoutQueryPtrStr != "BSND") {
        OPS_LOG_E(context, "The input layout query should be TND or BSND, but got %s.", inputLayoutQueryPtrStr.c_str());
        return GRAPH_FAILED;
    }

    outShape->SetDimNum(queryShape->GetDimNum());
    int64_t keyHeadNum = (inputLayoutKeyPtrStr == "TND") ? keyShape->GetDim(1) : keyShape->GetDim(2);
    if (inputLayoutQueryPtrStr == "BSND") {
        outShape->SetDim(0, queryShape->GetDim(0));  // 0:Dim B
        outShape->SetDim(1, queryShape->GetDim(1));  // 1:Dim S
        outShape->SetDim(2, keyHeadNum);             // 2:Dim N
        outShape->SetDim(3, *sparse_count);          // 3:Dim K
    } else {
        outShape->SetDim(0, queryShape->GetDim(0));  // 0:Dim T
        outShape->SetDim(1, keyHeadNum);             // 1:output shape's N Dim, 2: key shape's N Dim
        outShape->SetDim(2, *sparse_count);          // 2:Dim K
    }

    OPS_LOG_D(context->GetNodeName(), "LightningIndexerQuant InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLightningIndexerQuant(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OPS_LOG_E("LightningIndexerQuant", "InferDataTypeContext context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_D(context->GetNodeName(), "Enter LightningIndexerQuant InferDataType impl.");
    // default index data type is int32
    ge::DataType outputType = ge::DT_INT32;
    context->SetOutputDataType(0, outputType);
    OPS_LOG_D(context->GetNodeName(), "LightningIndexerQuant InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LightningIndexerQuant)
    .InferShape(InferShapeLightningIndexerQuant)
    .InferDataType(InferDataTypeLightningIndexerQuant);
}  // namespace ops
