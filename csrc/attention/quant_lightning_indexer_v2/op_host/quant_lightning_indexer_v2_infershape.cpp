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
 * \file quant_lightning_indexer_v2_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "err/ops_err.h"
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 0;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_KV_LAYOUT_INDEX = 4;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 7;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;

static ge::graphStatus InferShapeQuantLightningIndexerV2(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("QuantLightningIndexerV2", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    gert::Shape *sparseIndicesShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseIndicesShape);
    gert::Shape *sparseValuesShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseValuesShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutQueryPtr);
    const char *inputLayoutKeyPtr = attrs->GetAttrPointer<char>(ATTR_KV_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutKeyPtr);
    const int64_t *sparse_count = attrs->GetInt(ATTR_SPARSE_COUNT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparse_count);

    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    if (inputLayoutQueryPtrStr != "TND" && inputLayoutQueryPtrStr != "BSND") {
        OP_LOGE_FOR_INVALID_VALUE("QuantLightningIndexerV2", "layout_q", inputLayoutQueryPtrStr.c_str(), "BSND or TND");
        return GRAPH_FAILED;
    }

    int64_t keyHeadNum = (inputLayoutKeyPtrStr == "TND") ? keyShape->GetDim(1) : keyShape->GetDim(2);
    if (inputLayoutQueryPtrStr == "BSND") {
        sparseIndicesShape->SetDimNum(DIM_NUM_4);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim B
        sparseIndicesShape->SetDim(1, queryShape->GetDim(1)); // 1:Dim S
        sparseIndicesShape->SetDim(2, keyHeadNum);            // 2:Dim N
        sparseIndicesShape->SetDim(3, *sparse_count);         // 3:Dim K
    } else {
        sparseIndicesShape->SetDimNum(DIM_NUM_3);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim T
        sparseIndicesShape->SetDim(1, keyHeadNum);            // 1:output shape's N Dim, 2: key shape's N Dim
        sparseIndicesShape->SetDim(2, *sparse_count);         // 2:Dim K
    }
    const int32_t *return_value = attrs->GetAttrPointer<int32_t>(ATTR_RETURN_VALUE_INDEX);
    bool returnValueFlag = (return_value != nullptr) ? (*return_value != 0) : false;
    if (returnValueFlag) {
        *sparseValuesShape = *sparseIndicesShape;
    } else {
        sparseValuesShape->SetDimNum(1);
        sparseValuesShape->SetDim(0, 0);
    }

    OP_LOGD(context->GetNodeName(), "QuantLightningIndexerV2 InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeQuantLightningIndexerV2(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("QuantLightningIndexerV2", "InferDataTypeContext context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter QuantLightningIndexerV2 InferDataType impl.");
    // default index data type is int32
    ge::DataType outputType = ge::DT_INT32;
    context->SetOutputDataType(0, outputType);
    OP_LOGD(context->GetNodeName(), "QuantLightningIndexerV2 InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantLightningIndexerV2)
    .InferShape(InferShapeQuantLightningIndexerV2)
    .InferDataType(InferDataTypeQuantLightningIndexerV2);
} // namespace ops
