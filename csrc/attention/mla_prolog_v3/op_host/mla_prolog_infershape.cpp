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
 * \file mla_prolog_proto.cpp
 * \brief
 */

#include "mla_prolog_infershape.h"

using namespace ge;

namespace ops {

ge::graphStatus GetMlaPrologShapeDim(const gert::InferShapeContext *context, MlaPrologProtoShapeParam &shapeParam)
{
    auto tokenXShape = context->GetRequiredInputShape(TOKEN_X_INDEX);      // (B, S, He) | (T, He)
    OP_CHECK_NULL_WITH_CONTEXT(context, tokenXShape);
    auto weightUkShape = context->GetRequiredInputShape(WEIGHT_UK_INDEX);  // (N, D, Hckv)
    OP_CHECK_NULL_WITH_CONTEXT(context, weightUkShape);
    auto ropeSinShape = context->GetRequiredInputShape(ROPE_SIN_INDEX);    // (B, S, Dr) | (T, Dr)
    OP_CHECK_NULL_WITH_CONTEXT(context, ropeSinShape);
    if (std::strcmp(context->GetNodeType(), "MlaPrologV3") == 0) {
        auto kvCacheShape = context->GetRequiredInputShape(KV_CACHE_INDEX_V3);    // (B, Nkv, Skv, Hckv)
        OP_CHECK_NULL_WITH_CONTEXT(context, kvCacheShape);
        auto krCacheShape = context->GetRequiredInputShape(KR_CACHE_INDEX_V3);    // (B, Nkv, Skv, Dr)
        OP_CHECK_NULL_WITH_CONTEXT(context, krCacheShape);
    } else {
        auto kvCacheShape = context->GetRequiredInputShape(KV_CACHE_INDEX);    // (B, Nkv, Skv, Hckv)
        OP_CHECK_NULL_WITH_CONTEXT(context, kvCacheShape);
        auto krCacheShape = context->GetRequiredInputShape(KR_CACHE_INDEX);    // (B, Nkv, Skv, Dr)
        OP_CHECK_NULL_WITH_CONTEXT(context, krCacheShape);
    }

    OP_CHECK_IF(((tokenXShape->GetDimNum() != DIM_NUM_3) && (tokenXShape->GetDimNum() != DIM_NUM_2)),
        OP_LOGE(context->GetNodeName(), "tokenXShape is not 2 or 3, but %zu", tokenXShape->GetDimNum()), return ge::GRAPH_FAILED);

    if (tokenXShape->GetDimNum() == DIM_NUM_3) {                // BS
        shapeParam.isBsMerge = false;
        shapeParam.B = tokenXShape->GetDim(DIM_INDEX_0);
        shapeParam.S = tokenXShape->GetDim(DIM_INDEX_1);
        shapeParam.Dr = ropeSinShape->GetDim(DIM_INDEX_2);
        shapeParam.T = shapeParam.B * shapeParam.S;
    } else {                                                    // T
        shapeParam.isBsMerge = true;
        shapeParam.T = tokenXShape->GetDim(DIM_INDEX_0);
        shapeParam.Dr = ropeSinShape->GetDim(DIM_INDEX_1);
    }

    shapeParam.N = weightUkShape->GetDim(DIM_INDEX_0);
    shapeParam.Hckv = weightUkShape->GetDim(DIM_INDEX_2);
    return GRAPH_SUCCESS;
}

ge::graphStatus SetMlaPrologShapeDim(const MlaPrologProtoShapeParam &shapeParam, gert::InferShapeContext *context)
{
    auto queryShape = context->GetOutputShape(QUERY_INDEX);                 // query: (B, S, N, Hckv) | (T, N, Hckv)
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    auto queryRopeShape = context->GetOutputShape(QUERY_ROPE_INDEX);        // queryRope: (B, S, N, Dr) | (T, N, Dr)
    OP_CHECK_NULL_WITH_CONTEXT(context, queryRopeShape);
    auto kvCacheOutShape = context->GetOutputShape(KV_CACHE_OUT_INDEX);     // kvCacheOut: (B, Nkv, Skv, Hckv)
    OP_CHECK_NULL_WITH_CONTEXT(context, kvCacheOutShape);
    auto krCacheOutShape = context->GetOutputShape(KR_CACHE_OUT_INDEX);     // krCacheOut: (B, Nkv, Skv, Dr)
    OP_CHECK_NULL_WITH_CONTEXT(context, krCacheOutShape);

    // Set output shape
    if (!shapeParam.isBsMerge) {
        queryShape->SetDimNum(DIM_NUM_4);                   // (B, S, N, Hckv)
        queryShape->SetDim(DIM_INDEX_0, shapeParam.B);
        queryShape->SetDim(DIM_INDEX_1, shapeParam.S);
        queryShape->SetDim(DIM_INDEX_2, shapeParam.N);
        queryShape->SetDim(DIM_INDEX_3, shapeParam.Hckv);

        queryRopeShape->SetDimNum(DIM_NUM_4);               // (B, S, N, Dr)
        queryRopeShape->SetDim(DIM_INDEX_0, shapeParam.B);
        queryRopeShape->SetDim(DIM_INDEX_1, shapeParam.S);
        queryRopeShape->SetDim(DIM_INDEX_2, shapeParam.N);
        queryRopeShape->SetDim(DIM_INDEX_3, shapeParam.Dr);
    } else {
        queryShape->SetDimNum(DIM_NUM_3);                   // (T, N, Hckv)
        queryShape->SetDim(DIM_INDEX_0, shapeParam.T);
        queryShape->SetDim(DIM_INDEX_1, shapeParam.N);
        queryShape->SetDim(DIM_INDEX_2, shapeParam.Hckv);

        queryRopeShape->SetDimNum(DIM_NUM_3);               // (T, N, Dr)
        queryRopeShape->SetDim(DIM_INDEX_0, shapeParam.T);
        queryRopeShape->SetDim(DIM_INDEX_1, shapeParam.N);
        queryRopeShape->SetDim(DIM_INDEX_2, shapeParam.Dr);
    }

    if (std::strcmp(context->GetNodeType(), "MlaPrologV3") == 0) {
        *kvCacheOutShape = *context->GetRequiredInputShape(KV_CACHE_INDEX_V3);
        *krCacheOutShape = *context->GetRequiredInputShape(KR_CACHE_INDEX_V3);
    } else {
        *kvCacheOutShape = *context->GetRequiredInputShape(KV_CACHE_INDEX);
        *krCacheOutShape = *context->GetRequiredInputShape(KR_CACHE_INDEX);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeMlaProlog(gert::InferShapeContext *context) {
    OP_LOGI(context->GetNodeName(), "Enter MlaProlog infershape impl.");

    MlaPrologProtoShapeParam shapeParam {};
    auto apiRet = GetMlaPrologShapeDim(context, shapeParam);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context get input shape failed"), return ge::GRAPH_FAILED);

    apiRet = SetMlaPrologShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context set output shape failed"), return ge::GRAPH_FAILED);

    OP_LOGI(context->GetNodeName(), "MlaProlog infershape end.");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeMlaProlog(gert::InferDataTypeContext *context) {
    OP_LOGI(context->GetNodeName(), "Enter MlaProlog inferdatatype impl.");

    context->SetOutputDataType(QUERY_INDEX, context->GetRequiredInputDataType(WEIGHT_UK_INDEX));
    context->SetOutputDataType(QUERY_ROPE_INDEX, context->GetRequiredInputDataType(WEIGHT_UK_INDEX));
    if (std::strcmp(context->GetNodeType(), "MlaPrologV3") == 0) {
        context->SetOutputDataType(KV_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KV_CACHE_INDEX_V3));
        context->SetOutputDataType(KR_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KR_CACHE_INDEX_V3));
    } else {
        context->SetOutputDataType(KV_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KV_CACHE_INDEX));
        context->SetOutputDataType(KR_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KR_CACHE_INDEX));
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MlaProlog).InferShape(InferShapeMlaProlog).InferDataType(InferDataTypeMlaProlog);
}  // namespace ops