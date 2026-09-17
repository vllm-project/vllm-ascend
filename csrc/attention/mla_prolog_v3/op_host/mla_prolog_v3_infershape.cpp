/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mla_prolog_v3_infershape.h"

using namespace ge;

namespace ops {

ge::graphStatus GetMlaPrologV3ShapeDim(const gert::InferShapeContext *context, MlaPrologProtoShapeParam &shapeParam)
{
    auto tokenXShape = context->GetRequiredInputShape(TOKEN_X_INDEX); // (B, S, He) | (T, He)
    OP_CHECK_NULL_WITH_CONTEXT(context, tokenXShape);
    auto weightUkShape = context->GetRequiredInputShape(WEIGHT_UK_INDEX); // (N, D, Hckv)
    OP_CHECK_NULL_WITH_CONTEXT(context, weightUkShape);
    auto ropeSinShape = context->GetRequiredInputShape(ROPE_SIN_INDEX); // (B, S, Dr) | (T, Dr)
    OP_CHECK_NULL_WITH_CONTEXT(context, ropeSinShape);
    auto weightDqShape = context->GetRequiredInputShape(WEIGHT_DQ_INDEX); // (He, Hcq)
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDqShape);
    auto kvCacheShape = context->GetRequiredInputShape(KV_CACHE_INDEX_V3); // (B, Nkv, Skv, Hckv)
    OP_CHECK_NULL_WITH_CONTEXT(context, kvCacheShape);
    auto krCacheShape = context->GetRequiredInputShape(KR_CACHE_INDEX_V3); // (B, Nkv, Skv, Dr)
    OP_CHECK_NULL_WITH_CONTEXT(context, krCacheShape);

    OP_CHECK_IF(((tokenXShape->GetDimNum() != DIM_NUM_2) && (tokenXShape->GetDimNum() != DIM_NUM_3)),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "tokenX",
                                             std::to_string(tokenXShape->GetDimNum()) + "D", "2D or 3D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((weightUkShape->GetDimNum() != DIM_NUM_3),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "weightUk",
                                             std::to_string(weightUkShape->GetDimNum()) + "D", "3D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((ropeSinShape->GetDimNum() != DIM_NUM_2) && (ropeSinShape->GetDimNum() != DIM_NUM_3)),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "ropeSin",
                                             std::to_string(ropeSinShape->GetDimNum()) + "D", "2D or 3D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((weightDqShape->GetDimNum() != DIM_NUM_2),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "weightDq",
                                             std::to_string(weightDqShape->GetDimNum()) + "D", "2D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((kvCacheShape->GetDimNum() != DIM_NUM_3) && (kvCacheShape->GetDimNum() != DIM_NUM_4)),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "kvCache",
                                             std::to_string(kvCacheShape->GetDimNum()) + "D", "3D or 4D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((krCacheShape->GetDimNum() != DIM_NUM_1) && (krCacheShape->GetDimNum() != DIM_NUM_3) &&
                 (krCacheShape->GetDimNum() != DIM_NUM_4)),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "krCache",
                                             std::to_string(krCacheShape->GetDimNum()) + "D", "1D or 3D or 4D"),
                return ge::GRAPH_FAILED);

    if (tokenXShape->GetDimNum() == DIM_NUM_3) { // BS
        shapeParam.isBsMerge = false;
        shapeParam.B = tokenXShape->GetDim(DIM_INDEX_0);
        shapeParam.S = tokenXShape->GetDim(DIM_INDEX_1);
        shapeParam.Dr = ropeSinShape->GetDim(DIM_INDEX_2);
        shapeParam.T = shapeParam.B * shapeParam.S;
    } else { // T
        shapeParam.isBsMerge = true;
        shapeParam.T = tokenXShape->GetDim(DIM_INDEX_0);
        shapeParam.Dr = ropeSinShape->GetDim(DIM_INDEX_1);
    }

    shapeParam.N = weightUkShape->GetDim(DIM_INDEX_0);
    shapeParam.Hckv = weightUkShape->GetDim(DIM_INDEX_2);

    shapeParam.Hcq = weightDqShape->GetDim(DIM_INDEX_1);
    return GRAPH_SUCCESS;
}

static bool IsWeightFullQuantFamily(int64_t weightQuantMode)
{
    return weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT ||
           weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT_FP8 ||
           weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT_HIF8 ||
           weightQuantMode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT;
}

static void SetQueryAndRopeShape(const MlaPrologProtoShapeParam &shapeParam, gert::Shape *queryShape,
                                 gert::Shape *queryRopeShape)
{
    if (!shapeParam.isBsMerge) {
        queryShape->SetDimNum(DIM_NUM_4); // (B, S, N, Hckv)
        queryShape->SetDim(DIM_INDEX_0, shapeParam.B);
        queryShape->SetDim(DIM_INDEX_1, shapeParam.S);
        queryShape->SetDim(DIM_INDEX_2, shapeParam.N);
        queryShape->SetDim(DIM_INDEX_3, shapeParam.Hckv);

        queryRopeShape->SetDimNum(DIM_NUM_4); // (B, S, N, Dr)
        queryRopeShape->SetDim(DIM_INDEX_0, shapeParam.B);
        queryRopeShape->SetDim(DIM_INDEX_1, shapeParam.S);
        queryRopeShape->SetDim(DIM_INDEX_2, shapeParam.N);
        queryRopeShape->SetDim(DIM_INDEX_3, shapeParam.Dr);
    } else {
        queryShape->SetDimNum(DIM_NUM_3); // (T, N, Hckv)
        queryShape->SetDim(DIM_INDEX_0, shapeParam.T);
        queryShape->SetDim(DIM_INDEX_1, shapeParam.N);
        queryShape->SetDim(DIM_INDEX_2, shapeParam.Hckv);

        queryRopeShape->SetDimNum(DIM_NUM_3); // (T, N, Dr)
        queryRopeShape->SetDim(DIM_INDEX_0, shapeParam.T);
        queryRopeShape->SetDim(DIM_INDEX_1, shapeParam.N);
        queryRopeShape->SetDim(DIM_INDEX_2, shapeParam.Dr);
    }
}

static void SetDequantScaleQNopeShape(const MlaPrologProtoShapeParam &shapeParam, gert::Shape *dequantScaleQNopeShape,
                                      int64_t weightQuantMode, int64_t kvQuantMode)
{
    if (kvQuantMode == KV_QUANT_MODE_PER_TENSOR && IsWeightFullQuantFamily(weightQuantMode)) {
        dequantScaleQNopeShape->SetDimNum(DIM_NUM_3); // (B*S, N, 1) | (T, N, 1)
        dequantScaleQNopeShape->SetDim(DIM_INDEX_0, shapeParam.isBsMerge ? shapeParam.T : shapeParam.B * shapeParam.S);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_1, shapeParam.N);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_2, DIM_NUM_1); // 1: Fix dim 1
    } else {
        dequantScaleQNopeShape->SetDimNum(DIM_NUM_1);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_0, DIM_NUM_0);
    }
}

static ge::graphStatus SetQueryNormShape(const MlaPrologProtoShapeParam &shapeParam, gert::InferShapeContext *context,
                                         gert::Shape *queryNormShape, gert::Shape *dequantScaleQNormShape,
                                         int64_t weightQuantMode, bool queryNormFlag)
{
    if (!queryNormFlag) {
        queryNormShape->SetDimNum(DIM_NUM_1);
        queryNormShape->SetDim(DIM_INDEX_0, DIM_NUM_0);
        dequantScaleQNormShape->SetDimNum(DIM_NUM_1);
        dequantScaleQNormShape->SetDim(DIM_INDEX_0, DIM_NUM_0);
        return GRAPH_SUCCESS;
    }

    auto weightUqQrDesc = context->GetInputDesc(WEIGHT_UQ_QR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightUqQrDesc);
    (void)weightUqQrDesc; // 保持原有校验语义，desc 本身未被使用

    int64_t tSize = shapeParam.isBsMerge ? shapeParam.T : shapeParam.B * shapeParam.S;
    if (shapeParam.isBsMerge) {
        // [T, Hcq]
        queryNormShape->SetDimNum(DIM_NUM_2);
        queryNormShape->SetDim(DIM_INDEX_0, shapeParam.T);
        queryNormShape->SetDim(DIM_INDEX_1, shapeParam.Hcq);
    } else {
        // [B, S, Hcq]
        queryNormShape->SetDimNum(DIM_NUM_3);
        queryNormShape->SetDim(DIM_INDEX_0, shapeParam.B);
        queryNormShape->SetDim(DIM_INDEX_1, shapeParam.S);
        queryNormShape->SetDim(DIM_INDEX_2, shapeParam.Hcq);
    }

    if (weightQuantMode == WEIGHT_QUANT_MODE_NO_QUANT) {
        dequantScaleQNormShape->SetDimNum(DIM_NUM_1);
        dequantScaleQNormShape->SetDim(DIM_INDEX_0, DIM_NUM_0);
    } else if (weightQuantMode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT) {
        dequantScaleQNormShape->SetDimNum(DIM_NUM_2);
        dequantScaleQNormShape->SetDim(DIM_INDEX_0, tSize);
        dequantScaleQNormShape->SetDim(DIM_INDEX_1, shapeParam.Hcq / FP8_E4M3_BLOCK_SIZE);
    } else {
        dequantScaleQNormShape->SetDimNum(DIM_NUM_2);
        dequantScaleQNormShape->SetDim(DIM_INDEX_0, tSize);
        dequantScaleQNormShape->SetDim(DIM_INDEX_1, DIM_NUM_1);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus SetMlaPrologV3ShapeDim(const MlaPrologProtoShapeParam &shapeParam, gert::InferShapeContext *context)
{
    auto queryShape = context->GetOutputShape(QUERY_INDEX); // query: (B, S, N, Hckv) | (T, N, Hckv)
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    auto queryRopeShape = context->GetOutputShape(QUERY_ROPE_INDEX); // queryRope: (B, S, N, Dr) | (T, N, Dr)
    OP_CHECK_NULL_WITH_CONTEXT(context, queryRopeShape);
    auto kvCacheOutShape = context->GetOutputShape(KV_CACHE_OUT_INDEX); // kvCacheOut: (B, Nkv, Skv, Hckv)
    OP_CHECK_NULL_WITH_CONTEXT(context, kvCacheOutShape);
    auto krCacheOutShape = context->GetOutputShape(KR_CACHE_OUT_INDEX); // krCacheOut: (B, Nkv, Skv, Dr)
    OP_CHECK_NULL_WITH_CONTEXT(context, krCacheOutShape);

    // set output shape
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // Get attribute pointers and dereference once
    const int64_t *weightQuantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_WEIGHT_QUANT_MODE_FLAG_INDEX);
    const int64_t weightQuantMode = (weightQuantModePtr == nullptr) ? 0 : *weightQuantModePtr;
    const int64_t *kvQuantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_KV_QUANT_MODE_FLAG_INDEX);
    const int64_t kvQuantMode = (kvQuantModePtr == nullptr) ? 0 : *kvQuantModePtr;
    const bool *queryNormFlagPtr = attrs->GetAttrPointer<bool>(ATTR_QUERY_NORM_FLAG_INDEX);
    const bool queryNormFlag = (queryNormFlagPtr == nullptr) ? 0 : *queryNormFlagPtr;

    // dequantScaleQNope: (B*S, N ,1) | (T, N, 1). (1) if not enabled
    auto dequantScaleQNopeShape = context->GetOutputShape(DEQUANT_SCALE_Q_NOPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dequantScaleQNopeShape);

    // queryNorm
    gert::Shape *queryNormShape = context->GetOutputShape(QUERY_NORM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryNormShape);
    gert::Shape *dequantScaleQNormShape = context->GetOutputShape(DEQUANT_SCALE_Q_NORM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dequantScaleQNormShape);

    SetQueryAndRopeShape(shapeParam, queryShape, queryRopeShape);
    *kvCacheOutShape = *context->GetRequiredInputShape(KV_CACHE_INDEX_V3);
    *krCacheOutShape = *context->GetRequiredInputShape(KR_CACHE_INDEX_V3);
    SetDequantScaleQNopeShape(shapeParam, dequantScaleQNopeShape, weightQuantMode, kvQuantMode);
    return SetQueryNormShape(shapeParam, context, queryNormShape, dequantScaleQNormShape, weightQuantMode,
                             queryNormFlag);
}

ge::graphStatus InferShapeMlaPrologV3(gert::InferShapeContext *context)
{
    OP_LOGI(context->GetNodeName(), "Enter MlaPrologV3 infershape impl.");

    MlaPrologProtoShapeParam shapeParam{};
    auto apiRet = GetMlaPrologV3ShapeDim(context, shapeParam);
    if (apiRet != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    apiRet = SetMlaPrologV3ShapeDim(shapeParam, context);
    if (apiRet != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeMlaPrologV3(gert::InferDataTypeContext *context)
{
    OP_LOGI(context->GetNodeName(), "Enter MlaPrologV3 inferDataType impl.");

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // Get attribute pointers and dereference once
    const int64_t *weightQuantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_WEIGHT_QUANT_MODE_FLAG_INDEX);
    const int weightQuantMode = (weightQuantModePtr == nullptr) ? 0 : *weightQuantModePtr;
    const int64_t *kvQuantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_KV_QUANT_MODE_FLAG_INDEX);
    const int kvQuantMode = (kvQuantModePtr == nullptr) ? 0 : *kvQuantModePtr;

    // mxfp8 quant
    if (weightQuantMode == WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT) {
        bool isMxfp8FullQuant = (context->GetRequiredInputDataType(TOKEN_X_INDEX) == ge::DT_FLOAT8_E4M3FN &&
                                 context->GetOptionalInputDataType(QUANT_SCALE_CKV_INDEX) != ge::DT_UNDEFINED);

        context->SetOutputDataType(QUERY_INDEX, (isMxfp8FullQuant) ?
                                                    context->GetRequiredInputDataType(WEIGHT_DKV_KR_INDEX) :
                                                    context->GetRequiredInputDataType(WEIGHT_UK_INDEX));
        context->SetOutputDataType(QUERY_ROPE_INDEX, context->GetRequiredInputDataType(WEIGHT_UK_INDEX));
        context->SetOutputDataType(KV_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KV_CACHE_INDEX_V3));
        context->SetOutputDataType(KR_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KR_CACHE_INDEX_V3));
        context->SetOutputDataType(DEQUANT_SCALE_Q_NOPE_INDEX, ge::DT_FLOAT);
        context->SetOutputDataType(QUERY_NORM_INDEX, context->GetRequiredInputDataType(WEIGHT_UQ_QR_INDEX));
        context->SetOutputDataType(DEQUANT_SCALE_Q_NORM_INDEX, ge::DT_FLOAT8_E8M0);
    } else {
        context->SetOutputDataType(QUERY_INDEX, context->GetRequiredInputDataType(WEIGHT_UK_INDEX));
        context->SetOutputDataType(QUERY_ROPE_INDEX, context->GetRequiredInputDataType(WEIGHT_UK_INDEX));
        context->SetOutputDataType(KV_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KV_CACHE_INDEX_V3));
        context->SetOutputDataType(KR_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KR_CACHE_INDEX_V3));

        // full quant
        bool isQuantQuery =
            ((weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT || weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT_FP8 ||
              weightQuantMode == WEIGHT_QUANT_MODE_FULL_QUANT_HIF8) &&
             kvQuantMode == KV_QUANT_MODE_PER_TENSOR);

        context->SetOutputDataType(QUERY_INDEX,
                                   isQuantQuery ? context->GetRequiredInputDataType(TOKEN_X_INDEX) : ge::DT_BF16);
        context->SetOutputDataType(DEQUANT_SCALE_Q_NOPE_INDEX, ge::DT_FLOAT);

        if (weightQuantMode == WEIGHT_QUANT_MODE_NO_QUANT) {
            context->SetOutputDataType(QUERY_NORM_INDEX, ge::DT_BF16);
        } else {
            context->SetOutputDataType(QUERY_NORM_INDEX, context->GetRequiredInputDataType(WEIGHT_UQ_QR_INDEX));
        }
        context->SetOutputDataType(DEQUANT_SCALE_Q_NORM_INDEX, ge::DT_FLOAT);
    }

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MlaPrologV3).InferShape(InferShapeMlaPrologV3).InferDataType(InferDataTypeMlaPrologV3);
} // namespace ops