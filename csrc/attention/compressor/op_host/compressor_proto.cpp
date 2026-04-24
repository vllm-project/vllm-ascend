/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;

namespace ops {
    // INPUT
    constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
    constexpr uint32_t WEIGHT_KV_INPUT_INDEX = 1;
    constexpr uint32_t WEIGHT_WGATE_INPUT_INDEX = 2;
    constexpr uint32_t KV_STATE_INPUT_INDEX = 3;
    constexpr uint32_t SCORE_STATE_INPUT_INDEX = 4;
    constexpr uint32_t APE_INPUT_INDEX = 5;
    constexpr uint32_t NORM_WEIGHT_INPUT_INDEX = 6;
    constexpr uint32_t ROPE_SIN_INPUT_INDEX = 7;
    constexpr uint32_t ROPE_COS_INPUT_INDEX = 8;

    // INPUT(OPTION)
    constexpr uint32_t KV_BLOCK_TABLE_INPUT_INDEX = 9;
    constexpr uint32_t SCORE_BLOCK_TABLE_INPUT_INDEX = 10;
    constexpr uint32_t CU_SEQ_LEN_INPUT_INDEX = 11;
    constexpr uint32_t SEQ_USED_INPUT_INDEX = 12;
    constexpr uint32_t START_POS_INPUT_INDEX = 13;

    // ATTR
    constexpr uint32_t ROPE_HEAD_DIM_ATTR_INDEX = 0;
    constexpr uint32_t CMP_RATIO_ATTR_INDEX = 1;
    constexpr uint32_t COFF_ATTR_INDEX = 2;
    constexpr uint32_t NORM_EPS_ATTR_INDEX = 3;
    constexpr uint32_t ROTARY_MODE_ATTR_INDEX = 4;
    constexpr uint32_t ENABLE_GRAD_ATTR_INDEX = 5;

    // OUTPUT
    constexpr uint32_t CMP_KV_OUTPUT_INDEX = 0;
    constexpr uint32_t WKV_PROJ_OUTPUT_INDEX = 1;
    constexpr uint32_t SOFTMAX_RES_OUTPUT_INDEX = 2;
    constexpr uint32_t NORM_X_OUTPUT_INDEX = 3;
    constexpr uint32_t NORM_RSTD_OUTPUT_INDEX = 4;

    // ATTR DEFAULT VALUE
    constexpr uint32_t CMP_RATIO_VALUE = 4;
    constexpr uint32_t COFF_VALUE = 1;

struct CompressorProtoShapeParam {
    bool isBsMerge { false };
    int64_t B { 0 };
    int64_t T { 0 };
    int64_t S { 0 };
    int64_t Sr { 0 };
    int64_t H { 0 };
    int64_t D { 0 };
};

// tmp
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;

ge::graphStatus GetCompressorShapeDim(const gert::InferShapeContext* context, CompressorProtoShapeParam &shapeParam)
{
    auto xShape = context->GetRequiredInputShape(TOKEN_X_INPUT_INDEX);      // (B, S, H) | (T, H)
    OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED)
    auto wkvShape = context->GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);  // (coff * D, H)
    OPS_LOG_E_IF_NULL(context, wkvShape, return ge::GRAPH_FAILED)
    auto wgateShape = context->GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);  // (coff * D, H)
    OPS_LOG_E_IF_NULL(context, wgateShape, return ge::GRAPH_FAILED)
    auto kvStateShape = context->GetRequiredInputShape(KV_STATE_INPUT_INDEX);    // (block_num, block_size, coff * D)
    OPS_LOG_E_IF_NULL(context, kvStateShape, return ge::GRAPH_FAILED)
    auto scoreStateShape = context->GetRequiredInputShape(SCORE_STATE_INPUT_INDEX);    // (block_num, block_size, coff * D)
    OPS_LOG_E_IF_NULL(context, scoreStateShape, return ge::GRAPH_FAILED)
    auto apeShape = context->GetRequiredInputShape(APE_INPUT_INDEX);    // (r, coff * D)
    OPS_LOG_E_IF_NULL(context, apeShape, return ge::GRAPH_FAILED)
    auto normWeightShape = context->GetRequiredInputShape(NORM_WEIGHT_INPUT_INDEX);    // (D)
    OPS_LOG_E_IF_NULL(context, normWeightShape, return ge::GRAPH_FAILED)
    auto ropeSinShape = context->GetRequiredInputShape(ROPE_SIN_INPUT_INDEX);    // (B, ceil(S / r), rD) | (min(T, T/r + B), rD)
    OPS_LOG_E_IF_NULL(context, ropeSinShape, return ge::GRAPH_FAILED)
    auto ropeCosShape = context->GetRequiredInputShape(ROPE_COS_INPUT_INDEX);    // (B, ceil(S / r), rD) | (min(T, T/r + B), rD)
    OPS_LOG_E_IF_NULL(context, ropeCosShape, return ge::GRAPH_FAILED)
    auto kvBlockTableShape = context->GetRequiredInputShape(KV_BLOCK_TABLE_INPUT_INDEX);    // (B, sMax/block_size)
    OPS_LOG_E_IF_NULL(context, kvBlockTableShape, return ge::GRAPH_FAILED)
    auto scoreBlockTableShape = context->GetRequiredInputShape(SCORE_BLOCK_TABLE_INPUT_INDEX);    // (B, sMax/block_size)
    OPS_LOG_E_IF_NULL(context, scoreBlockTableShape, return ge::GRAPH_FAILED)
    auto cuSeqlensShape = context->GetRequiredInputShape(CU_SEQ_LEN_INPUT_INDEX);    // (B+1,)
    OPS_LOG_E_IF_NULL(context, cuSeqlensShape, return ge::GRAPH_FAILED)
    auto seqUsedShape = context->GetRequiredInputShape(SEQ_USED_INPUT_INDEX);    // (B,)
    OPS_LOG_E_IF_NULL(context, seqUsedShape, return ge::GRAPH_FAILED)
    auto startPosShape = context->GetRequiredInputShape(START_POS_INPUT_INDEX);    // (B,)
    OPS_LOG_E_IF_NULL(context, startPosShape, return ge::GRAPH_FAILED)

    if (xShape->GetDimNum() == DIM_NUM_3) {                // BS
        shapeParam.isBsMerge = false;
        shapeParam.B = xShape->GetDim(DIM_INDEX_0);
        shapeParam.S = xShape->GetDim(DIM_INDEX_1);
        shapeParam.H = xShape->GetDim(DIM_INDEX_2);
        shapeParam.T = shapeParam.B * shapeParam.S;
    } else {                                                    // T
        shapeParam.isBsMerge = true;
        shapeParam.T = xShape->GetDim(DIM_INDEX_0);
        shapeParam.H = xShape->GetDim(DIM_INDEX_1);
    }

    shapeParam.D = normWeightShape->GetDim(DIM_INDEX_0);
    shapeParam.Sr = ropeSinShape->GetDim(DIM_INDEX_1);

    return GRAPH_SUCCESS;
}

ge::graphStatus SetCompressorShapeDim(const CompressorProtoShapeParam &shapeParam, gert::InferShapeContext* context)
{
    auto cmpKvShape = context->GetOutputShape(CMP_KV_OUTPUT_INDEX);                 // query: (B, S, N, Hckv) | (T, N, Hckv)
    OPS_LOG_E_IF_NULL(context, cmpKvShape, return ge::GRAPH_FAILED)
    auto wkvProjShape = context->GetOutputShape(WKV_PROJ_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, wkvProjShape, return ge::GRAPH_FAILED)
    auto softmaxResShape = context->GetOutputShape(SOFTMAX_RES_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, softmaxResShape, return ge::GRAPH_FAILED)
    auto normXShape = context->GetOutputShape(NORM_X_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, normXShape, return ge::GRAPH_FAILED)
    auto normRstdShape = context->GetOutputShape(NORM_RSTD_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, normRstdShape, return ge::GRAPH_FAILED)
    auto attr = context->GetAttrs();
    const uint32_t *cmpRatioPtr = attr->GetAttrPointer<uint32_t>(CMP_RATIO_ATTR_INDEX);
    uint32_t cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : CMP_RATIO_VALUE;
    const uint32_t *coffPtr = attr->GetAttrPointer<uint32_t>(COFF_ATTR_INDEX);
    uint32_t coff = (coffPtr != nullptr) ? *coffPtr : COFF_VALUE;
    const bool *enableGradPtr = attr->GetAttrPointer<bool>(ENABLE_GRAD_ATTR_INDEX);
    bool enableGrad = (enableGradPtr != nullptr) ? *enableGradPtr : false;
    // Set output shape
    if (!shapeParam.isBsMerge) {
        cmpKvShape->SetDimNum(DIM_NUM_3);                   // (B, Sr, H)
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.B);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_2, shapeParam.H);
        if (enableGrad) {
            wkvProjShape->SetDimNum(DIM_NUM_3);
            wkvProjShape->SetDim(DIM_INDEX_0, shapeParam.B);
            wkvProjShape->SetDim(DIM_INDEX_1, shapeParam.S);
            wkvProjShape->SetDim(DIM_INDEX_2, coff * shapeParam.D);
            softmaxResShape->SetDimNum(DIM_NUM_4);
            softmaxResShape->SetDim(DIM_INDEX_0, shapeParam.B);
            softmaxResShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
            softmaxResShape->SetDim(DIM_INDEX_2, coff * cmpRatio);
            softmaxResShape->SetDim(DIM_INDEX_3, shapeParam.D);
            normXShape->SetDimNum(DIM_NUM_3);
            normXShape->SetDim(DIM_INDEX_0, shapeParam.B);
            normXShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
            normXShape->SetDim(DIM_INDEX_2, coff * shapeParam.D);
            normRstdShape->SetDimNum(DIM_NUM_2);
            normRstdShape->SetDim(DIM_INDEX_0, shapeParam.B);
            normRstdShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
        } else {
            wkvProjShape->SetDimNum(DIM_NUM_3);
            wkvProjShape->SetDim(DIM_INDEX_0, 0);
            wkvProjShape->SetDim(DIM_INDEX_1, 0);
            wkvProjShape->SetDim(DIM_INDEX_2, 0);
            softmaxResShape->SetDimNum(DIM_NUM_4);
            softmaxResShape->SetDim(DIM_INDEX_0, 0);
            softmaxResShape->SetDim(DIM_INDEX_1, 0);
            softmaxResShape->SetDim(DIM_INDEX_2, 0);
            softmaxResShape->SetDim(DIM_INDEX_3, 0);
            normXShape->SetDimNum(DIM_NUM_3);
            normXShape->SetDim(DIM_INDEX_0, 0);
            normXShape->SetDim(DIM_INDEX_1, 0);
            normXShape->SetDim(DIM_INDEX_2, 0);
            normRstdShape->SetDimNum(DIM_NUM_2);
            normRstdShape->SetDim(DIM_INDEX_0, 0);
            normRstdShape->SetDim(DIM_INDEX_1, 0);
        }
    } else {
        cmpKvShape->SetDimNum(DIM_NUM_2);                   // (T, N, Hckv)
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.H);
        if (enableGrad) {
            wkvProjShape->SetDimNum(DIM_NUM_2);
            wkvProjShape->SetDim(DIM_INDEX_0, shapeParam.T);
            wkvProjShape->SetDim(DIM_INDEX_1, coff * shapeParam.D);
            softmaxResShape->SetDimNum(DIM_NUM_3);
            softmaxResShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
            softmaxResShape->SetDim(DIM_INDEX_1, coff * cmpRatio);
            softmaxResShape->SetDim(DIM_INDEX_2, shapeParam.D);
            normXShape->SetDimNum(DIM_NUM_2);
            normXShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
            normXShape->SetDim(DIM_INDEX_1, coff * shapeParam.D);
            normRstdShape->SetDimNum(DIM_NUM_1);
            normRstdShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
        } else {
            wkvProjShape->SetDimNum(DIM_NUM_2);
            wkvProjShape->SetDim(DIM_INDEX_0, 0);
            wkvProjShape->SetDim(DIM_INDEX_1, 0);
            softmaxResShape->SetDimNum(DIM_NUM_3);
            softmaxResShape->SetDim(DIM_INDEX_0, 0);
            softmaxResShape->SetDim(DIM_INDEX_1, 0);
            softmaxResShape->SetDim(DIM_INDEX_2, 0);
            normXShape->SetDimNum(DIM_NUM_2);
            normXShape->SetDim(DIM_INDEX_0, 0);
            normXShape->SetDim(DIM_INDEX_1, 0);
            normRstdShape->SetDimNum(DIM_NUM_1);
            normRstdShape->SetDim(DIM_INDEX_0, 0);
        }
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeCompressor(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Compressor", "Context is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_LOG_I(context->GetNodeName(), "Enter Compressor inferDataType impl.");

    context->SetOutputDataType(CMP_KV_OUTPUT_INDEX, context->GetRequiredInputDataType(TOKEN_X_INPUT_INDEX));

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeCompressor(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Compressor", "Context is nullptr."),
               return ge::GRAPH_FAILED);
    OPS_LOG_I(context->GetNodeName(), "Enter Compressor infershape impl.");

    CompressorProtoShapeParam shapeParam {};
    auto apiRet = GetCompressorShapeDim(context, shapeParam);
    OPS_LOG_E_IF((apiRet != GRAPH_SUCCESS), context, return ge::GRAPH_FAILED, "Context get input shape failed");

    apiRet = SetCompressorShapeDim(shapeParam, context);
    OPS_LOG_E_IF((apiRet != GRAPH_SUCCESS), context, return ge::GRAPH_FAILED, "Context set output shape failed");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Compressor).InferShape(InferShapeCompressor).InferDataType(InferDataTypeCompressor);
}  // namespace ops