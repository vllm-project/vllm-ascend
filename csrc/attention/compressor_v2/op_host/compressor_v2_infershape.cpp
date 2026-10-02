/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
// INPUT
constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
constexpr uint32_t WEIGHT_KV_INPUT_INDEX = 1;
constexpr uint32_t WEIGHT_WGATE_INPUT_INDEX = 2;
constexpr uint32_t STATE_CACHE_INPUT_INDEX = 3;

// INPUT(OPTION)
constexpr uint32_t STATE_BLOCK_TABLE_INPUT_INDEX = 4;
constexpr uint32_t CU_SEQ_LEN_INPUT_INDEX = 5;
constexpr uint32_t SEQ_USED_INPUT_INDEX = 6;
constexpr uint32_t START_POS_INPUT_INDEX = 7;

// ATTR
constexpr uint32_t CMP_RATIO_ATTR_INDEX = 0;
constexpr uint32_t STATE_CACHE_STRIDE_DIM0_ATTR_INDEX = 1;

// OUTPUT
constexpr uint32_t CMP_KV_OUTPUT_INDEX = 0;
constexpr uint32_t STATE_CACHE_OUTPUT_INDEX = 1;

// ATTR DEFAULT VALUE
constexpr uint32_t CMP_RATIO_VALUE = 4;
// V2 固定 coff=1（COFF::DISABLE），不再作为 attr 传入
constexpr uint32_t COFF_VALUE = 1;

struct CompressorV2ProtoShapeParam {
    bool isBsMerge{false};
    int64_t B{0};
    int64_t T{0};
    int64_t S{0};
    int64_t Sr{0};
    int64_t H{0};
    int64_t D{0};
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

ge::graphStatus GetCompressorV2ShapeDim(const gert::InferShapeContext *context, CompressorV2ProtoShapeParam &shapeParam)
{
    auto xShape = context->GetRequiredInputShape(TOKEN_X_INPUT_INDEX); // (B, S, H) | (T, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto wkvShape = context->GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX); // (D, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, wkvShape);
    auto wgateShape = context->GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX); // (D, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, wgateShape);
    // (block_num, block_size, 2 * D) | (B, token_size, 2 * D)
    auto stateCacheShape = context->GetRequiredInputShape(STATE_CACHE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, stateCacheShape);

    // TH 布局下 cu_seqlens 必填；state_block_table / seqused / start_pos 均为可选输入，
    // 传 nullptr 表示对应功能未启用，infershape 不做非空校验（与 v1 一致）
    if (xShape->GetDimNum() == DIM_NUM_2) {
        auto cuSeqlensShape = context->GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX); // (B+1,)
        OP_CHECK_NULL_WITH_CONTEXT(context, cuSeqlensShape);
    }

    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    const int64_t *cmpRatioPtr = attr->GetAttrPointer<int64_t>(CMP_RATIO_ATTR_INDEX);
    int64_t cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : CMP_RATIO_VALUE;
    // V2 coff 固定为 1（COFF::DISABLE）
    constexpr int64_t coff = COFF_VALUE;

    if (xShape->GetDimNum() == DIM_NUM_3) { // BS
        shapeParam.isBsMerge = false;
        shapeParam.B = xShape->GetDim(DIM_INDEX_0);
        shapeParam.S = xShape->GetDim(DIM_INDEX_1);
        shapeParam.Sr = (xShape->GetDim(DIM_INDEX_1) + cmpRatio - 1) / cmpRatio;
        shapeParam.H = xShape->GetDim(DIM_INDEX_2);
        shapeParam.T = shapeParam.B * shapeParam.S;
    } else { // T
        shapeParam.isBsMerge = true;
        auto cuSeqlensShape = context->GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
        shapeParam.Sr = std::min(xShape->GetDim(DIM_INDEX_0),
                                 xShape->GetDim(DIM_INDEX_0) / cmpRatio + cuSeqlensShape->GetDim(DIM_INDEX_0) - 1);
        shapeParam.T = xShape->GetDim(DIM_INDEX_0);
        shapeParam.H = xShape->GetDim(DIM_INDEX_1);
    }

    shapeParam.D = wkvShape->GetDim(DIM_INDEX_0) / coff;

    return GRAPH_SUCCESS;
}

ge::graphStatus SetCompressorV2ShapeDim(const CompressorV2ProtoShapeParam &shapeParam, gert::InferShapeContext *context)
{
    // query: (B, Sr, D) | (Sr, D)
    auto cmpKvShape = context->GetOutputShape(CMP_KV_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cmpKvShape);
    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    const int64_t *cmpRatioPtr = attr->GetAttrPointer<int64_t>(CMP_RATIO_ATTR_INDEX);
    int64_t cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : CMP_RATIO_VALUE;
    // V2 coff 固定为 1，extraDim = coff * cmpRatio = cmpRatio
    int64_t extraDim = cmpRatio;

    // Set output shape
    if (!shapeParam.isBsMerge) {
        cmpKvShape->SetDimNum(DIM_NUM_3); // (B, Sr, D)
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.B);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_2, shapeParam.D);
    } else {
        cmpKvShape->SetDimNum(DIM_NUM_2); // (Sr, D)
        cmpKvShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
        cmpKvShape->SetDim(DIM_INDEX_1, shapeParam.D);
    }

    // state_cache 为 in-place output（output index 1），shape 与 input 一致，框架自动处理，无需显式 set。

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeCompressorV2(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorV2", "context", "is nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter CompressorV2 inferDataType impl.");

    context->SetOutputDataType(CMP_KV_OUTPUT_INDEX, context->GetRequiredInputDataType(TOKEN_X_INPUT_INDEX));
    // state_cache output dtype 与 input state_cache 一致（in-place）
    context->SetOutputDataType(STATE_CACHE_OUTPUT_INDEX, context->GetRequiredInputDataType(STATE_CACHE_INPUT_INDEX));

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeCompressorV2(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorV2", "context", "is nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter CompressorV2 infershape impl.");

    CompressorV2ProtoShapeParam shapeParam{};
    auto apiRet = GetCompressorV2ShapeDim(context, shapeParam);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "context", "get input shape failed"),
                return ge::GRAPH_FAILED);

    apiRet = SetCompressorV2ShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "context", "set output shape failed"),
                return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CompressorV2).InferShape(InferShapeCompressorV2).InferDataType(InferDataTypeCompressorV2);
} // namespace ops
