/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_v2_sink_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"
#include "log/log.h"
using namespace ge;

namespace {
static constexpr uint32_t FIA_LAYOUT_DIM0 = 0;
static constexpr uint32_t FIA_LAYOUT_DIM1 = 1;
static constexpr uint32_t FIA_LAYOUT_DIM2 = 2;
static constexpr uint32_t FIA_LAYOUT_DIM3 = 3;
static constexpr uint32_t FIA_LAYOUT_DIM4 = 4;
static constexpr uint32_t FIA_LAYOUT_DIM_NUMS_1 = 1;
static constexpr uint32_t FIA_LAYOUT_DIM_NUMS_3 = 3;
static constexpr uint32_t FIA_LAYOUT_DIM_NUMS_4 = 4;
static constexpr uint32_t LAYOUT_BSH_DIM_NUMS = 3;
static constexpr uint32_t LAYOUT_BNSD_DIM_NUMS = 4;
static constexpr uint32_t LAYOUT_BSND_DIM_NUMS = 4;
static constexpr uint32_t LAYOUT_TND_DIM_NUMS = 3;
static constexpr uint32_t LAYOUT_NTD_DIM_NUMS = 3;
static constexpr uint32_t LAYOUT_NSD_DIM_NUMS = 3;
static constexpr int32_t FIA_UNKNOWN_DIMS = -2;
static constexpr uint32_t LAYOUT_PA_BBH_DIM_NUMS = 3;
static constexpr uint32_t LAYOUT_PA_BNBD_DIM_NUMS = 4;
static constexpr uint32_t LAYOUT_PA_NZ_DIM_NUMS = 5;
static constexpr uint32_t NUM_0 = 0;
static constexpr uint32_t NUM_1 = 1;
static constexpr uint32_t FIA_QUERY_INDEX = 0;
static constexpr uint32_t FIA_VALUE_INDEX = 2;
static constexpr uint32_t FIA_DYNAMIC_VALUE_INDEX = 0;
static constexpr uint32_t FIA_BLOCK_TABLE_INDEX = 14;
static constexpr uint32_t FIA_QUANT_SCALE2_INDEX = 10;
static constexpr uint32_t FIA_ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t FIA_SOFTMAX_LSE_INDEX = 1;
static constexpr uint32_t FIA_SOFTMAX_MAX_INDEX = 2;
static constexpr uint32_t FIA_SOFTMAX_SUM_INDEX = 3;
static constexpr uint32_t FIA_ATTR_NUM_HEADS_INDEX = 0;
static constexpr uint32_t FIA_ATTR_NUM_KV_HEADS_INDEX = 5;
static constexpr uint32_t FIA_ATTR_INPUT_LAYOUT_INDEX = 4;
static constexpr uint32_t FIA_INPUT_ACTUAL_SEQ_LENGTHS_INDEX = 5;
static constexpr uint32_t FIA_INPUT_ACTUAL_SEQ_LENGTHS_KV_INDEX = 6;
static constexpr uint32_t FIA_ATTR_INPUT_SOFTMAX_LSE_FLAG_INDEX = 10;
static constexpr uint32_t FIA_INPUT_QUERY_PADDING_SIZE_INDEX = 15;
static constexpr uint32_t FIA_INPUT_KV_PADDING_SIZE_INDEX = 16;
static constexpr uint32_t FIA_ATTR_SOFTMAX_MAX_SUM_FLAG_INDEX = 17;
static constexpr uint32_t FIA_QUERY_ROPE_INDEX = 24;
static constexpr uint32_t FIA_OUT_DTYPE_INDEX = 15;

static const std::map<int64_t, ge::DataType> TORCH_DTYPE_ENUM_VALUE_TO_GE_DTYPE_MAP = {
    {5, ge::DT_FLOAT16}, {15, ge::DT_BF16}, {24, ge::DT_FLOAT8_E4M3FN}, {290, ge::DT_HIFLOAT8}};

static ge::graphStatus GetQueryAndOutLayout(std::string &queryLayout,
                                            std::string &attentionOutLayout,
                                            const gert::Shape *queryShape,
                                            const char *inputLayoutPtr)
{
    struct parserLayout {
        std::string qLayout;
        std::string outLayout;
        int32_t qDim;
    };

    const std::map<std::string, parserLayout> LAYOUT_MAP = {{"BSH", {"BSH", "BSH", LAYOUT_BSH_DIM_NUMS}},
                                                            {"BSND", {"BSND", "BSND", LAYOUT_BSND_DIM_NUMS}},
                                                            {"BNSD", {"BNSD", "BNSD", LAYOUT_BNSD_DIM_NUMS}},
                                                            {"TND", {"TND", "TND", LAYOUT_TND_DIM_NUMS}},
                                                            {"NTD", {"NTD", "NTD", LAYOUT_NTD_DIM_NUMS}},
                                                            {"BNSD_BSND", {"BNSD", "BSND", LAYOUT_BNSD_DIM_NUMS}},
                                                            {"BSH_BNSD", {"BSH", "BNSD", LAYOUT_BSH_DIM_NUMS}},
                                                            {"BSND_BNSD", {"BSND", "BNSD", LAYOUT_BSND_DIM_NUMS}},
                                                            {"NTD_TND", {"NTD", "TND", LAYOUT_NTD_DIM_NUMS}},
                                                            {"BSH_NBSD", {"BSH", "NBSD", LAYOUT_BSH_DIM_NUMS}},
                                                            {"BSND_NBSD", {"BSND", "NBSD", LAYOUT_BSND_DIM_NUMS}},
                                                            {"BNSD_NBSD", {"BNSD", "NBSD", LAYOUT_BNSD_DIM_NUMS}},
                                                            {"TND_NTD", {"TND", "NTD", LAYOUT_TND_DIM_NUMS}},
                                                            {"NSD", {"NSD", "NSD", LAYOUT_NSD_DIM_NUMS}}};

    auto it = LAYOUT_MAP.find(std::string(inputLayoutPtr));
    if (it != LAYOUT_MAP.end()) {
        queryLayout = it->second.qLayout;
        attentionOutLayout = it->second.outLayout;
        int32_t queryDim = it->second.qDim;

        if (queryShape->GetDimNum() != queryDim) {
            OP_LOGE("FusedInferAttentionScoreV2Sink",
                    "Layout %s, query's dim(%zu) must be %zu!",
                    queryLayout.c_str(),
                    queryShape->GetDimNum(),
                    queryDim);
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE("FusedInferAttentionScoreV2Sink",
                "Only support layout BSH, BSND, BNSD, TND, NTD, BNSD_BSND, BSH_BNSD, "
                "BSND_BNSD, NTD_TND, BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND_NTD, but got %s.",
                *inputLayoutPtr);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQueryBSND(const gert::Shape *queryShape,
                                    const std::string queryLayout,
                                    const int64_t *numHeadsPtr,
                                    int64_t &b,
                                    int64_t &s1,
                                    int64_t &n1,
                                    int64_t &d1)
{
    if (queryLayout == "BSH") {
        b = (*queryShape)[FIA_LAYOUT_DIM0];
        s1 = (*queryShape)[FIA_LAYOUT_DIM1];
        n1 = (*numHeadsPtr);
        d1 = (*queryShape)[FIA_LAYOUT_DIM2] / (*numHeadsPtr);
    } else if (queryLayout == "BSND") {
        b = (*queryShape)[FIA_LAYOUT_DIM0];
        s1 = (*queryShape)[FIA_LAYOUT_DIM1];
        n1 = (*queryShape)[FIA_LAYOUT_DIM2];
        d1 = (*queryShape)[FIA_LAYOUT_DIM3];
    } else if (queryLayout == "BNSD") {
        b = (*queryShape)[FIA_LAYOUT_DIM0];
        s1 = (*queryShape)[FIA_LAYOUT_DIM2];
        n1 = (*queryShape)[FIA_LAYOUT_DIM1];
        d1 = (*queryShape)[FIA_LAYOUT_DIM3];
    } else if (queryLayout == "NSD") {
        b = 1;
        s1 = (*queryShape)[FIA_LAYOUT_DIM1];
        n1 = (*queryShape)[FIA_LAYOUT_DIM0];
        d1 = (*queryShape)[FIA_LAYOUT_DIM2];
    } else {
        OP_LOGE("FusedInferAttentionScoreV2Sink",
                "Layout %s is not supported in GetQueryBSND function!",
                queryLayout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQueryTND(
    const gert::Shape *queryShape,
    const std::string queryLayout,
    int64_t &t,
    int64_t &n1,
    int64_t &d1)
{
    if (queryLayout == "TND") {
        t = (*queryShape)[FIA_LAYOUT_DIM0];
        n1 = (*queryShape)[FIA_LAYOUT_DIM1];
        d1 = (*queryShape)[FIA_LAYOUT_DIM2];
    } else if (queryLayout == "NTD") {
        t = (*queryShape)[FIA_LAYOUT_DIM1];
        n1 = (*queryShape)[FIA_LAYOUT_DIM0];
        d1 = (*queryShape)[FIA_LAYOUT_DIM2];
    } else {
        OP_LOGE("FusedInferAttentionScoreV2Sink",
                "Layout %s is not supported in GetQueryTND function!",
                queryLayout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetValueD(bool isPageAttention,
                                 int64_t &valueD,
                                 const gert::Shape *valueShape,
                                 const gert::Shape *queryShape,
                                 const std::string queryLayout,
                                 int64_t numKeyValueHeads)
{
    if (isPageAttention) { // PA场景
        if (valueShape->GetDimNum() == LAYOUT_PA_BBH_DIM_NUMS) {
            valueD = (*valueShape)[FIA_LAYOUT_DIM2] / numKeyValueHeads;
        } else if (valueShape->GetDimNum() == LAYOUT_PA_BNBD_DIM_NUMS) {
            valueD = (*valueShape)[FIA_LAYOUT_DIM3];
        } else if (valueShape->GetDimNum() == LAYOUT_PA_NZ_DIM_NUMS) {
            valueD = (*valueShape)[FIA_LAYOUT_DIM2] * (*valueShape)[FIA_LAYOUT_DIM4];
        } else {
            OP_LOGE("FusedInferAttentionScoreV2Sink",
                    "when Page Attention enabled, value's dim should be 3/4/5, but got %zu.",
                    valueShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else { // 非PA场景
        if (valueShape->GetDimNum() != queryShape->GetDimNum()) {
            OP_LOGE("FusedInferAttentionScoreV2Sink",
                    "when Page Attention not enabled, value'dim(%zu) should equal to query's dim(%zu)!",
                    valueShape->GetDimNum(),
                    queryShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (queryLayout == "BSH") {
            valueD = (*valueShape)[FIA_LAYOUT_DIM2] / numKeyValueHeads;
        } else if (queryLayout == "BSND" || queryLayout == "BNSD") {
            valueD = (*valueShape)[FIA_LAYOUT_DIM3];
        } else if (queryLayout == "TND" || queryLayout == "NTD" || queryLayout == "NSD") {
            valueD = (*valueShape)[FIA_LAYOUT_DIM2];
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferAttentionOutShape(std::string attentionOutLayout,
                                              gert::Shape *attentionOutShape,
                                              const gert::Shape *queryShape,
                                              const gert::Shape *valueShape,
                                              const std::string queryLayout,
                                              const int64_t *numHeadsPtr,
                                              int64_t valueD)
{
    int64_t b = 0;
    int64_t s1 = 0;
    int64_t n1 = 0;
    int64_t d1 = 0;
    int64_t t = 0;
    if (attentionOutLayout == "BSH") {
        if (valueShape->GetDim(FIA_LAYOUT_DIM2) != -1) { // 动态图
            attentionOutShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_3);
            GetQueryBSND(queryShape, queryLayout, numHeadsPtr, b, s1, n1, d1);
            int64_t outH = (*numHeadsPtr) * valueD;
            outH = (outH == 0 || (*queryShape)[FIA_LAYOUT_DIM2] == 0) ? (*queryShape)[FIA_LAYOUT_DIM2] : outH;
            *attentionOutShape = {b, s1, outH};
        }
    } else if (attentionOutLayout == "BSND") {
        attentionOutShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_4);
        GetQueryBSND(queryShape, queryLayout, numHeadsPtr, b, s1, n1, d1);
        int64_t outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        *attentionOutShape = {b, s1, n1, outD};
    } else if (attentionOutLayout == "BNSD") {
        attentionOutShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_4);
        GetQueryBSND(queryShape, queryLayout, numHeadsPtr, b, s1, n1, d1);
        int64_t outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        *attentionOutShape = {b, n1, s1, outD};
    } else if (attentionOutLayout == "NBSD") {
        attentionOutShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_4);
        GetQueryBSND(queryShape, queryLayout, numHeadsPtr, b, s1, n1, d1);
        int64_t outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        *attentionOutShape = {n1, b, s1, outD};
    } else if (attentionOutLayout == "TND") {
        attentionOutShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_3);
        GetQueryTND(queryShape, queryLayout, t, n1, d1);
        int64_t outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        *attentionOutShape = {t, n1, outD};
    } else if (attentionOutLayout == "NTD") {
        attentionOutShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_3);
        GetQueryTND(queryShape, queryLayout, t, n1, d1);
        int64_t outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        *attentionOutShape = {n1, t, outD};
    } else if (attentionOutLayout == "NSD") {
        attentionOutShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_3);
        GetQueryBSND(queryShape, queryLayout, numHeadsPtr, b, s1, n1, d1);
        int64_t outD = valueD;
        outD = (outD == 0 || d1 == 0) ? d1 : outD;
        *attentionOutShape = {n1, s1, outD};
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferLseTypeOutShape(const char *inputLayoutPtr,
                                        gert::Shape *softmaxLseShape,
                                        const gert::Shape *queryShape,
                                        const std::string queryLayout,
                                        const int64_t *numHeadsPtr)
{
    int64_t n1 = 0;
    int64_t d1 = 0;
    if (strcmp(inputLayoutPtr, "TND") == 0 || strcmp(inputLayoutPtr, "NTD") == 0 ||
        strcmp(inputLayoutPtr, "TND_NTD") == 0 || strcmp(inputLayoutPtr, "NTD_TND") == 0) {
        int64_t t = 0;
        softmaxLseShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_3);
        GetQueryTND(queryShape, queryLayout, t, n1, d1);
        *softmaxLseShape = {t, n1, NUM_1};
    } else {
        int64_t b = 0;
        int64_t s1 = 0;
        softmaxLseShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_4);
        GetQueryBSND(queryShape, queryLayout, numHeadsPtr, b, s1, n1, d1);
        *softmaxLseShape = {b, n1, s1, NUM_1};
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeFusedInferAttentionScoreV2Sink(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("FusedInferAttentionScoreV2Sink", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter FusedInferAttentionScoreV2Sink InferShape impl.");
    // query shape
    const gert::Shape *queryShape = context->GetInputShape(FIA_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    // value shape
    const gert::Shape *valueShape = context->GetDynamicInputShape(FIA_VALUE_INDEX, FIA_DYNAMIC_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);

    // Page Attention
    bool isPageAttention = (context->GetOptionalInputShape(FIA_BLOCK_TABLE_INDEX) == nullptr) ? false : true;

    // attentionOut
    gert::Shape *attentionOutShape = context->GetOutputShape(FIA_ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);

    // lseout
    gert::Shape *softmaxLseShape = context->GetOutputShape(FIA_SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);

    // softmax_max / softmax_sum（与 lse 同布局）
    gert::Shape *softmaxMaxShape = context->GetOutputShape(FIA_SOFTMAX_MAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxMaxShape);
    gert::Shape *softmaxSumShape = context->GetOutputShape(FIA_SOFTMAX_SUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);

    // Get attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutPtr = attrs->GetAttrPointer<char>(FIA_ATTR_INPUT_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutPtr);
    const int64_t *numHeadsPtr = attrs->GetInt(FIA_ATTR_NUM_HEADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numHeadsPtr);
    const int64_t *numKeyValueHeadsPtr = attrs->GetInt(FIA_ATTR_NUM_KV_HEADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numKeyValueHeadsPtr);

    // KV_N除零保护, 当KV_N为零时KV_N = Q_N
    if (*numHeadsPtr == 0) {
        OP_LOGE(context->GetNodeName(), "numHeads can not be 0!");
        return ge::GRAPH_FAILED;
    }
    int64_t numKeyValueHeads = (*numKeyValueHeadsPtr == 0) ? *numHeadsPtr : *numKeyValueHeadsPtr;

    // set AttentionOut shape
    *attentionOutShape = *queryShape;

    // UNKNOWN DIM
    if (((queryShape->GetDimNum() == FIA_LAYOUT_DIM_NUMS_1) &&
         (queryShape->GetDim(FIA_LAYOUT_DIM0) == FIA_UNKNOWN_DIMS)) ||
        ((valueShape->GetDimNum() == FIA_LAYOUT_DIM_NUMS_1) &&
         (valueShape->GetDim(FIA_LAYOUT_DIM0) == FIA_UNKNOWN_DIMS))) {
        attentionOutShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_1);
        (*attentionOutShape)[FIA_LAYOUT_DIM0] = FIA_UNKNOWN_DIMS;
        softmaxLseShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_1);
        (*softmaxLseShape)[FIA_LAYOUT_DIM0] = FIA_UNKNOWN_DIMS;
        softmaxMaxShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_1);
        (*softmaxMaxShape)[FIA_LAYOUT_DIM0] = FIA_UNKNOWN_DIMS;
        softmaxSumShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_1);
        (*softmaxSumShape)[FIA_LAYOUT_DIM0] = FIA_UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    std::string queryLayout = "BSH";
    std::string attentionOutLayout = "BSH";
    GetQueryAndOutLayout(queryLayout, attentionOutLayout, queryShape, inputLayoutPtr);

    int64_t valueD = 0;
    GetValueD(isPageAttention, valueD, valueShape, queryShape, queryLayout, numKeyValueHeads);

    InferAttentionOutShape(
        attentionOutLayout, attentionOutShape, queryShape, valueShape, queryLayout, numHeadsPtr, valueD);

    const bool *softmaxLsePtr = attrs->GetAttrPointer<bool>(FIA_ATTR_INPUT_SOFTMAX_LSE_FLAG_INDEX);
    bool softmaxLseFlag = (softmaxLsePtr != nullptr) ? *softmaxLsePtr : false;
    const bool *softmaxMaxSumFlagPtr = attrs->GetAttrPointer<bool>(FIA_ATTR_SOFTMAX_MAX_SUM_FLAG_INDEX);
    bool softmaxMaxSumFlag = (softmaxMaxSumFlagPtr != nullptr) ? *softmaxMaxSumFlagPtr : false;
    // softmaxLseFlag 独立控制 softmaxLse shape
    if (softmaxLseFlag) {
        InferLseTypeOutShape(inputLayoutPtr, softmaxLseShape, queryShape, queryLayout, numHeadsPtr);
    } else {
        softmaxLseShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_1);
        (*softmaxLseShape)[FIA_LAYOUT_DIM0] = NUM_0;
    }

    // softmaxMaxSumFlag 独立控制 softmaxMax/softmaxSum shape
    if (softmaxMaxSumFlag) {
        InferLseTypeOutShape(inputLayoutPtr, softmaxMaxShape, queryShape, queryLayout, numHeadsPtr);
        InferLseTypeOutShape(inputLayoutPtr, softmaxSumShape, queryShape, queryLayout, numHeadsPtr);
    } else {
        softmaxMaxShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_1);
        (*softmaxMaxShape)[FIA_LAYOUT_DIM0] = NUM_0;
        softmaxSumShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_1);
        (*softmaxSumShape)[FIA_LAYOUT_DIM0] = NUM_0;
    }

    OP_LOGD(context->GetNodeName(), "FusedInferAttentionScoreV2Sink InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFusedInferAttentionScoreV2Sink(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("FusedInferAttentionScoreV2Sink", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter FusedInferAttentionScoreV2Sink InferDataType impl.");
    // default set q's dtype as fia's output type
    ge::DataType outputType = context->GetInputDataType(FIA_QUERY_INDEX);
    // 10 is quant_scale2's index, if not instantiated or illegal return ge::DT_UNDEFINED
    if (context->GetOptionalInputDataType(FIA_QUANT_SCALE2_INDEX) != ge::DT_UNDEFINED) {
        outputType = ge::DT_INT8;

        auto attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const int64_t *outTypePtr = attrs->GetInt(FIA_OUT_DTYPE_INDEX);
        if (outTypePtr != nullptr) {
            auto iter = TORCH_DTYPE_ENUM_VALUE_TO_GE_DTYPE_MAP.find(*outTypePtr);
            if (iter != TORCH_DTYPE_ENUM_VALUE_TO_GE_DTYPE_MAP.end()) {
                outputType = iter->second;
            }
        }
    } else if (context->GetInputDataType(FIA_QUERY_INDEX) == ge::DT_INT8 ||
               context->GetInputDataType(FIA_QUERY_INDEX) == ge::DT_FLOAT8_E4M3FN ||
               context->GetInputDataType(FIA_QUERY_INDEX) == ge::DT_HIFLOAT8) {
        // 1. MLA: if the dtype of input query is int8, the dtype of output is same as the dtype of input query_rope.
        // 2. GQA: the int8 dtype of input query is not supported.
        outputType = context->GetOptionalInputDataType(FIA_QUERY_ROPE_INDEX);
        if (outputType == ge::DT_UNDEFINED) {
            outputType = ge::DT_FLOAT16;
        }
        auto attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const int64_t *outTypePtr = attrs->GetInt(FIA_OUT_DTYPE_INDEX);
        if (outTypePtr != nullptr) {
            auto iter = TORCH_DTYPE_ENUM_VALUE_TO_GE_DTYPE_MAP.find(*outTypePtr);
            if (iter != TORCH_DTYPE_ENUM_VALUE_TO_GE_DTYPE_MAP.end()) {
                outputType = iter->second;
            }
        }
    }
    // attention_out, outidx:0
    context->SetOutputDataType(FIA_ATTENTION_OUT_INDEX, outputType);
    context->SetOutputDataType(FIA_SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(FIA_SOFTMAX_MAX_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(FIA_SOFTMAX_SUM_INDEX, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "FusedInferAttentionScoreV2Sink InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedInferAttentionScoreV2Sink)
    .InferShape(InferShapeFusedInferAttentionScoreV2Sink)
    .InferDataType(InferDataTypeFusedInferAttentionScoreV2Sink);
} // namespace ops