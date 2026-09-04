/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "log/error_code.h"

using namespace ge;

namespace ops {
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
static constexpr uint32_t FIA_ATTR_NUM_HEADS_INDEX = 0;
static constexpr uint32_t FIA_ATTR_NUM_KV_HEADS_INDEX = 5;
static constexpr uint32_t FIA_ATTR_INPUT_LAYOUT_INDEX = 4;
static constexpr uint32_t FIA_INPUT_ACTUAL_SEQ_LENGTHS_INDEX = 5;
static constexpr uint32_t FIA_INPUT_ACTUAL_SEQ_LENGTHS_KV_INDEX = 6;
static constexpr uint32_t FIA_ATTR_INPUT_SOFTMAX_LSE_FLAG_INDEX = 10;
static constexpr uint32_t FIA_INPUT_QUERY_PADDING_SIZE_INDEX = 15;
static constexpr uint32_t FIA_INPUT_KV_PADDING_SIZE_INDEX = 16;
static constexpr uint32_t FIA_INPUT_ACTUAL_SHARED_PREFIX_LEN_INDEX = 23;
static constexpr uint32_t FIA_QUERY_ROPE_INDEX = 24;
static constexpr uint32_t FIA_OUT_DTYPE_INDEX = 15;

static const std::map<int64_t, ge::DataType> TORCH_DTYPE_ENUM_VALUE_TO_GE_DTYPE_MAP = {
    {5,  ge::DT_FLOAT16}, 
    {15, ge::DT_BF16},
    {24, ge::DT_FLOAT8_E4M3FN},
    {290, ge::DT_HIFLOAT8}
};

static const std::map<int64_t, std::string> TORCH_DTYOE_NOT_SUPPORT_MAP = {
    {23,  "DT_FLOAT8_E5M2"}, 
};

static ge::graphStatus GetQueryAndOutLayout(std::string& queryLayout,
                                            std::string& attentionOutLayout,
                                            const gert::Shape *queryShape,
                                            const char *inputLayoutPtr)
{
    struct parserLayout {
        std::string qLayout;
        std::string outLayout;
        int32_t qDim;
    };

    const std::map<std::string, parserLayout> LAYOUT_MAP = {
        {"BSH",              {"BSH", "BSH", LAYOUT_BSH_DIM_NUMS}},
        {"BSND",             {"BSND", "BSND", LAYOUT_BSND_DIM_NUMS}},
        {"BNSD",             {"BNSD", "BNSD", LAYOUT_BNSD_DIM_NUMS}},
        {"TND",              {"TND", "TND", LAYOUT_TND_DIM_NUMS}},
        {"NTD",              {"NTD", "NTD", LAYOUT_NTD_DIM_NUMS}},
        {"BNSD_BSND",        {"BNSD", "BSND", LAYOUT_BNSD_DIM_NUMS}},
        {"BSH_BNSD",         {"BSH", "BNSD", LAYOUT_BSH_DIM_NUMS}},
        {"BSND_BNSD",        {"BSND", "BNSD", LAYOUT_BSND_DIM_NUMS}},
        {"NTD_TND",          {"NTD", "TND", LAYOUT_NTD_DIM_NUMS}},
        {"BSH_NBSD",         {"BSH", "NBSD", LAYOUT_BSH_DIM_NUMS}},
        {"BSND_NBSD",        {"BSND", "NBSD", LAYOUT_BSND_DIM_NUMS}},
        {"BNSD_NBSD",        {"BNSD", "NBSD", LAYOUT_BNSD_DIM_NUMS}},
        {"TND_NTD",          {"TND", "NTD", LAYOUT_TND_DIM_NUMS}},
        {"NSD",              {"NSD", "NSD", LAYOUT_NSD_DIM_NUMS}}
    };

    int32_t queryDim = 0;
    auto it = LAYOUT_MAP.find(std::string(inputLayoutPtr));
    if (it != LAYOUT_MAP.end()) {
        queryLayout = it->second.qLayout;
        attentionOutLayout = it->second.outLayout;
        queryDim = it->second.qDim;

        if (queryShape->GetDimNum() != queryDim) {
            std::string dimStr = std::to_string(queryShape->GetDimNum());
            std::string reason = "The shape dim of query must be " + std::to_string(queryDim) +
                " when layout is " + queryLayout;
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("FusedInferAttentionScore", "query",
                dimStr.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        std::string reason = "The value of input_layout must be in BSH, BSND, BNSD, TND, NTD, BNSD_BSND, BSH_BNSD, "
            "BSND_BNSD, NTD_TND, BSH_NBSD, BSND_NBSD, BNSD_NBSD, TND_NTD";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("FusedInferAttentionScore", "input_layout",
            std::string(inputLayoutPtr).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQueryBSND(const gert::Shape *queryShape,
                                   const std::string queryLayout,
                                   const int64_t *numHeadsPtr,
                                   int64_t& b, int64_t& s1, int64_t& n1, int64_t& d1)
{
    if (queryLayout == "BSH") {
        b = (*queryShape)[FIA_LAYOUT_DIM0];
        s1 = (*queryShape)[FIA_LAYOUT_DIM1];
        n1 = (*numHeadsPtr);
        if ((*queryShape)[FIA_LAYOUT_DIM2] == -1) {
            d1 = (*queryShape)[FIA_LAYOUT_DIM2];
        } else {
            d1 = (*queryShape)[FIA_LAYOUT_DIM2] / (*numHeadsPtr);
        }
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
        OP_LOGE("FusedInferAttentionScore", "Layout %s is not supported in GetQueryBSND function!", queryLayout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetQueryTND(const gert::Shape *queryShape,
                                  const std::string queryLayout,
                                  int64_t& t, int64_t& n1, int64_t& d1)
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
        OP_LOGE("FusedInferAttentionScore", "Layout %s is not supported in GetQueryTND function!", queryLayout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetValueD(bool isPageAttention, int64_t& valueD,
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
            std::string valueDimStr = std::to_string(valueShape->GetDimNum()) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("FusedInferAttentionScore", "value", valueDimStr.c_str(),
                "When Page Attention enabled, the shape dim of value should be 3D, 4D or 5D");
            return ge::GRAPH_FAILED;
        }
    } else { // 非PA场景
        if (valueShape->GetDimNum() != queryShape->GetDimNum()) {
            std::string dimMsg = std::to_string(valueShape->GetDimNum()) + " and " +
                std::to_string(queryShape->GetDimNum());
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON("FusedInferAttentionScore", "value and query", dimMsg.c_str(),
                "When Page Attention not enabled, the shape dim of value should be the same as query");
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
                                              const int64_t *numHeadsPtr, int64_t valueD)
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

static ge::graphStatus InferLseOutShape(const char *inputLayoutPtr,
                                        gert::Shape *softmaxLseShape,
                                        const gert::Shape *queryShape,
                                        const std::string queryLayout,
                                        const int64_t *numHeadsPtr)
{
    int64_t b = 0;
    int64_t s1 = 0;
    int64_t n1 = 0;
    int64_t d1 = 0;
    int64_t t = 0;
    if (strcmp(inputLayoutPtr, "TND") == 0 || strcmp(inputLayoutPtr, "NTD") == 0 ||
        strcmp(inputLayoutPtr, "TND_NTD") == 0 || strcmp(inputLayoutPtr, "NTD_TND") == 0) {
        softmaxLseShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_3);
        GetQueryTND(queryShape, queryLayout, t, n1, d1);
        *softmaxLseShape = {t, n1, NUM_1};
    } else {
        softmaxLseShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_4);
        GetQueryBSND(queryShape, queryLayout, numHeadsPtr, b, s1, n1, d1);
        *softmaxLseShape = {b, n1, s1, NUM_1};
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeFusedInferAttentionScore(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter FusedInferAttentionScore InferShape impl.");
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "num_heads",
            std::to_string(*numHeadsPtr).c_str(), "The value of num_heads cannot be 0");
        return ge::GRAPH_FAILED;
    }
    int64_t numKeyValueHeads = (*numKeyValueHeadsPtr == 0) ? *numHeadsPtr : *numKeyValueHeadsPtr;

    // set AttentionOut shape
    *attentionOutShape = *queryShape;

    // UNKNOWN DIM
    if (((queryShape->GetDimNum() == FIA_LAYOUT_DIM_NUMS_1) && (queryShape->GetDim(FIA_LAYOUT_DIM0) == FIA_UNKNOWN_DIMS))||
        ((valueShape->GetDimNum() == FIA_LAYOUT_DIM_NUMS_1) && (valueShape->GetDim(FIA_LAYOUT_DIM0) == FIA_UNKNOWN_DIMS))) {
        attentionOutShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_1);
        (*attentionOutShape)[FIA_LAYOUT_DIM0] = FIA_UNKNOWN_DIMS;
        softmaxLseShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_1);
        (*softmaxLseShape)[FIA_LAYOUT_DIM0] = FIA_UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    std::string queryLayout = "BSH";
    std::string attentionOutLayout = "BSH";
    GetQueryAndOutLayout(queryLayout, attentionOutLayout, queryShape, inputLayoutPtr);

    int64_t valueD = 0;
    GetValueD(isPageAttention, valueD, valueShape, queryShape, queryLayout, numKeyValueHeads);

    InferAttentionOutShape(attentionOutLayout, attentionOutShape, queryShape, valueShape, queryLayout, numHeadsPtr, valueD);

    const bool *softmaxLsePtr = attrs->GetAttrPointer<bool>(FIA_ATTR_INPUT_SOFTMAX_LSE_FLAG_INDEX);
    bool softmaxLseFlag = (softmaxLsePtr != nullptr) ? *softmaxLsePtr : false;
    if (softmaxLseFlag) {
        InferLseOutShape(inputLayoutPtr, softmaxLseShape, queryShape, queryLayout, numHeadsPtr);
    } else {
        softmaxLseShape->SetDimNum(FIA_LAYOUT_DIM_NUMS_1);
        (*softmaxLseShape)[FIA_LAYOUT_DIM0] = NUM_0;
    }

    OP_LOGD(context->GetNodeName(), "FusedInferAttentionScore InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFusedInferAttentionScore(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("FusedInferAttentionScore", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter FusedInferAttentionScore InferDataType impl.");
    // default set q's dtype as fia's output type
    ge::DataType outputType = context->GetInputDataType(FIA_QUERY_INDEX);
    // 10 is quant_scale2's index, if not instantiated or illegal return ge::DT_UNDEFINED
    if (context->GetOptionalInputDataType(FIA_QUANT_SCALE2_INDEX) != ge::DT_UNDEFINED) {
        outputType = ge::DT_INT8;
        
        auto attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        const int64_t *outTypePtr = attrs->GetInt(FIA_OUT_DTYPE_INDEX);
        if (outTypePtr != nullptr) {
            auto it = TORCH_DTYOE_NOT_SUPPORT_MAP.find(*outTypePtr);
            if (it != TORCH_DTYOE_NOT_SUPPORT_MAP.end()){
                std::string reason = "The dtype of output cannot be " + it->second + " when Fia graph mode is enabled";
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("FusedInferAttentionScore", "output",
                    it->second.c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
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
    OP_LOGD(context->GetNodeName(), "FusedInferAttentionScore InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedInferAttentionScore)
    .InferShape(InferShapeFusedInferAttentionScore)
    .InferDataType(InferDataTypeFusedInferAttentionScore)
    .InputsDataDependency({FIA_INPUT_ACTUAL_SEQ_LENGTHS_INDEX, FIA_INPUT_ACTUAL_SEQ_LENGTHS_KV_INDEX,
                           FIA_INPUT_QUERY_PADDING_SIZE_INDEX, FIA_INPUT_KV_PADDING_SIZE_INDEX,
                           FIA_INPUT_ACTUAL_SHARED_PREFIX_LEN_INDEX});
} // namespace ops