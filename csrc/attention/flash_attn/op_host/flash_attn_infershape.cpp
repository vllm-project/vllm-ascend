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
 * \file flash_attn_infershape.cpp
 * \brief FlashAttn算子InferShape实现
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

// 属性索引 对其def注册索引
static constexpr size_t ATTR_IDX_SOFTMAX_SCALE = 0;
static constexpr size_t ATTR_IDX_MASK_MODE = 1;
static constexpr size_t ATTR_IDX_WIN_LEFT = 2;
static constexpr size_t ATTR_IDX_WIN_RIGHT = 3;
static constexpr size_t ATTR_IDX_MAX_SEQLEN_Q = 4;
static constexpr size_t ATTR_IDX_MAX_SEQLEN_KV = 5;
static constexpr size_t ATTR_IDX_LAYOUT_Q = 6;
static constexpr size_t ATTR_IDX_LAYOUT_KV = 7;
static constexpr size_t ATTR_IDX_LAYOUT_OUT = 8;
static constexpr size_t ATTR_IDX_RETURN_SOFTMAX_LSE = 9;
static constexpr size_t ATTR_IDX_DETERMINISTIC = 10;

// 输入索引
static constexpr size_t INPUT_IDX_Q = 0;
static constexpr size_t INPUT_IDX_K = 1;
static constexpr size_t INPUT_IDX_V = 2;
// 输出索引
static constexpr size_t OUTPUT_IDX_ATTN_OUT = 0;
static constexpr size_t OUTPUT_IDX_SOFTMAX_LSE = 1;

static constexpr int FA_SOFTMAX_LSE_LAST_DIM = 1; // softmax_lse最后一维元素数（每head一个float）

ge::graphStatus InferShapeFlashAttn(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *qShape = context->GetInputShape(INPUT_IDX_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    const gert::Shape *kShape = context->GetInputShape(INPUT_IDX_K);
    OP_CHECK_NULL_WITH_CONTEXT(context, kShape);
    const gert::Shape *vShape = context->GetInputShape(INPUT_IDX_V);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char *layoutQ = attrs->GetAttrPointer<char>(ATTR_IDX_LAYOUT_Q);
    const char *layoutKv = attrs->GetAttrPointer<char>(ATTR_IDX_LAYOUT_KV);
    const char *layoutOut = attrs->GetAttrPointer<char>(ATTR_IDX_LAYOUT_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutQ);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutKv);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutOut);

    auto returnSoftmaxLsePtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_RETURN_SOFTMAX_LSE);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnSoftmaxLsePtr);
    int64_t returnSoftmaxLse = *returnSoftmaxLsePtr;

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKvStr = std::string(layoutKv);
    std::string layoutOutStr = std::string(layoutOut);

    // 转为大写以便统一比较
    for (auto &c : layoutQStr) {
        c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
    }
    for (auto &c : layoutKvStr) {
        c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
    }
    for (auto &c : layoutOutStr) {
        c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
    }


    int64_t batchSize = 1;
    int64_t numHeadsQ = 0;
    int64_t seqLenQ = 0;
    int64_t headDim = 0;
    bool isTND = false;

    if (layoutQStr == "BSND") {
        if (qShape->GetDimNum() != 4) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "q",
                                                     std::to_string(qShape->GetDimNum()).c_str(),
                                                     "The shape dim of q must be 4 when layout_q is BSND");
            return ge::GRAPH_FAILED;
        }
        batchSize = qShape->GetDim(0);
        seqLenQ = qShape->GetDim(1);
        numHeadsQ = qShape->GetDim(2);
        headDim = qShape->GetDim(3);
    } else if (layoutQStr == "BNSD") {
        if (qShape->GetDimNum() != 4) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "q",
                                                     std::to_string(qShape->GetDimNum()).c_str(),
                                                     "The shape dim of q must be 4 when layout_q is BNSD");
            return ge::GRAPH_FAILED;
        }
        batchSize = qShape->GetDim(0);
        numHeadsQ = qShape->GetDim(1);
        seqLenQ = qShape->GetDim(2);
        headDim = qShape->GetDim(3);
    } else if (layoutQStr == "TND") {
        if (qShape->GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "q",
                                                     std::to_string(qShape->GetDimNum()).c_str(),
                                                     "The shape dim of q must be 3 when layout_q is TND");
            return ge::GRAPH_FAILED;
        }
        seqLenQ = qShape->GetDim(0); // T = total tokens
        numHeadsQ = qShape->GetDim(1);
        headDim = qShape->GetDim(2);
        isTND = true;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "layout_q", layoutQStr.c_str(),
                                              "The value of layout_q must be in BSND/BNSD/TND");
        return ge::GRAPH_FAILED;
    }

    int64_t headDimV = headDim;
    if ((layoutKvStr == "BSND" || layoutKvStr == "BNSD") && vShape->GetDimNum() == 4) {
        headDimV = vShape->GetDim(3);
    } else if (layoutKvStr == "TND" && vShape->GetDimNum() == 3) {
        headDimV = vShape->GetDim(2);
    } else if ((layoutKvStr == "PA_BBND" || layoutKvStr == "PA_BNBD") && vShape->GetDimNum() == 4) {
        // PA_BBND (Bn,Bs,N2,D) / PA_BNBD (Bn,N2,Bs,D) 的 D 维在 index 3
        headDimV = vShape->GetDim(3);
    } else if (layoutKvStr == "PA_NZ" && vShape->GetDimNum() == 5) {
        // PA_NZ (Bn,N2,D/16,block_size,16) 的 D = dim2 * dim4
        headDimV = vShape->GetDim(2) * vShape->GetDim(4);
    }

    gert::Shape *attnOutShape = context->GetOutputShape(OUTPUT_IDX_ATTN_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, attnOutShape);

    if (layoutOutStr == "BSND") {
        attnOutShape->SetDimNum(4);
        attnOutShape->SetDim(0, batchSize);
        attnOutShape->SetDim(1, seqLenQ);
        attnOutShape->SetDim(2, numHeadsQ);
        attnOutShape->SetDim(3, headDimV);
    } else if (layoutOutStr == "BNSD") {
        attnOutShape->SetDimNum(4);
        attnOutShape->SetDim(0, batchSize);
        attnOutShape->SetDim(1, numHeadsQ);
        attnOutShape->SetDim(2, seqLenQ);
        attnOutShape->SetDim(3, headDimV);
    } else if (layoutOutStr == "TND") {
        attnOutShape->SetDimNum(3);
        attnOutShape->SetDim(0, seqLenQ); // T总token数
        attnOutShape->SetDim(1, numHeadsQ);
        attnOutShape->SetDim(2, headDimV);
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "layout_out", layoutOutStr.c_str(),
                                              "The value of layout_out must be in BSND/BNSD/TND");
        return ge::GRAPH_FAILED;
    }

    gert::Shape *lseShape = context->GetOutputShape(OUTPUT_IDX_SOFTMAX_LSE);
    if (lseShape != nullptr) {
        if (returnSoftmaxLse != 0) {
            if (isTND) {
                lseShape->SetDimNum(2);
                lseShape->SetDim(0, seqLenQ);
                lseShape->SetDim(1, numHeadsQ);
            } else {
                lseShape->SetDimNum(3);
                lseShape->SetDim(0, batchSize);
                lseShape->SetDim(1, numHeadsQ);
                lseShape->SetDim(2, seqLenQ);
            }
        } else {
            // 不输出softmax_lse时设置为空shape
            lseShape->SetDimNum(1);
            lseShape->SetDim(0, 0);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeFlashAttn(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // attn_out数据类型与q一致
    auto qDtype = context->GetInputDataType(INPUT_IDX_Q);
    context->SetOutputDataType(OUTPUT_IDX_ATTN_OUT, qDtype);
    // softmax_lse固定为FLOAT32
    context->SetOutputDataType(OUTPUT_IDX_SOFTMAX_LSE, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlashAttn).InferShape(InferShapeFlashAttn).InferDataType(InferDataTypeFlashAttn);

} // namespace ops
