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
 * \file flash_mla_with_kvcache_infershape.cpp
 * \brief FlashMlaWithKvcache算子InferShape实现
 * \note 接口基准 flash_mla_with_kvcache：
 *       q/k_cache 最后维必须为 576 = head_dim_v(nope 512) + rope 64；
 *       attn_out 的最后维取 head_dim_v（q 的 576 含 rope，不能直接作为输出宽度）。
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

// 属性索引 对齐def注册索引（head_dim_v 在首位）
static constexpr size_t ATTR_IDX_HEAD_DIM_V = 0;
static constexpr size_t ATTR_IDX_SOFTMAX_SCALE = 1;
static constexpr size_t ATTR_IDX_MASK_MODE = 2;
static constexpr size_t ATTR_IDX_MAX_SEQLEN_Q = 3;
static constexpr size_t ATTR_IDX_MAX_SEQLEN_KV = 4;
static constexpr size_t ATTR_IDX_LAYOUT_Q = 5;
static constexpr size_t ATTR_IDX_LAYOUT_KV = 6;
static constexpr size_t ATTR_IDX_LAYOUT_OUT = 7;
static constexpr size_t ATTR_IDX_RETURN_SOFTMAX_LSE = 8;

// 输入索引
static constexpr size_t INPUT_IDX_Q = 0;
static constexpr size_t INPUT_IDX_K_CACHE = 1;
// 输出索引
static constexpr size_t OUTPUT_IDX_ATTN_OUT = 0;
static constexpr size_t OUTPUT_IDX_SOFTMAX_LSE = 1;

// MLA rope 合并语义常量：q/k_cache 最后维 = head_dim_v + ROPE_HEAD_DIM
static constexpr int64_t FLASH_MLA_WITH_KVCACHE_ROPE_HEAD_DIM = 64;

ge::graphStatus InferShapeFlashMlaWithKvcache(gert::InferShapeContext *context)
{
    OP_LOGI(context, "FlashMlaWithKvcache InferShape start.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *qShape = context->GetInputShape(INPUT_IDX_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    const gert::Shape *kCacheShape = context->GetInputShape(INPUT_IDX_K_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char *layoutQ = attrs->GetAttrPointer<char>(ATTR_IDX_LAYOUT_Q);
    const char *layoutKv = attrs->GetAttrPointer<char>(ATTR_IDX_LAYOUT_KV);
    const char *layoutOut = attrs->GetAttrPointer<char>(ATTR_IDX_LAYOUT_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutQ);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutKv);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutOut);

    auto headDimVPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_HEAD_DIM_V);
    OP_CHECK_NULL_WITH_CONTEXT(context, headDimVPtr);
    int64_t headDimV = *headDimVPtr;

    auto returnSoftmaxLsePtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_RETURN_SOFTMAX_LSE);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnSoftmaxLsePtr);
    int64_t returnSoftmaxLse = *returnSoftmaxLsePtr;

    // head_dim_v + rope(64) == 576 一致性校验（MLA 正确性关键，QA 项）
    if (headDimV <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "head_dim_v",
                                              std::to_string(headDimV).c_str(), "The value of head_dim_v must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    int64_t expectedLastDim = headDimV + FLASH_MLA_WITH_KVCACHE_ROPE_HEAD_DIM;

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

    OP_LOGI(context, "FlashMlaWithKvcache InferShape: layoutQ=%s, layoutKv=%s, layoutOut=%s, headDimV=%ld, returnLSE=%ld.",
            layoutQStr.c_str(), layoutKvStr.c_str(), layoutOutStr.c_str(), headDimV, returnSoftmaxLse);

    int64_t batchSize = 1;
    int64_t numHeadsQ = 0;
    int64_t seqLenQ = 0;
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
    } else if (layoutQStr == "TND") {
        if (qShape->GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "q",
                                                     std::to_string(qShape->GetDimNum()).c_str(),
                                                     "The shape dim of q must be 3 when layout_q is TND");
            return ge::GRAPH_FAILED;
        }
        seqLenQ = qShape->GetDim(0); // T = total tokens
        numHeadsQ = qShape->GetDim(1);
        isTND = true;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "layout_q",
                                              layoutQStr.c_str(), "The value of layout_q must be in BSND/BNSD/TND");
        return ge::GRAPH_FAILED;
    }

    // 输出布局必须与查询布局严格一致（不支持转置；q 无 NTD 形态，故 NTD 输出一律拒绝）
    if (layoutOutStr != layoutQStr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "layout_out",
                                              layoutOutStr.c_str(),
                                              "The value of layout_out must be equal to layout_q");
        return ge::GRAPH_FAILED;
    }

    // q/k_cache 最后维 == head_dim_v + 64（rope 合并语义，静态形状下强制校验）
    int64_t qLastDim = qShape->GetDim(qShape->GetDimNum() - 1);
    if (qLastDim != -1 && qLastDim != expectedLastDim) {
        std::string reason = "The last dim of q must be equal to head_dim_v + rope(64) = " +
                             std::to_string(expectedLastDim);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "q",
                                              std::to_string(qLastDim).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t kCacheLastDim = kCacheShape->GetDim(kCacheShape->GetDimNum() - 1);
    if (kCacheLastDim != -1 && kCacheLastDim != expectedLastDim) {
        std::string reason = "The last dim of k_cache must be equal to head_dim_v + rope(64) = " +
                             std::to_string(expectedLastDim);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "k_cache",
                                              std::to_string(kCacheLastDim).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    gert::Shape *attnOutShape = context->GetOutputShape(OUTPUT_IDX_ATTN_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, attnOutShape);

    // 输出最后维 = head_dim_v（q 的 576 含 rope，不能直接作为输出宽度）
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "layout_out",
                                              layoutOutStr.c_str(), "The value of layout_out must be in BSND/BNSD/TND");
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

    OP_LOGI(context, "FlashMlaWithKvcache InferShape done. attnOut dims=%zu.", attnOutShape->GetDimNum());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeFlashMlaWithKvcache(gert::InferDataTypeContext *context)
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

IMPL_OP_INFERSHAPE(FlashMlaWithKvcache)
    .InferShape(InferShapeFlashMlaWithKvcache)
    .InferDataType(InferDataTypeFlashMlaWithKvcache);

} // namespace ops
