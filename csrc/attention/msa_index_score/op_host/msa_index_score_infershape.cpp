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
 * \file msa_index_score_infershape.cpp
 * \brief MsaIndexScore InferShape / InferDataType。
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

#include <cstring>

using namespace ge;

namespace ops {
namespace {
constexpr uint32_t MSA_QUERY_INDEX = 0;
constexpr uint32_t MSA_KEY_INDEX = 1;
constexpr uint32_t MSA_BLOCK_TABLE_INDEX = 2;
constexpr uint32_t MSA_ACTUAL_SEQ_KLEN_INDEX = 6;
constexpr uint32_t MSA_QUERY_DIM_NUM = 3;
constexpr uint32_t MSA_KEY_TND_DIM_NUM = 3;
constexpr uint32_t MSA_BLOCK_TABLE_DIM_NUM = 2;
constexpr uint32_t MSA_ATTR_LAYOUT_KEY = 0;
constexpr int64_t MSA_SCORE_STRIDE_ALIGN = 16;
constexpr int64_t MSA_BLOCK_SIZE = 128;

inline int64_t RoundUpTo(int64_t value, int64_t align) { return (value + align - 1) / align * align; }
} // namespace

static ge::graphStatus InferShapeMsaIndexScore(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MsaIndexScore", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter MsaIndexScore InferShape impl.");

    const gert::Shape *queryShape = context->GetInputShape(MSA_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(MSA_KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape *blockTableShape = context->GetOptionalInputShape(MSA_BLOCK_TABLE_INDEX);
    gert::Shape *scoreShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreShape);

    OP_CHECK_IF(queryShape->GetDimNum() != MSA_QUERY_DIM_NUM,
                OP_LOGE(context, "query must be TND with 3 dims, but got %zu.", queryShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    const char *layoutKeyPtr = (attrs != nullptr) ? attrs->GetStr(MSA_ATTR_LAYOUT_KEY) : nullptr;
    const char *layoutKey = (layoutKeyPtr == nullptr || layoutKeyPtr[0] == '\0') ? "BBND" : layoutKeyPtr;
    OP_CHECK_IF(std::strcmp(layoutKey, "TND") != 0 && std::strcmp(layoutKey, "BBND") != 0 &&
                    std::strcmp(layoutKey, "BNBD") != 0,
                OP_LOGE(context, "layout_key must be TND, BBND or BNBD, got %s.", layoutKey), return ge::GRAPH_FAILED);
    const bool isTnd = (std::strcmp(layoutKey, "TND") == 0);

    const int64_t totalQ = queryShape->GetDim(0);
    const int64_t numQHeads = queryShape->GetDim(1);
    int64_t maxBlocks = 0;
    if (!isTnd) {
        OP_CHECK_IF(blockTableShape == nullptr, OP_LOGE(context, "layout_key=BBND/BNBD requires block_table."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(blockTableShape->GetDimNum() != MSA_BLOCK_TABLE_DIM_NUM,
                    OP_LOGE(context, "block_table must have 2 dims, but got %zu.", blockTableShape->GetDimNum()),
                    return ge::GRAPH_FAILED);
        maxBlocks = blockTableShape->GetDim(1);
    } else {
        OP_CHECK_IF(keyShape->GetDimNum() != MSA_KEY_TND_DIM_NUM,
                    OP_LOGE(context, "layout_key=TND requires key rank 3."), return ge::GRAPH_FAILED);
        const gert::Shape *klenShape = context->GetOptionalInputShape(MSA_ACTUAL_SEQ_KLEN_INDEX);
        const gert::Tensor *klenTensor = context->GetOptionalInputTensor(MSA_ACTUAL_SEQ_KLEN_INDEX);
        const int32_t *klenData = (klenTensor != nullptr) ? klenTensor->GetData<int32_t>() : nullptr;
        if (klenData != nullptr && klenShape != nullptr && klenShape->GetDimNum() == 1 && klenShape->GetDim(0) >= 2) {
            const int64_t prefixN = klenShape->GetDim(0);
            for (int64_t i = 0; i + 1 < prefixN; ++i) {
                const int64_t kv = static_cast<int64_t>(klenData[i + 1]) - static_cast<int64_t>(klenData[i]);
                const int64_t blocks = (kv <= 0) ? 0 : ((kv + MSA_BLOCK_SIZE - 1) / MSA_BLOCK_SIZE);
                if (blocks > maxBlocks) {
                    maxBlocks = blocks;
                }
            }
        } else {
            const int64_t totalK = keyShape->GetDim(0);
            maxBlocks = (totalK + MSA_BLOCK_SIZE - 1) / MSA_BLOCK_SIZE;
        }
    }
    OP_CHECK_IF(maxBlocks <= 0, OP_LOGE(context, "maxBlocksPerSeq must be positive."), return ge::GRAPH_FAILED);

    // maxpool 已经把一个 block 内的 blockSize 个 token 归约成 1 个分数，
    // 因此输出末维是 block 数（16 对齐）而不是 kv token 数。
    scoreShape->SetDimNum(MSA_QUERY_DIM_NUM);
    scoreShape->SetDim(0, numQHeads);                                    // 0: numQHeads
    scoreShape->SetDim(1, totalQ);                                       // 1: totalQ
    scoreShape->SetDim(2, RoundUpTo(maxBlocks, MSA_SCORE_STRIDE_ALIGN)); // 2: scoreBlockStride

    OP_LOGI(context->GetNodeName(), "MsaIndexScore InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMsaIndexScore(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MsaIndexScore", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    // score 恒为 fp32：Cube 以 fp32 累加，下游 topk 直接消费 fp32。
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MsaIndexScore).InferShape(InferShapeMsaIndexScore).InferDataType(InferDataTypeMsaIndexScore);
} // namespace ops
