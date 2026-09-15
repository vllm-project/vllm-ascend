/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sparse_compression_checker.h"
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *Op(const CheckContext &context) { return context.opName == nullptr ? "SparseMla" : context.opName; }
} // namespace

ge::graphStatus SparseCompressionChecker::CheckIndex(const CheckContext &context, const TensorParam &param,
                                                     const char *name) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    const size_t dimNum = context.qLayout == Layout::BSND ? 4U : 3U;
    if (CheckTensorDesc(context, param, name, {ge::DT_INT32}) != ge::GRAPH_SUCCESS ||
        CheckDimNum(context, param, name, {dimNum}) != ge::GRAPH_SUCCESS ||
        CheckNoEmptyDim(context, param, name) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckTopkLength(const CheckContext &context, const TensorParam &param,
                                                          const char *name) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    const size_t dimNum = context.qLayout == Layout::BSND ? 3U : 2U;
    if (CheckTensorDesc(context, param, name, {ge::DT_INT32}) != ge::GRAPH_SUCCESS ||
        CheckDimNum(context, param, name, {dimNum}) != ge::GRAPH_SUCCESS ||
        CheckNoEmptyDim(context, param, name) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckSinglePara(const CheckContext &context) const
{
    OP_CHECK_IF(
        context.cmpRatio < 0 || context.cmpRatio > 128 || (context.cmpKv.present && context.cmpRatio == 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(Op(context), "cmp_ratio", std::to_string(context.cmpRatio).c_str(),
                                              "Cmp_ratio must be 0 or 1 when cmp_kv is absent, or in range [1, 128] "
                                              "when cmp_kv is present"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context.topkValueMode != 1,
        OP_LOGE_FOR_INVALID_VALUE(Op(context), "topk_value_mode", std::to_string(context.topkValueMode).c_str(), "1"),
        return ge::GRAPH_FAILED);
    if (CheckIndex(context, context.oriSparseIndices, "ori_sparse_indices") != ge::GRAPH_SUCCESS ||
        CheckIndex(context, context.cmpSparseIndices, "cmp_sparse_indices") != ge::GRAPH_SUCCESS ||
        CheckTopkLength(context, context.oriTopkLength, "ori_topk_length") != ge::GRAPH_SUCCESS ||
        CheckTopkLength(context, context.cmpTopkLength, "cmp_topk_length") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckParaExistence(const CheckContext &context) const
{
    if (!context.cmpKv.present) {
        OP_CHECK_IF(context.cmpSparseIndices.present,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Op(context), "cmp_sparse_indices",
                                                             "Cmp_sparse_indices requires cmp_kv"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            context.cmpTopkLength.present,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Op(context), "cmp_topk_length", "Cmp_topk_length requires cmp_kv"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            context.cmpResidualKv.present,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Op(context), "cmp_residual_kv", "Cmp_residual_kv requires cmp_kv"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            context.cmpBlockTable.present,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Op(context), "cmp_block_table", "Cmp_block_table requires cmp_kv"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(context.cuSeqlensCmpKv.present,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Op(context), "cu_seqlens_cmp_kv",
                                                             "Cu_seqlens_cmp_kv requires cmp_kv"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            context.sequsedCmpKv.present,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Op(context), "seqused_cmp_kv", "Seqused_cmp_kv requires cmp_kv"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            context.cmpRatio != 0 && context.cmpRatio != 1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(Op(context), "cmp_ratio", std::to_string(context.cmpRatio).c_str(),
                                                  "Cmp_ratio must be 0 or 1 when cmp_kv is absent"),
            return ge::GRAPH_FAILED);
    }

    const bool needResidual = context.cmpKv.present && context.cmpMaskMode == 3 && context.cmpRatio != 1;
    OP_CHECK_IF(needResidual && !context.cmpResidualKv.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "cmp_residual_kv",
                    "Cmp_residual_kv is required when cmp_mask_mode is 3 and cmp_ratio is not 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.oriSparseIndices.present && context.oriMaskMode == 0 && !context.oriTopkLength.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "ori_topk_length",
                    "Ori_topk_length is required when ori_sparse_indices is present and ori_mask_mode is 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.cmpSparseIndices.present && context.cmpMaskMode == 0 && !context.cmpTopkLength.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "cmp_topk_length",
                    "Cmp_topk_length is required when cmp_sparse_indices is present and cmp_mask_mode is 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckFeature(const CheckContext &context) const
{
    OP_CHECK_IF(context.cmpSparseIndices.present && !context.cmpKv.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Op(context), "cmp_sparse_indices",
                                                         "Cmp_sparse_indices requires cmp_kv"),
                return ge::GRAPH_FAILED);
    if (context.cmpSparseIndices.present) {
        OP_CHECK_IF(context.cmpTopk <= 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        Op(context), "cmp_sparse_indices", std::to_string(context.cmpTopk).c_str(),
                        "The last dimension must be greater than 0 when cmp_sparse_indices is present"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckIndexShape(const CheckContext &context, const TensorParam &param,
                                                          const char *name) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    if (context.qLayout == Layout::BSND) {
        const std::string actualShape = "(" + std::to_string(GetDim(param, 0)) + ", " +
                                        std::to_string(GetDim(param, 1)) + ", " + std::to_string(GetDim(param, 2)) +
                                        ", " + std::to_string(GetDim(param, 3)) + ")";
        OP_CHECK_IF(GetDim(param, 0) != GetDim(context.q, 0) || GetDim(param, 1) != GetDim(context.q, 1) ||
                        GetDim(param, 2) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE(Op(context), name, actualShape.c_str(), "(b, q_s, kv_n, topk)"),
                    return ge::GRAPH_FAILED);
    } else {
        const std::string actualShape = "(" + std::to_string(GetDim(param, 0)) + ", " +
                                        std::to_string(GetDim(param, 1)) + ", " + std::to_string(GetDim(param, 2)) +
                                        ")";
        OP_CHECK_IF(GetDim(param, 0) != GetDim(context.q, 0) || GetDim(param, 1) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE(Op(context), name, actualShape.c_str(), "(q_t, kv_n, topk)"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckTopkLengthShape(const CheckContext &context, const TensorParam &param,
                                                               const char *name) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    if (context.qLayout == Layout::BSND) {
        const std::string actualShape = "(" + std::to_string(GetDim(param, 0)) + ", " +
                                        std::to_string(GetDim(param, 1)) + ", " + std::to_string(GetDim(param, 2)) +
                                        ")";
        OP_CHECK_IF(GetDim(param, 0) != GetDim(context.q, 0) || GetDim(param, 1) != GetDim(context.q, 1) ||
                        GetDim(param, 2) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE(Op(context), name, actualShape.c_str(), "(b, q_s, kv_n)"),
                    return ge::GRAPH_FAILED);
    } else {
        const std::string actualShape =
            "(" + std::to_string(GetDim(param, 0)) + ", " + std::to_string(GetDim(param, 1)) + ")";
        OP_CHECK_IF(GetDim(param, 0) != GetDim(context.q, 0) || GetDim(param, 1) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE(Op(context), name, actualShape.c_str(), "(q_t, kv_n)"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseCompressionChecker::CheckMultiPara(const CheckContext &context) const
{
    if (CheckIndexShape(context, context.oriSparseIndices, "ori_sparse_indices") != ge::GRAPH_SUCCESS ||
        CheckIndexShape(context, context.cmpSparseIndices, "cmp_sparse_indices") != ge::GRAPH_SUCCESS ||
        CheckTopkLengthShape(context, context.oriTopkLength, "ori_topk_length") != ge::GRAPH_SUCCESS ||
        CheckTopkLengthShape(context, context.cmpTopkLength, "cmp_topk_length") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace sparse_mla_checker
} // namespace optiling
