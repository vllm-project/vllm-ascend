/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common_checker_sparse_flash_mla.h"
#include <cmath>
#include <vector>
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;

namespace {
const char *Op(const CheckContext &context)
{
    return context.opName == nullptr ? "SparseMla" : context.opName;
}
} // namespace

ge::graphStatus CommonChecker::CheckQuery(const CheckContext &context) const
{
    if (!context.q.present) {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (context.variant == OperatorVariant::SPARSE) {
        status = CheckTensorDesc(context, context.q, "q", {ge::DT_FLOAT16, ge::DT_BF16});
    } else if (context.variant == OperatorVariant::MIXED_QUANT) {
        status = CheckTensorDesc(context, context.q, "q", {ge::DT_BF16});
    } else {
        status = CheckTensorDesc(context, context.q, "q", {ge::DT_HIFLOAT8});
    }
    if (status != ge::GRAPH_SUCCESS || CheckNoEmptyDim(context, context.q, "q") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const size_t expectedDimNum = context.qLayout == Layout::BSND ? 4U : 3U;
    return CheckDimNum(context, context.q, "q", {expectedDimNum});
}

ge::graphStatus CommonChecker::CheckKv(const CheckContext &context, const TensorParam &kv, const char *name) const
{
    if (!kv.present) {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (context.variant == OperatorVariant::SPARSE) {
        status = CheckTensorDesc(context, kv, name, {ge::DT_FLOAT16, ge::DT_BF16});
    } else if (context.variant == OperatorVariant::MIXED_QUANT) {
        status = CheckTensorDesc(context, kv, name, {ge::DT_FLOAT8_E4M3FN});
    } else {
        status = CheckTensorDesc(context, kv, name, {ge::DT_HIFLOAT8});
    }
    if (status != ge::GRAPH_SUCCESS || CheckNoEmptyDim(context, kv, name) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const size_t expectedDimNum = context.kvLayout == Layout::TND ? 3U : 4U;
    return CheckDimNum(context, kv, name, {expectedDimNum});
}

ge::graphStatus CommonChecker::CheckOutput(const CheckContext &context) const
{
    if (!context.attentionOut.present) {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus status = ge::GRAPH_FAILED;
    if (context.variant == OperatorVariant::SPARSE) {
        status = CheckTensorDesc(context, context.attentionOut, "attention_out", {ge::DT_FLOAT16, ge::DT_BF16});
    } else {
        status = CheckTensorDesc(context, context.attentionOut, "attention_out", {ge::DT_BF16});
    }
    if (status != ge::GRAPH_SUCCESS ||
        CheckNoEmptyDim(context, context.attentionOut, "attention_out") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const size_t expectedDimNum = context.qLayout == Layout::BSND ? 4U : 3U;
    return CheckDimNum(context, context.attentionOut, "attention_out", {expectedDimNum});
}

ge::graphStatus CommonChecker::CheckSinglePara(const CheckContext &context) const
{
    OP_CHECK_IF(
        context.qLayout != Layout::BSND && context.qLayout != Layout::TND,
        OP_LOGE_FOR_INVALID_VALUE(Op(context), "layout_q",
                                  std::to_string(static_cast<uint32_t>(context.qLayout)).c_str(), "BSND or TND"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context.kvLayout != Layout::BSND && context.kvLayout != Layout::TND && context.kvLayout != Layout::PA_BBND,
        OP_LOGE_FOR_INVALID_VALUE(Op(context), "layout_kv",
                                  std::to_string(static_cast<uint32_t>(context.kvLayout)).c_str(),
                                  "BSND, TND or PA_BBND"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !std::isfinite(context.softmaxScale),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            Op(context), "softmax_scale", std::to_string(context.softmaxScale).c_str(), "Softmax_scale must be finite"),
        return ge::GRAPH_FAILED);
    if (CheckQuery(context) != ge::GRAPH_SUCCESS || CheckKv(context, context.oriKv, "ori_kv") != ge::GRAPH_SUCCESS ||
        CheckKv(context, context.cmpKv, "cmp_kv") != ge::GRAPH_SUCCESS || CheckOutput(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckParaExistence(const CheckContext &context) const
{
    OP_CHECK_IF(!context.q.present, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Op(context), "q", "Q is required"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !context.oriKv.present,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Op(context), "ori_kv", "Ori_kv is required in all supported modes"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!context.attentionOut.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Op(context), "attention_out", "Attention_out is required"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckFeature(const CheckContext &context) const
{
    OP_CHECK_IF(
        context.kvLayout != Layout::PA_BBND && context.qLayout != context.kvLayout,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(Op(context), "layout_q and layout_kv",
                                               (std::to_string(static_cast<uint32_t>(context.qLayout)) + " and " +
                                                std::to_string(static_cast<uint32_t>(context.kvLayout)))
                                                   .c_str(),
                                               "Non-PA layout_q and layout_kv must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckQueryAxes(const CheckContext &context) const
{
    const int64_t qSeq = GetDim(context.q, 0);
    int64_t qHeads = -1;
    int64_t qDim = -1;
    if (context.qLayout == Layout::BSND) {
        OP_CHECK_IF(GetDim(context.q, 0) <= 0 || GetDim(context.q, 1) <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(Op(context), "q",
                                                          ("Batch=" + std::to_string(GetDim(context.q, 0)) +
                                                           ", sequence=" + std::to_string(GetDim(context.q, 1)))
                                                              .c_str(),
                                                          "Batch and sequence dimensions must be greater than 0"),
                    return ge::GRAPH_FAILED);
        qHeads = GetDim(context.q, DIM_IDX_TWO);
        qDim = GetDim(context.q, DIM_IDX_THREE);
    } else {
        OP_CHECK_IF(qSeq <= 0, OP_LOGE_FOR_INVALID_SHAPESIZE(Op(context), "q_t", std::to_string(qSeq).c_str(), "> 0"),
                    return ge::GRAPH_FAILED);
        qHeads = GetDim(context.q, 1);
        qDim = GetDim(context.q, DIM_IDX_TWO);
    }
    OP_CHECK_IF(qHeads <= 0 || qHeads > 128, // 128：最大支持的注意力头数
                OP_LOGE_FOR_INVALID_VALUE(Op(context), "q_n", std::to_string(qHeads).c_str(), "[1, 128]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qDim != 512, // 512：每个注意力头的固定维度大小
                OP_LOGE_FOR_INVALID_VALUE(Op(context), "q head dimension", std::to_string(qDim).c_str(), "512"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKvAxes(const CheckContext &context, const TensorParam &kv, const char *name) const
{
    if (!kv.present) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t numHeads = -1;
    int64_t headDim = -1;
    if (context.kvLayout == Layout::TND) {
        numHeads = GetDim(kv, 1);
        headDim = GetDim(kv, DIM_IDX_TWO);
    } else {
        numHeads = GetDim(kv, DIM_IDX_TWO);
        headDim = GetDim(kv, DIM_IDX_THREE);
    }
    OP_CHECK_IF(numHeads != 1,
                OP_LOGE_FOR_INVALID_VALUE(Op(context), (std::string(name) + " kv_n").c_str(),
                                          std::to_string(numHeads).c_str(), "1"),
                return ge::GRAPH_FAILED);
    int64_t expectedDim = 512; // 512：预期维度大小
    if (context.variant == OperatorVariant::MIXED_QUANT) {
        expectedDim = context.quantMode == 1 ? 608 : 584; // 608，584：混合量化模式下根据量化模式选择不同维度
    }
    OP_CHECK_IF(headDim != expectedDim,
                OP_LOGE_FOR_INVALID_VALUE(Op(context), (std::string(name) + " head dimension").c_str(),
                                          std::to_string(headDim).c_str(), std::to_string(expectedDim).c_str()),
                return ge::GRAPH_FAILED);
    if (context.kvLayout == Layout::PA_BBND) {
        const int64_t blockSize = GetDim(kv, 1);
        OP_CHECK_IF(blockSize <= 0 || blockSize > 1024,
                    OP_LOGE_FOR_INVALID_VALUE(Op(context), (std::string(name) + " block size").c_str(),
                                              std::to_string(blockSize).c_str(), "[1, 1024]"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiPara(const CheckContext &context) const
{
    if (CheckSameShape(context, context.q, "q", context.attentionOut, "attention_out") != ge::GRAPH_SUCCESS ||
        CheckQueryAxes(context) != ge::GRAPH_SUCCESS ||
        CheckKvAxes(context, context.oriKv, "ori_kv") != ge::GRAPH_SUCCESS ||
        CheckKvAxes(context, context.cmpKv, "cmp_kv") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(context.variant == OperatorVariant::SPARSE &&
                    context.q.desc->GetDataType() != context.attentionOut.desc->GetDataType(),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    Op(context), "q and attention_out",
                    (std::to_string(static_cast<int32_t>(context.q.desc->GetDataType())) + " and " +
                     std::to_string(static_cast<int32_t>(context.attentionOut.desc->GetDataType())))
                        .c_str(),
                    "Q and attention_out dtype must be the same"),
                return ge::GRAPH_FAILED);
    if (context.variant == OperatorVariant::SPARSE) {
        OP_CHECK_IF(context.q.desc->GetDataType() != context.oriKv.desc->GetDataType(),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        Op(context), "q and ori_kv",
                        (std::to_string(static_cast<int32_t>(context.q.desc->GetDataType())) + " and " +
                         std::to_string(static_cast<int32_t>(context.oriKv.desc->GetDataType())))
                            .c_str(),
                        "Q and ori_kv dtype must be the same"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context.cmpKv.present && context.q.desc->GetDataType() != context.cmpKv.desc->GetDataType(),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        Op(context), "q and cmp_kv",
                        (std::to_string(static_cast<int32_t>(context.q.desc->GetDataType())) + " and " +
                         std::to_string(static_cast<int32_t>(context.cmpKv.desc->GetDataType())))
                            .c_str(),
                        "Q and cmp_kv dtype must be the same"),
                    return ge::GRAPH_FAILED);
    }
    if (context.kvLayout == Layout::BSND) {
        const int64_t qBatch = GetDim(context.q, 0);
        OP_CHECK_IF(GetDim(context.oriKv, 0) != qBatch,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        Op(context), "q and ori_kv batch",
                        (std::to_string(qBatch) + " and " + std::to_string(GetDim(context.oriKv, 0))).c_str(),
                        "Ori_kv batch must equal q batch"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context.cmpKv.present && GetDim(context.cmpKv, 0) != qBatch,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        Op(context), "q and cmp_kv batch",
                        (std::to_string(qBatch) + " and " + std::to_string(GetDim(context.cmpKv, 0))).c_str(),
                        "Cmp_kv batch must equal q batch"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace sparse_mla_checker
} // namespace optiling
