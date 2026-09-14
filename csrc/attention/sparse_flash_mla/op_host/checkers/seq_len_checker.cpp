/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "seq_len_checker_sparse_flash_mla.h"
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *Op(const CheckContext &context)
{
    return context.opName == nullptr ? "SparseMla" : context.opName;
}
} // namespace

ge::graphStatus SeqLenChecker::CheckLengthTensor(const CheckContext &context, const TensorParam &param,
                                                 const char *name) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    if (CheckTensorDesc(context, param, name, {ge::DT_INT32}) != ge::GRAPH_SUCCESS ||
        CheckDimNum(context, param, name, {1U}) != ge::GRAPH_SUCCESS ||
        CheckNoEmptyDim(context, param, name) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckSinglePara(const CheckContext &context) const
{
    if (CheckLengthTensor(context, context.cuSeqlensQ, "cu_seqlens_q") != ge::GRAPH_SUCCESS ||
        CheckLengthTensor(context, context.cuSeqlensOriKv, "cu_seqlens_ori_kv") != ge::GRAPH_SUCCESS ||
        CheckLengthTensor(context, context.cuSeqlensCmpKv, "cu_seqlens_cmp_kv") != ge::GRAPH_SUCCESS ||
        CheckLengthTensor(context, context.sequsedQ, "seqused_q") != ge::GRAPH_SUCCESS ||
        CheckLengthTensor(context, context.sequsedOriKv, "seqused_ori_kv") != ge::GRAPH_SUCCESS ||
        CheckLengthTensor(context, context.sequsedCmpKv, "seqused_cmp_kv") != ge::GRAPH_SUCCESS ||
        CheckLengthTensor(context, context.cmpResidualKv, "cmp_residual_kv") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckParaExistence(const CheckContext &context) const
{
    if (context.qLayout == Layout::TND) {
        OP_CHECK_IF(!context.cuSeqlensQ.present,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        Op(context), "cu_seqlens_q", "Cu_seqlens_q is required when layout_q is TND"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(context.cuSeqlensQ.present,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        Op(context), "cu_seqlens_q", "Cu_seqlens_q is only supported when layout_q is TND"),
                    return ge::GRAPH_FAILED);
    }

    if (context.kvLayout == Layout::TND) {
        OP_CHECK_IF(!context.cuSeqlensOriKv.present,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        Op(context), "cu_seqlens_ori_kv",
                        "Cu_seqlens_ori_kv is required when layout_kv is TND"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context.cmpKv.present && !context.cuSeqlensCmpKv.present,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        Op(context), "cu_seqlens_cmp_kv",
                        "Cu_seqlens_cmp_kv is required when TND cmp_kv is present"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(context.cuSeqlensOriKv.present || context.cuSeqlensCmpKv.present,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        Op(context), "cu_seqlens_ori_kv and cu_seqlens_cmp_kv",
                        "KV cu_seqlens inputs are only supported when layout_kv is TND"),
                    return ge::GRAPH_FAILED);
    }

    if (context.kvLayout == Layout::PA_BBND) {
        OP_CHECK_IF(!context.sequsedOriKv.present && !CanOmitSequsedOriKv(context),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        Op(context), "seqused_ori_kv and ori_topk_length",
                        "Seqused_ori_kv is required for Paged Attention unless ORI_SPARSE or ORI_CMP_SPARSE "
                        "uses mask_mode=0 with ori_topk_length"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context.cmpKv.present && !context.sequsedCmpKv.present && !CanOmitSequsedCmpKv(context),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        Op(context), "seqused_cmp_kv and cmp_topk_length",
                        "Seqused_cmp_kv is required for Paged Attention cmp_kv unless ORI_CMP_SPARSE uses "
                        "mask_mode=0 with cmp_topk_length"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckLength(const CheckContext &context, const TensorParam &param, const char *name,
                                           int64_t expected) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(GetDim(param, 0) != expected,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    Op(context), name, std::to_string(GetDim(param, 0)).c_str(),
                    ("The length must be " + std::to_string(expected)).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SeqLenChecker::CheckMultiPara(const CheckContext &context) const
{
    const int64_t batch = context.bSize;
    OP_CHECK_IF(batch <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(Op(context), "batch size", std::to_string(batch).c_str(),
                                                      "Batch size must be greater than 0"),
                return ge::GRAPH_FAILED);
    if (CheckLength(context, context.cuSeqlensQ, "cu_seqlens_q", batch + 1) != ge::GRAPH_SUCCESS ||
        CheckLength(context, context.cuSeqlensOriKv, "cu_seqlens_ori_kv", batch + 1) != ge::GRAPH_SUCCESS ||
        CheckLength(context, context.cuSeqlensCmpKv, "cu_seqlens_cmp_kv", batch + 1) != ge::GRAPH_SUCCESS ||
        CheckLength(context, context.sequsedQ, "seqused_q", batch) != ge::GRAPH_SUCCESS ||
        CheckLength(context, context.sequsedOriKv, "seqused_ori_kv", batch) != ge::GRAPH_SUCCESS ||
        CheckLength(context, context.sequsedCmpKv, "seqused_cmp_kv", batch) != ge::GRAPH_SUCCESS ||
        CheckLength(context, context.cmpResidualKv, "cmp_residual_kv", batch) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace sparse_mla_checker
} // namespace optiling
