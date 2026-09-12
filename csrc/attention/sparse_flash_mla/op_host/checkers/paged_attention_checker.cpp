/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "paged_attention_checker_sparse_flash_mla.h"
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *Op(const CheckContext &context)
{
    return context.opName == nullptr ? "SparseMla" : context.opName;
}
} // namespace

ge::graphStatus PagedAttentionChecker::CheckBlockTable(const CheckContext &context, const TensorParam &param,
                                                       const char *name) const
{
    if (!param.present) {
        return ge::GRAPH_SUCCESS;
    }
    if (CheckTensorDesc(context, param, name, {ge::DT_INT32}) != ge::GRAPH_SUCCESS ||
        CheckDimNum(context, param, name, {2U}) != ge::GRAPH_SUCCESS ||
        CheckNoEmptyDim(context, param, name) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const CheckContext &context) const
{
    if (CheckBlockTable(context, context.oriBlockTable, "ori_block_table") != ge::GRAPH_SUCCESS ||
        CheckBlockTable(context, context.cmpBlockTable, "cmp_block_table") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckParaExistence(const CheckContext &context) const
{
    if (context.kvLayout != Layout::PA_BBND) {
        OP_CHECK_IF(context.oriBlockTable.present || context.cmpBlockTable.present,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        Op(context), "ori_block_table and cmp_block_table",
                        "Block tables are only supported when layout_kv is PA_BBND"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(!context.oriBlockTable.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "ori_block_table", "Ori_block_table is required when layout_kv is PA_BBND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.cmpKv.present != context.cmpBlockTable.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "cmp_block_table",
                    "Cmp_block_table must be present exactly when PA cmp_kv is present"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.oriBlockTable.present && !context.sequsedOriKv.present && !CanOmitSequsedOriKv(context),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "seqused_ori_kv and ori_topk_length",
                    "Seqused_ori_kv is required with ori_block_table unless ORI_SPARSE or ORI_CMP_SPARSE uses "
                    "mask_mode=0 with ori_topk_length"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context.cmpBlockTable.present && !context.sequsedCmpKv.present && !CanOmitSequsedCmpKv(context),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "seqused_cmp_kv and cmp_topk_length",
                    "Seqused_cmp_kv is required with cmp_block_table unless ORI_CMP_SPARSE uses mask_mode=0 "
                    "with cmp_topk_length"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiPara(const CheckContext &context) const
{
    if (context.oriBlockTable.present) {
        OP_CHECK_IF(GetDim(context.oriBlockTable, 0) != context.bSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        Op(context), "ori_block_table", std::to_string(GetDim(context.oriBlockTable, 0)).c_str(),
                        ("The first dimension must equal batch size " + std::to_string(context.bSize)).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (context.cmpBlockTable.present) {
        OP_CHECK_IF(GetDim(context.cmpBlockTable, 0) != context.bSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        Op(context), "cmp_block_table", std::to_string(GetDim(context.cmpBlockTable, 0)).c_str(),
                        ("The first dimension must equal batch size " + std::to_string(context.bSize)).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace sparse_mla_checker
} // namespace optiling
