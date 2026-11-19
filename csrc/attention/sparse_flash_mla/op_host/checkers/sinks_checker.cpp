/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sinks_checker_sparse_flash_mla.h"
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *Op(const CheckContext &context)
{
    return context.opName == nullptr ? "SparseMla" : context.opName;
}
} // namespace

ge::graphStatus SinksChecker::CheckSinglePara(const CheckContext &context) const
{
    if (!context.sinks.present) {
        return ge::GRAPH_SUCCESS;
    }
    if (CheckTensorDesc(context, context.sinks, "sinks", {ge::DT_FLOAT}) != ge::GRAPH_SUCCESS ||
        CheckDimNum(context, context.sinks, "sinks", {1U}) != ge::GRAPH_SUCCESS ||
        CheckNoEmptyDim(context, context.sinks, "sinks") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SinksChecker::CheckParaExistence(const CheckContext &context) const
{
    OP_CHECK_IF(!context.sinks.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "sinks", "Sinks is required in the current version"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SinksChecker::CheckMultiPara(const CheckContext &context) const
{
    OP_CHECK_IF(GetDim(context.sinks, 0) != context.qNumHeads,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    Op(context), "sinks", std::to_string(GetDim(context.sinks, 0)).c_str(),
                    ("The length must equal q_n " + std::to_string(context.qNumHeads)).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace sparse_mla_checker
} // namespace optiling
