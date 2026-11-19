/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "softmax_lse_checker_sparse_flash_mla.h"
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *Op(const CheckContext &context)
{
    return context.opName == nullptr ? "SparseMla" : context.opName;
}
} // namespace

ge::graphStatus SoftmaxLseChecker::CheckSinglePara(const CheckContext &context) const
{
    if (!context.returnSoftmaxLse || !context.softmaxLse.present) {
        return ge::GRAPH_SUCCESS;
    }
    return CheckTensorDesc(context, context.softmaxLse, "softmax_lse", {ge::DT_FLOAT});
}

ge::graphStatus SoftmaxLseChecker::CheckMultiPara(const CheckContext &context) const
{
    if (!context.returnSoftmaxLse || !context.softmaxLse.present) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(context.qNumHeads % context.kvNumHeads != 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    Op(context), "qNumHeads and kvNumHeads",
                    (std::to_string(context.qNumHeads) + ", " + std::to_string(context.kvNumHeads)).c_str(),
                    "QNumHeads must be divisible by kvNumHeads for softmax_lse"),
                return ge::GRAPH_FAILED);
    const int64_t group = context.qNumHeads / context.kvNumHeads;
    if (context.qLayout == Layout::BSND) {
        return CheckShape(context, context.softmaxLse, "softmax_lse",
                          {context.bSize, context.kvNumHeads, context.qSeqSize, group});
    }
    return CheckShape(context, context.softmaxLse, "softmax_lse",
                      {context.kvNumHeads, context.qTotalSize, group});
}

} // namespace sparse_mla_checker
} // namespace optiling
