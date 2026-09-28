/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "metadata_checker_sparse_flash_mla.h"
#include "log/log.h"

namespace optiling {
namespace sparse_mla_checker {
namespace {
const char *Op(const CheckContext &context)
{
    return context.opName == nullptr ? "SparseMla" : context.opName;
}
} // namespace

ge::graphStatus MetadataChecker::CheckSinglePara(const CheckContext &context) const
{
    if (!context.metadata.present) {
        return ge::GRAPH_SUCCESS;
    }
    if (CheckTensorDesc(context, context.metadata, "metadata", {ge::DT_INT32}) != ge::GRAPH_SUCCESS ||
        CheckShape(context, context.metadata, "metadata", {1024}) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MetadataChecker::CheckParaExistence(const CheckContext &context) const
{
    OP_CHECK_IF(!context.metadata.present,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Op(context), "metadata", "Metadata is required in the current version"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace sparse_mla_checker
} // namespace optiling
