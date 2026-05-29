/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "aclnn_paged_select_attention.h"
#include "aclnn_paged_select_attention_inner.h"
#include "opdev/op_errno.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {
const aclTensor *CreateEmptySoftmaxLsePlaceholder()
{
    static thread_local std::vector<int64_t> shape = {0};
    static thread_local int64_t addr = 0xff;
    return aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(),
                           0, ACL_FORMAT_ND, shape.data(), shape.size(), static_cast<void *>(&addr));
}
} // namespace

aclnnStatus aclnnPagedSelectAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclIntArray *actualSeqLengths,
    const aclIntArray *actualSeqLengthsKv,
    const aclTensor *blockTable,
    const aclTensor *selectedKvIndices,
    int64_t numHeads,
    double scaleValue,
    int64_t numKeyValueHeads,
    int64_t blockSize,
    const aclTensor *attentionOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    const aclTensor *softmaxLsePlaceholder = CreateEmptySoftmaxLsePlaceholder();
    if (softmaxLsePlaceholder == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    aclnnStatus ret = aclnnInnerPagedSelectAttentionGetWorkspaceSize(
        query, key, value,
        actualSeqLengths, actualSeqLengthsKv,
        blockTable,
        selectedKvIndices,
        numHeads, scaleValue, numKeyValueHeads, blockSize,
        attentionOut, softmaxLsePlaceholder, workspaceSize, executor);
    aclDestroyTensor(const_cast<aclTensor *>(softmaxLsePlaceholder));
    return ret;
}

aclnnStatus aclnnPagedSelectAttention(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnInnerPagedSelectAttention(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
