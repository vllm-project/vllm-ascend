/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_PAGED_SELECT_ATTENTION_INNER_H_
#define ACLNN_PAGED_SELECT_ATTENTION_INNER_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerPagedSelectAttentionGetWorkspaceSize(
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
    const aclTensor *softmaxLse,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerPagedSelectAttention(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
