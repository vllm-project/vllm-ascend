/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_LIGHTNING_INDEXER_H
#define ACLNN_LIGHTNING_INDEXER_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnLightningIndexerGetWorkspaceSize calculates the workspace size based on the specific calculation process.
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLightningIndexerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    char *layoutQueryOptional,
    char *layoutKeyOptional,
    int64_t sparseCount,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    bool returnValues,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnLightningIndexer is used to perform calculations.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLightningIndexer(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_LIGHTNING_INDEXER_H
