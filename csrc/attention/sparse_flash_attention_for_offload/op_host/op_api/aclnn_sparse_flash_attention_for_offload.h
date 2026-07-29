/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef ACLNN_SPARSE_FLASH_ATTENTION_FOR_OFFLOAD_H
#define ACLNN_SPARSE_FLASH_ATTENTION_FOR_OFFLOAD_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnSparseFlashAttentionForOffloadGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *tailInfo,
    const aclTensor *blockTable,
    const aclTensor *actualSeqLengthsQuery,
    const aclTensor *actualSeqLengthsKv,
    const aclTensor *queryRope,
    const aclTensor *keyRope,
    double scaleValue,
    int64_t sparseBlockSize,
    char *layoutQuery,
    char *layoutKv,
    int64_t sparseMode,
    const aclTensor *attentionOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnSparseFlashAttentionForOffload(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SPARSE_FLASH_ATTENTION_FOR_OFFLOAD_H
