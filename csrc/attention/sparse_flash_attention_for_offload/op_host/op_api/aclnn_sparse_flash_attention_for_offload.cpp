/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "aclnn_sparse_flash_attention_for_offload.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerSparseFlashAttentionForOffloadGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *sparseIndices, const aclTensor *tailInfo, const aclTensor *blockTable,
    const aclTensor *actualSeqLengthsQuery, const aclTensor *actualSeqLengthsKv,
    const aclTensor *queryRope, const aclTensor *keyRope, double scaleValue,
    int64_t sparseBlockSize, char *layoutQuery, char *layoutKv, int64_t sparseMode,
    const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerSparseFlashAttentionForOffload(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnSparseFlashAttentionForOffloadGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *sparseIndices, const aclTensor *tailInfo, const aclTensor *blockTable,
    const aclTensor *actualSeqLengthsQuery, const aclTensor *actualSeqLengthsKv,
    const aclTensor *queryRope, const aclTensor *keyRope, double scaleValue,
    int64_t sparseBlockSize, char *layoutQuery, char *layoutKv, int64_t sparseMode,
    const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerSparseFlashAttentionForOffloadGetWorkspaceSize(
        query, key, value, sparseIndices, tailInfo, blockTable, actualSeqLengthsQuery,
        actualSeqLengthsKv, queryRope, keyRope, scaleValue, sparseBlockSize, layoutQuery,
        layoutKv, sparseMode, attentionOut, workspaceSize, executor);
}

aclnnStatus aclnnSparseFlashAttentionForOffload(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerSparseFlashAttentionForOffload(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
