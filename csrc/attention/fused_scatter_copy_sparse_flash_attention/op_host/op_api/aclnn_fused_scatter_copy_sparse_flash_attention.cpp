/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "aclnn_fused_scatter_copy_sparse_flash_attention.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerFusedScatterCopySparseFlashAttentionGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, double, int64_t, char *, char *,
    int64_t, const aclTensor *, uint64_t *, aclOpExecutor **);
extern aclnnStatus aclnnInnerFusedScatterCopySparseFlashAttention(
    void *, uint64_t, aclOpExecutor *, const aclrtStream);

aclnnStatus aclnnFusedScatterCopySparseFlashAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *sparseIndices, const aclTensor *cacheTokens,
    const aclTensor *hbmBlockTable, const aclTensor *actualSeqLengthsQuery,
    const aclTensor *actualSeqLengthsKv, const aclTensor *queryRope,
    const aclTensor *hbmKeyRope, const aclTensor *dramKeyRope,
    const aclTensor *dramKvCache, const aclTensor *dramBlockTable,
    const aclTensor *topkSourceIds, const aclTensor *topkMissCounts,
    const aclTensor *missSourceIds, const aclTensor *missDstSlots,
    const aclTensor *missCounts, double scaleValue, int64_t sparseBlockSize,
    char *layoutQuery, char *layoutKv, int64_t sparseMode,
    const aclTensor *attentionOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerFusedScatterCopySparseFlashAttentionGetWorkspaceSize(
        query, key, value, sparseIndices, cacheTokens, hbmBlockTable,
        actualSeqLengthsQuery, actualSeqLengthsKv, queryRope, hbmKeyRope,
        dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds,
        topkMissCounts, missSourceIds, missDstSlots, missCounts, scaleValue,
        sparseBlockSize, layoutQuery, layoutKv, sparseMode, attentionOut,
        workspaceSize, executor);
}

aclnnStatus aclnnFusedScatterCopySparseFlashAttention(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnInnerFusedScatterCopySparseFlashAttention(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
