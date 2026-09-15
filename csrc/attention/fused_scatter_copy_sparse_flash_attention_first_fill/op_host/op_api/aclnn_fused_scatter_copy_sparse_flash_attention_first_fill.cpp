/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "aclnn_fused_scatter_copy_sparse_flash_attention_first_fill.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerFusedScatterCopySparseFlashAttentionFirstFillGetWorkspaceSize(
    aclTensor *, aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
extern aclnnStatus aclnnInnerFusedScatterCopySparseFlashAttentionFirstFill(
    void *, uint64_t, aclOpExecutor *, const aclrtStream);

aclnnStatus aclnnFusedScatterCopySparseFlashAttentionFirstFillGetWorkspaceSize(
    aclTensor *hbmKRope, aclTensor *hbmKvCache,
    const aclTensor *dramKRope, const aclTensor *dramKvCache,
    const aclTensor *hbmBlockTable, const aclTensor *dramBlockTable,
    const aclTensor *missSourceIds, const aclTensor *missDstSlots,
    const aclTensor *missCounts, const aclTensor *numCacheTokens,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerFusedScatterCopySparseFlashAttentionFirstFillGetWorkspaceSize(
        hbmKRope, hbmKvCache, dramKRope, dramKvCache, hbmBlockTable,
        dramBlockTable, missSourceIds, missDstSlots, missCounts,
        numCacheTokens, workspaceSize, executor);
}

aclnnStatus aclnnFusedScatterCopySparseFlashAttentionFirstFill(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnInnerFusedScatterCopySparseFlashAttentionFirstFill(
        workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
