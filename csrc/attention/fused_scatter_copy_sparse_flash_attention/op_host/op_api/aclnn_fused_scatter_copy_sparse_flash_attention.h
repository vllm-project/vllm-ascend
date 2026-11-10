/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef ACLNN_FUSED_SCATTER_COPY_SPARSE_FLASH_ATTENTION_H_
#define ACLNN_FUSED_SCATTER_COPY_SPARSE_FLASH_ATTENTION_H_

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
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
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnFusedScatterCopySparseFlashAttention(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
