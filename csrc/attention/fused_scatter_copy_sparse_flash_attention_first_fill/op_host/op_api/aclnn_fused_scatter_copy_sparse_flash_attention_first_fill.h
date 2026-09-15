/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef ACLNN_FUSED_SCATTER_COPY_SPARSE_FLASH_ATTENTION_FIRST_FILL_H_
#define ACLNN_FUSED_SCATTER_COPY_SPARSE_FLASH_ATTENTION_FIRST_FILL_H_

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnFusedScatterCopySparseFlashAttentionFirstFillGetWorkspaceSize(
    aclTensor *hbmKRope, aclTensor *hbmKvCache,
    const aclTensor *dramKRope, const aclTensor *dramKvCache,
    const aclTensor *hbmBlockTable, const aclTensor *dramBlockTable,
    const aclTensor *missSourceIds, const aclTensor *missDstSlots,
    const aclTensor *missCounts, const aclTensor *numCacheTokens,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnFusedScatterCopySparseFlashAttentionFirstFill(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
