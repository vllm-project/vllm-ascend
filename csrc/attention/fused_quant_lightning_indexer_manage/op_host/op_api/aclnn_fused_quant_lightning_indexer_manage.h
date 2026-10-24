#ifndef ACLNN_FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_H
#define ACLNN_FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_H
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default")))
aclnnStatus aclnnFusedQuantLightningIndexerManageWithStridesGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, int64_t, int64_t, int64_t,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **);
__attribute__((visibility("default")))
aclnnStatus aclnnFusedQuantLightningIndexerManageWithStrides(
    void *, uint64_t, aclOpExecutor *, const aclrtStream);
__attribute__((visibility("default")))
aclnnStatus aclnnFusedQuantLightningIndexerManageGetWorkspaceSize(
    const aclTensor *indexWeights, const aclTensor *queryDequantScale,
    const aclTensor *query, const aclTensor *indexKeyDequantScale,
    const aclTensor *indexKeyCache, const aclTensor *indexBlockTable,
    const aclTensor *actualSeqLengthsQuery, const aclTensor *actualSeqLengthsKey,
    const aclTensor *offloadSeqLengthsKey, const aclTensor *numCacheTokens,
    const aclTensor *requestState, const aclTensor *reqPoolEntries,
    const aclTensor *cacheSlotsPool, const aclTensor *topkSrcIds,
    const aclTensor *topkDstSlots, const aclTensor *topkMissCounts,
    const aclTensor *missSrcIds, const aclTensor *missDstSlots,
    const aclTensor *missCounts, const aclTensor *cacheSlotsPoolOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);
__attribute__((visibility("default")))
aclnnStatus aclnnFusedQuantLightningIndexerManage(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif
