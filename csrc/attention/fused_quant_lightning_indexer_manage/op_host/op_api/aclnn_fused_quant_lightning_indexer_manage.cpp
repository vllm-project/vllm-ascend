#include "aclnn_fused_quant_lightning_indexer_manage.h"
#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnInnerFusedQuantLightningIndexerManageGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, int64_t, int64_t, int64_t,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **);
extern aclnnStatus aclnnInnerFusedQuantLightningIndexerManage(
    void *, uint64_t, aclOpExecutor *, const aclrtStream);

aclnnStatus aclnnFusedQuantLightningIndexerManageWithStridesGetWorkspaceSize(
    const aclTensor *a0, const aclTensor *a1, const aclTensor *a2,
    const aclTensor *a3, const aclTensor *a4, const aclTensor *a5,
    const aclTensor *a6, const aclTensor *a7, const aclTensor *a8,
    const aclTensor *a9, const aclTensor *a10, const aclTensor *a11,
    const aclTensor *a12, int64_t keyStride, int64_t scaleStride,
    int64_t weightStride, const aclTensor *a13, const aclTensor *a14,
    const aclTensor *a15, const aclTensor *a16, const aclTensor *a17,
    const aclTensor *a18, const aclTensor *a19,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerFusedQuantLightningIndexerManageGetWorkspaceSize(
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,
        keyStride, scaleStride, weightStride, a13,
        a14, a15, a16, a17, a18, a19, workspaceSize, executor);
}
aclnnStatus aclnnFusedQuantLightningIndexerManageWithStrides(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnInnerFusedQuantLightningIndexerManage(
        workspace, workspaceSize, executor, stream);
}
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
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerFusedQuantLightningIndexerManageGetWorkspaceSize(
        indexWeights, queryDequantScale, query, indexKeyDequantScale,
        indexKeyCache, indexBlockTable, actualSeqLengthsQuery,
        actualSeqLengthsKey, offloadSeqLengthsKey, numCacheTokens,
        requestState, reqPoolEntries, cacheSlotsPool, 0, 0, 0, topkSrcIds,
        topkDstSlots, topkMissCounts, missSrcIds, missDstSlots, missCounts,
        cacheSlotsPoolOut, workspaceSize, executor);
}
aclnnStatus aclnnFusedQuantLightningIndexerManage(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnInnerFusedQuantLightningIndexerManage(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
