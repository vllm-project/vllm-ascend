#include "aclnn_rebatch_and_update.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerRebatchAndUpdateGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    int64_t, int64_t, int64_t, int64_t,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, uint64_t *, aclOpExecutor **);

extern aclnnStatus aclnnInnerRebatchAndUpdate(
    void *, uint64_t, aclOpExecutor *, aclrtStream);

aclnnStatus aclnnRebatchAndUpdateGetWorkspaceSize(
    const aclTensor *batchBeforeDispatch,
    const aclTensor *residualsBeforeDispatch,
    const aclTensor *cutIdx,
    const aclTensor *prefixIndex,
    const aclTensor *discardLabel,
    const aclTensor *discardTokens,
    const aclTensor *discardMoeReqIdTensor,
    const aclTensor *reqIdList,
    const aclTensor *discardResiduals,
    const aclTensor *runningTopk,
    const aclTensor *discardTopk,
    const aclTensor *runningTopkWeight,
    const aclTensor *discardTopkWeight,
    const aclTensor *moeCntInput,
    const aclTensor *actualBs,
    const aclTensor *historyBs,
    const aclTensor *runningBs,
    const aclTensor *originalBs,
    const aclTensor *positions,
    const aclTensor *slotMapping,
    const aclTensor *blockTable,
    const aclTensor *seqLens,
    const aclTensor *discardPositions,
    const aclTensor *discardSlotMapping,
    const aclTensor *discardContextLens,
    const aclTensor *discardBlockTables,
    const aclTensor *discardAttnCnt,
    const aclTensor *discardReqIds,
    int64_t isCombineRebatch,
    int64_t maxBs,
    int64_t k,
    int64_t layerIdx,
    const aclTensor *rebatchResiduals,
    const aclTensor *outPositions,
    const aclTensor *outSlots,
    const aclTensor *outSeqLens,
    const aclTensor *outBlockTable,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerRebatchAndUpdateGetWorkspaceSize(
        batchBeforeDispatch, residualsBeforeDispatch, cutIdx, prefixIndex, discardLabel,
        discardTokens, discardMoeReqIdTensor, reqIdList, discardResiduals,
        runningTopk, discardTopk, runningTopkWeight, discardTopkWeight,
        moeCntInput, actualBs, historyBs, runningBs, originalBs, positions,
        slotMapping, blockTable, seqLens, discardPositions, discardSlotMapping,
        discardContextLens, discardBlockTables, discardAttnCnt, discardReqIds,
        isCombineRebatch, maxBs, k, layerIdx, rebatchResiduals, outPositions,
        outSlots, outSeqLens, outBlockTable, workspaceSize, executor);
}

aclnnStatus aclnnRebatchAndUpdate(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return aclnnInnerRebatchAndUpdate(
        workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
