#ifndef ACLNN_REBATCH_AND_UPDATE_H_
#define ACLNN_REBATCH_AND_UPDATE_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
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
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnRebatchAndUpdate(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_REBATCH_AND_UPDATE_H_
