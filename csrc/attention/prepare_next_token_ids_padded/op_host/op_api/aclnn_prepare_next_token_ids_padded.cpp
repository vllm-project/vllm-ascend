#include "aclnn_prepare_next_token_ids_padded.h"


#ifdef __cplusplus
extern "C"{
#endif

extern aclnnStatus
aclnnInnerPrepareNextTokenIdsPaddedGetWorkspaceSize(
    const aclTensor *sampledTokenIds,
    const aclTensor *discardRequestMask,
    const aclTensor *backupNextTokenIds,
    int64_t vocabSize,
    const aclTensor *nextTokenIds,
    const aclTensor *validSampledTokensCount,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus
aclnnInnerPrepareNextTokenIdsPadded(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);





aclnnStatus aclnnPrepareNextTokenIdsPaddedGetWorkspaceSize(
    const aclTensor *sampledTokenIds,
    const aclTensor *discardRequestMask,
    const aclTensor *backupNextTokenIds,
    int64_t vocabSize,
    const aclTensor *nextTokenIds,
    const aclTensor *validSampledTokensCount,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
    {
    return
        aclnnInnerPrepareNextTokenIdsPaddedGetWorkspaceSize(
            sampledTokenIds,
            discardRequestMask,
            backupNextTokenIds,
            vocabSize,
            nextTokenIds,
            validSampledTokensCount,
            workspaceSize,
            executor);
    }

aclnnStatus aclnnPrepareNextTokenIdsPadded(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
    {
    return aclnnInnerPrepareNextTokenIdsPadded(
        workspace,
        workspaceSize,
        executor,
        stream);
    }

#ifdef __cplusplus
}
#endif

