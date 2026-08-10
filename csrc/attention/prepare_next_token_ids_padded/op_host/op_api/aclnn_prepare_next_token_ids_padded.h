#ifndef ACLNN_PREPARE_NEXT_TOKEN_IDS_PADDED_H
#define ACLNN_PREPARE_NEXT_TOKEN_IDS_PADDED_H


#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif


__attribute__((visibility("default")))
aclnnStatus aclnnPrepareNextTokenIdsPaddedGetWorkspaceSize(
    const aclTensor *sampledTokenIds,
    const aclTensor *discardRequestMask,
    const aclTensor *backupNextTokenIds,
    int64_t vocabSize,
    const aclTensor *nextTokenIds,
    const aclTensor *validSampledTokensCount,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);


__attribute__((visibility("default")))
aclnnStatus aclnnPrepareNextTokenIdsPadded(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
