#ifndef ACLNN_MOE_DISPATCH_NORMAL_H_
#define ACLNN_MOE_DISPATCH_NORMAL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnMoeDispatchNormalGetWorkspaceSize(const aclTensor *x,
    const aclTensor *topkIdx, const aclTensor *sendOffset, const aclTensor *sendTokenIdx, const aclTensor *recvOffset,
    const aclTensor *recvCount, char *groupEp, int64_t epWorldSize, int64_t epRankId, char *groupTpOptional,
    int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum, int64_t quantMode, int64_t globalBs,
    const aclTensor *recvX, const aclTensor *recvXScales, const aclTensor *assistInfoForCombine,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnMoeDispatchNormal(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif