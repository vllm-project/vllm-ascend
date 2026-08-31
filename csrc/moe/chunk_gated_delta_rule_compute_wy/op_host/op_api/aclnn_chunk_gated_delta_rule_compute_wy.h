#ifndef OP_API_INC_ACLNN_CHUNK_GATED_DELTA_RULE_COMPUTE_WY_H
#define OP_API_INC_ACLNN_CHUNK_GATED_DELTA_RULE_COMPUTE_WY_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleComputeWyGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    int64_t chunkSize,
    const aclTensor *qKernelOut,
    const aclTensor *kKernelOut,
    const aclTensor *wKernelOut,
    const aclTensor *uKernelOut,
    const aclTensor *gKernelOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleComputeWy(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
