/**
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OP_API_INC_ACLNN_LIM_COPY_DESCRIPTORS_LINEAR_H
#define OP_API_INC_ACLNN_LIM_COPY_DESCRIPTORS_LINEAR_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnLimCopyDescriptorsLinearGetWorkspaceSize(
    const aclTensor *missSrcIds,
    const aclTensor *missDstSlots,
    const aclTensor *missCounts,
    const aclTensor *sourceBlockTable,
    const aclTensor *hostPtrs,
    const aclTensor *devPtrs,
    const aclTensor *params,
    const aclTensor *activeGeometry,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnLimCopyDescriptorsLinear(void *workspace, uint64_t workspaceSize,
                                     aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
