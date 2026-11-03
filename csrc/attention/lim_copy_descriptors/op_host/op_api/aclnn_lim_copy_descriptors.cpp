/**
 * SPDX-License-Identifier: Apache-2.0
 */

#include "aclnn_lim_copy_descriptors.h"
#include "lim_copy_descriptors.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnLimCopyDescriptorsGetWorkspaceSize(
    const aclTensor *missSrcIds,
    const aclTensor *missDstSlots,
    const aclTensor *missCounts,
    const aclTensor *sourceBlockTable,
    const aclTensor *poolEntries,
    const aclTensor *hostPtrs,
    const aclTensor *devPtrs,
    const aclTensor *params,
    const aclTensor *activeGeometry,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnLimCopyDescriptors,
                   DFX_IN(missSrcIds, missCounts),
                   DFX_OUT());
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    // No-output operator: the l0op entry returns the launch status directly
    // (an empty-output-array convention can never signal a launcher failure,
    // which would silently drop the miss onboarding copies).
    auto status = l0op::LimCopyDescriptors(missSrcIds, missDstSlots, missCounts,
                                           sourceBlockTable, poolEntries, hostPtrs,
                                           devPtrs, params, activeGeometry, executorPtr);
    CHECK_RET(status == ACLNN_SUCCESS, status);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLimCopyDescriptors(void *workspace, uint64_t workspaceSize,
                                     aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLimCopyDescriptors);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
