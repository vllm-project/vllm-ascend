/**
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lim_copy_descriptors.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LimCopyDescriptors);

aclnnStatus LimCopyDescriptors(
    const aclTensor *missSrcIds,
    const aclTensor *missDstSlots,
    const aclTensor *missCounts,
    const aclTensor *sourceBlockTable,
    const aclTensor *poolEntries,
    const aclTensor *hostPtrs,
    const aclTensor *devPtrs,
    const aclTensor *params,
    const aclTensor *activeGeometry,
    aclOpExecutor *executor)
{
    L0_DFX(LimCopyDescriptors, missSrcIds, missCounts);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        LimCopyDescriptors,
        OP_INPUT(missSrcIds, missDstSlots, missCounts, sourceBlockTable, poolEntries,
                 hostPtrs, devPtrs, params, activeGeometry),
        OP_OUTPUT(),
        OP_ATTR());
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ret, "ADD_TO_LAUNCHER_LIST_AICORE LimCopyDescriptors failed.");
        return ret;
    }
    (void)executor;
    return ACLNN_SUCCESS;
}
} // namespace l0op
