/**
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "aclnn/aclnn_base.h"

namespace l0op {
// No-output operator: the copy itself is the effect, so instead of an output
// array the launcher-level entry returns the launch status directly — an
// empty output array can never signal an ADD_TO_LAUNCHER_LIST_AICORE
// failure, and the caller would treat a dropped copy as success.
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
    aclOpExecutor *executor);
} // namespace l0op
