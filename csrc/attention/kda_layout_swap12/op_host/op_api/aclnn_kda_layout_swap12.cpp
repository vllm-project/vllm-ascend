/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "aclnn_kda_layout_swap12.h"
#include "kda_layout_swap12.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
aclnnStatus KdaLayoutSwapDataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

bool KdaLayoutSwapSameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    auto lhsShape = lhs->GetViewShape();
    auto rhsShape = rhs->GetViewShape();
    if (lhsShape.GetDimNum() != rhsShape.GetDimNum()) {
        return false;
    }
    for (size_t idx = 0; idx < lhsShape.GetDimNum(); ++idx) {
        if (lhsShape.GetDim(idx) != rhsShape.GetDim(idx)) {
            return false;
        }
    }
    return true;
}

aclnnStatus KdaLayoutSwapCheckParams(
    const aclTensor *x,
    const aclTensor *dependencyOptional,
    const aclTensor *yOut)
{
    CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    CHECK_COND(yOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "yOut must not be nullptr.");
    auto xShape = x->GetViewShape();
    auto yShape = yOut->GetViewShape();
    CHECK_COND(xShape.GetDimNum() >= 3, ACLNN_ERR_PARAM_INVALID, "x must have rank >= 3.");
    CHECK_COND(yShape.GetDimNum() == xShape.GetDimNum(), ACLNN_ERR_PARAM_INVALID,
               "yOut rank must match x rank.");
    if (xShape.GetDimNum() == 3) {
        CHECK_COND(yShape.GetDim(0) == xShape.GetDim(1) && yShape.GetDim(1) == xShape.GetDim(0) &&
                       yShape.GetDim(2) == xShape.GetDim(2),
                   ACLNN_ERR_PARAM_INVALID, "rank3 yOut shape must be [x.dim1, x.dim0, x.dim2].");
    } else {
        CHECK_COND(yShape.GetDim(0) == xShape.GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "yOut dim 0 must match x dim 0.");
        CHECK_COND(yShape.GetDim(1) == xShape.GetDim(2) && yShape.GetDim(2) == xShape.GetDim(1),
                   ACLNN_ERR_PARAM_INVALID, "yOut dims 1 and 2 must swap x dims 1 and 2.");
        for (size_t idx = 3; idx < xShape.GetDimNum(); ++idx) {
            CHECK_COND(yShape.GetDim(idx) == xShape.GetDim(idx), ACLNN_ERR_PARAM_INVALID,
                       "yOut tail dims must match x tail dims.");
        }
    }
    CHECK_COND(x->GetDataType() == yOut->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "x and yOut dtype must match.");
    if (dependencyOptional != nullptr) {
        CHECK_COND(KdaLayoutSwapSameShape(dependencyOptional, yOut), ACLNN_ERR_PARAM_INVALID,
                   "dependencyOptional shape must match yOut shape.");
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnKdaLayoutSwap12GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *dependencyOptional,
    const aclTensor *yOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnKdaLayoutSwap12, DFX_IN(x, dependencyOptional), DFX_OUT(yOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(KdaLayoutSwapCheckParams(x, dependencyOptional, yOut) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaLayoutSwapDataContiguous(x, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaLayoutSwapDataContiguous(dependencyOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto result = l0op::KdaLayoutSwap12(x, dependencyOptional, yOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKdaLayoutSwap12(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnKdaLayoutSwap12);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
