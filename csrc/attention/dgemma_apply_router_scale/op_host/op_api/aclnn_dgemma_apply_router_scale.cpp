/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "aclnn_dgemma_apply_router_scale.h"
#include "dgemma_apply_router_scale.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/contiguous.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
static const std::initializer_list<op::DataType> WEIGHT_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> ID_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> SCALE_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
}

aclnnStatus aclnnDgemmaApplyRouterScaleGetWorkspaceSize(
    const aclTensor *weights, const aclTensor *ids, const aclTensor *scale,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnDgemmaApplyRouterScale, DFX_IN(weights, ids, scale), DFX_OUT(out));
    OP_CHECK_NULL(weights, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(ids, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(scale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_DTYPE_NOT_SUPPORT(weights, WEIGHT_TYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(ids, ID_TYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(scale, SCALE_TYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto weightsC = l0op::Contiguous(weights, uniqueExecutor.get());
    CHECK_RET(weightsC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto idsC = l0op::Contiguous(ids, uniqueExecutor.get());
    CHECK_RET(idsC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto scaleC = l0op::Contiguous(scale, uniqueExecutor.get());
    CHECK_RET(scaleC != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::DgemmaApplyRouterScale(weightsC, idsC, scaleC, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vc = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(vc != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDgemmaApplyRouterScale(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDgemmaApplyRouterScale);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
