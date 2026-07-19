/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file aclnn_dgemma_fused_router_front.cpp */
#include <dlfcn.h>
#include "aclnn_dgemma_fused_router_front.h"
#include "dgemma_fused_router_front.h"
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/contiguous.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
struct RouterParams {
    const aclTensor *x{nullptr};
    const aclTensor *scale{nullptr};
    const aclTensor *projWeight{nullptr};
    const aclTensor *normScratch{nullptr};
    const aclTensor *logitsScratch{nullptr};
    const aclTensor *perExpertScale{nullptr};
    const aclTensor *syncScratch{nullptr};
    aclTensor *topkWeights{nullptr};
    aclTensor *topkIds{nullptr};
};
static const std::initializer_list<op::DataType> IN_TYPE_SUPPORT_LIST =
    {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
static inline bool CheckNotNull(const RouterParams &p)
{
    OP_CHECK_NULL(p.x, return false);
    OP_CHECK_NULL(p.scale, return false);
    OP_CHECK_NULL(p.projWeight, return false);
    OP_CHECK_NULL(p.normScratch, return false);
    OP_CHECK_NULL(p.logitsScratch, return false);
    OP_CHECK_NULL(p.perExpertScale, return false);
    OP_CHECK_NULL(p.syncScratch, return false);
    OP_CHECK_NULL(p.topkWeights, return false);
    OP_CHECK_NULL(p.topkIds, return false);
    return true;
}
static inline bool CheckDtype(const RouterParams &p)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(p.x, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.scale, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.projWeight, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.normScratch, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.logitsScratch, {op::DataType::DT_FLOAT}, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.syncScratch, {op::DataType::DT_INT32}, return false);
    return true;
}
static aclnnStatus CheckParams(const RouterParams &p)
{
    CHECK_RET(CheckNotNull(p), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(p), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnDgemmaFusedRouterFrontGetWorkspaceSize(
    const aclTensor *x, const aclTensor *scale, const aclTensor *projWeight,
    const aclTensor *normScratch, const aclTensor *logitsScratch,
    const aclTensor *perExpertScale, const aclTensor *syncScratch,
    float epsilon, int64_t hiddenSize, int64_t numExperts, int64_t topK,
    int64_t syncBase,
    aclTensor *topkWeights, aclTensor *topkIds,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnDgemmaFusedRouterFront,
                   DFX_IN(x, scale, projWeight, normScratch, logitsScratch, perExpertScale,
                          syncScratch, epsilon, hiddenSize, numExperts, topK, syncBase),
                   DFX_OUT(topkWeights, topkIds));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    RouterParams params{x, scale, projWeight, normScratch, logitsScratch, perExpertScale, syncScratch,
                        topkWeights, topkIds};
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto xC  = l0op::Contiguous(x, uniqueExecutor.get());
    auto sC  = l0op::Contiguous(scale, uniqueExecutor.get());
    auto wC  = l0op::Contiguous(projWeight, uniqueExecutor.get());
    auto scrC = l0op::Contiguous(normScratch, uniqueExecutor.get());
    auto logitsC = l0op::Contiguous(logitsScratch, uniqueExecutor.get());
    auto scaleC = l0op::Contiguous(perExpertScale, uniqueExecutor.get());
    auto syncC = l0op::Contiguous(syncScratch, uniqueExecutor.get());
    CHECK_RET(xC && sC && wC && scrC && logitsC && scaleC && syncC, ACLNN_ERR_INNER_NULLPTR);

    aclTensor *idsTmp = nullptr;
    auto result = l0op::DgemmaFusedRouterFront(xC, sC, wC, scrC, logitsC, scaleC, syncC,
                                               epsilon, hiddenSize, numExperts, topK,
                                               syncBase,
                                               &idsTmp, uniqueExecutor.get());
    CHECK_RET(result != nullptr && idsTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto vc = l0op::ViewCopy(result, topkWeights, uniqueExecutor.get());
    CHECK_RET(vc != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto idsVc = l0op::ViewCopy(idsTmp, topkIds, uniqueExecutor.get());
    CHECK_RET(idsVc != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDgemmaFusedRouterFront(void *workspace, uint64_t workspaceSize,
                                           aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDgemmaFusedRouterFront);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
