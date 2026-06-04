/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file aclnn_fused_gdn_gating.cpp
 * \brief ACLNN C-API (GetWorkspaceSize + Execute).
 */

#include <dlfcn.h>
#include "aclnn_fused_gdn_gating.h"
#include "fused_gdn_gating.h"

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

struct FusedGdnGatingParams {
    const aclTensor *aLog{nullptr};
    const aclTensor *a{nullptr};
    const aclTensor *b{nullptr};
    const aclTensor *dtBias{nullptr};
    float beta{1.0f};
    aclTensor *g{nullptr};
    aclTensor *betaOutput{nullptr};
};

static const std::initializer_list<op::DataType> AB_TYPE_SUPPORT_LIST =
    {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> FP32_TYPE_SUPPORT_LIST =
    {op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const FusedGdnGatingParams &params)
{
    OP_CHECK_NULL(params.aLog,   return false);
    OP_CHECK_NULL(params.a,      return false);
    OP_CHECK_NULL(params.b,      return false);
    OP_CHECK_NULL(params.dtBias, return false);
    OP_CHECK_NULL(params.g,          return false);
    OP_CHECK_NULL(params.betaOutput, return false);
    return true;
}

static inline bool CheckDtype(const FusedGdnGatingParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.aLog,       FP32_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.dtBias,     FP32_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.a,          AB_TYPE_SUPPORT_LIST,   return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.b,          AB_TYPE_SUPPORT_LIST,   return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.g,          FP32_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.betaOutput, AB_TYPE_SUPPORT_LIST,   return false);
    return true;
}

static aclnnStatus CheckParams(const FusedGdnGatingParams &params)
{
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(params),   ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnFusedGdnGatingGetWorkspaceSize(
    const aclTensor *aLog, const aclTensor *a, const aclTensor *b,
    const aclTensor *dtBias, float beta,
    aclTensor *g, aclTensor *betaOutput,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFusedGdnGating,
                   DFX_IN(aLog, a, b, dtBias, beta),
                   DFX_OUT(g, betaOutput));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    FusedGdnGatingParams params{aLog, a, b, dtBias, beta, g, betaOutput};
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // Bring inputs to a contiguous form that the kernel expects.
    auto aLogContig   = l0op::Contiguous(aLog,   uniqueExecutor.get());
    auto aContig      = l0op::Contiguous(a,      uniqueExecutor.get());
    auto bContig      = l0op::Contiguous(b,      uniqueExecutor.get());
    auto dtBiasContig = l0op::Contiguous(dtBias, uniqueExecutor.get());
    CHECK_RET(aLogContig   != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(aContig      != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(bContig      != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dtBiasContig != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::FusedGdnGating(aLogContig, aContig, bContig, dtBiasContig,
                                       beta, uniqueExecutor.get());
    CHECK_RET(result.g != nullptr && result.beta_output != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    // Copy kernel results into the caller-provided output tensors.
    auto vcG = l0op::ViewCopy(result.g, g, uniqueExecutor.get());
    CHECK_RET(vcG != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vcBeta = l0op::ViewCopy(result.beta_output, betaOutput, uniqueExecutor.get());
    CHECK_RET(vcBeta != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedGdnGating(void *workspace, uint64_t workspaceSize,
                                aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedGdnGating);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
