/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file aclnn_dgemma_fused_norm_rope.cpp
 *  \brief ACLNN C-API (GetWorkspaceSize + Execute). */
#include <dlfcn.h>
#include "aclnn_dgemma_fused_norm_rope.h"
#include "dgemma_fused_norm_rope.h"
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
struct DgemmaParams {
    const aclTensor *q{nullptr};
    const aclTensor *k{nullptr};
    const aclTensor *v{nullptr};
    const aclTensor *qWeight{nullptr};
    const aclTensor *kWeight{nullptr};
    const aclTensor *cos{nullptr};
    const aclTensor *sin{nullptr};
    aclTensor *qOut{nullptr};
    aclTensor *kOut{nullptr};
    aclTensor *vOut{nullptr};
};
static const std::initializer_list<op::DataType> QKV_TYPE_SUPPORT_LIST =
    {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> CS_TYPE_SUPPORT_LIST =
    {op::DataType::DT_FLOAT};
static inline bool CheckNotNull(const DgemmaParams &p)
{
    OP_CHECK_NULL(p.q, return false);
    OP_CHECK_NULL(p.k, return false);
    OP_CHECK_NULL(p.v, return false);
    OP_CHECK_NULL(p.qWeight, return false);
    OP_CHECK_NULL(p.kWeight, return false);
    OP_CHECK_NULL(p.cos, return false);
    OP_CHECK_NULL(p.sin, return false);
    OP_CHECK_NULL(p.qOut, return false);
    OP_CHECK_NULL(p.kOut, return false);
    OP_CHECK_NULL(p.vOut, return false);
    return true;
}
static inline bool CheckDtype(const DgemmaParams &p)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(p.q, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.k, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.v, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.qWeight, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.kWeight, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.cos, CS_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.sin, CS_TYPE_SUPPORT_LIST, return false);
    return true;
}
static aclnnStatus CheckParams(const DgemmaParams &p)
{
    CHECK_RET(CheckNotNull(p), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(p), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnDgemmaFusedNormRopeGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *qWeight, const aclTensor *kWeight,
    const aclTensor *cos, const aclTensor *sin,
    float epsilon, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    aclTensor *qOut, aclTensor *kOut, aclTensor *vOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnDgemmaFusedNormRope,
                   DFX_IN(q, k, v, qWeight, kWeight, cos, sin,
                          epsilon, numQHeads, numKvHeads, headDim),
                   DFX_OUT(qOut, kOut, vOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    DgemmaParams params{q, k, v, qWeight, kWeight, cos, sin, qOut, kOut, vOut};
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto qC  = l0op::Contiguous(q, uniqueExecutor.get());
    auto kC  = l0op::Contiguous(k, uniqueExecutor.get());
    auto vC  = l0op::Contiguous(v, uniqueExecutor.get());
    auto qwC = l0op::Contiguous(qWeight, uniqueExecutor.get());
    auto kwC = l0op::Contiguous(kWeight, uniqueExecutor.get());
    auto cosC = l0op::Contiguous(cos, uniqueExecutor.get());
    auto sinC = l0op::Contiguous(sin, uniqueExecutor.get());
    CHECK_RET(qC && kC && vC && qwC && kwC && cosC && sinC, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::DgemmaFusedNormRope(qC, kC, vC, qwC, kwC, cosC, sinC,
                                            epsilon, numQHeads, numKvHeads, headDim,
                                            uniqueExecutor.get());
    CHECK_RET(result.q_out && result.k_out && result.v_out, ACLNN_ERR_INNER_NULLPTR);

    auto vcQ = l0op::ViewCopy(result.q_out, qOut, uniqueExecutor.get());
    CHECK_RET(vcQ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vcK = l0op::ViewCopy(result.k_out, kOut, uniqueExecutor.get());
    CHECK_RET(vcK != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vcV = l0op::ViewCopy(result.v_out, vOut, uniqueExecutor.get());
    CHECK_RET(vcV != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDgemmaFusedNormRope(void *workspace, uint64_t workspaceSize,
                                     aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDgemmaFusedNormRope);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
