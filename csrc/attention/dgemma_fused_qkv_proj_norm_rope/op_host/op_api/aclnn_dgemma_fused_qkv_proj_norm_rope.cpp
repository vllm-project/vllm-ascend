/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file aclnn_dgemma_fused_qkv_proj_norm_rope.cpp */
#include <dlfcn.h>
#include "aclnn_dgemma_fused_qkv_proj_norm_rope.h"
#include "dgemma_fused_qkv_proj_norm_rope.h"
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
struct MixParams {
    const aclTensor *hidden{nullptr};
    const aclTensor *wqkv{nullptr};
    const aclTensor *qWeight{nullptr};
    const aclTensor *kWeight{nullptr};
    const aclTensor *cos{nullptr};
    const aclTensor *sin{nullptr};
    aclTensor *qkvScratch{nullptr};
    aclTensor *qkvScratchOut{nullptr};
    aclTensor *qOut{nullptr};
    aclTensor *kOut{nullptr};
    aclTensor *vOut{nullptr};
};
static const std::initializer_list<op::DataType> IN_TYPE_SUPPORT_LIST =
    {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> CS_TYPE_SUPPORT_LIST =
    {op::DataType::DT_FLOAT};
static inline bool CheckNotNull(const MixParams &p)
{
    OP_CHECK_NULL(p.hidden, return false);
    OP_CHECK_NULL(p.wqkv, return false);
    OP_CHECK_NULL(p.qWeight, return false);
    OP_CHECK_NULL(p.kWeight, return false);
    OP_CHECK_NULL(p.cos, return false);
    OP_CHECK_NULL(p.sin, return false);
    OP_CHECK_NULL(p.qkvScratch, return false);
    OP_CHECK_NULL(p.qkvScratchOut, return false);
    OP_CHECK_NULL(p.qOut, return false);
    OP_CHECK_NULL(p.kOut, return false);
    OP_CHECK_NULL(p.vOut, return false);
    return true;
}
static inline bool CheckDtype(const MixParams &p)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(p.hidden, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.wqkv, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.qWeight, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.kWeight, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.cos, CS_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.sin, CS_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.qkvScratch, IN_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(p.qkvScratchOut, IN_TYPE_SUPPORT_LIST, return false);
    return true;
}
static aclnnStatus CheckParams(const MixParams &p)
{
    CHECK_RET(CheckNotNull(p), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(p), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnDgemmaFusedQkvProjNormRopeGetWorkspaceSize(
    const aclTensor *hidden, const aclTensor *wqkv,
    const aclTensor *qWeight, const aclTensor *kWeight,
    const aclTensor *cos, const aclTensor *sin, aclTensor *qkvScratch,
    float epsilon, int64_t numQHeads, int64_t numKvHeads, int64_t headDim, int64_t hiddenSize,
    int64_t syncBase,
    aclTensor *qkvScratchOut, aclTensor *qOut, aclTensor *kOut, aclTensor *vOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnDgemmaFusedQkvProjNormRope,
                   DFX_IN(hidden, wqkv, qWeight, kWeight, cos, sin, qkvScratch,
                          epsilon, numQHeads, numKvHeads, headDim, hiddenSize, syncBase),
                   DFX_OUT(qkvScratchOut, qOut, kOut, vOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    MixParams params{hidden, wqkv, qWeight, kWeight, cos, sin, qkvScratch, qkvScratchOut, qOut, kOut, vOut};
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto hC  = l0op::Contiguous(hidden, uniqueExecutor.get());
    auto wC  = l0op::Contiguous(wqkv, uniqueExecutor.get());
    auto qwC = l0op::Contiguous(qWeight, uniqueExecutor.get());
    auto kwC = l0op::Contiguous(kWeight, uniqueExecutor.get());
    auto cosC = l0op::Contiguous(cos, uniqueExecutor.get());
    auto sinC = l0op::Contiguous(sin, uniqueExecutor.get());
    qkvScratch->SetStorageShape(qkvScratch->GetViewShape());
    qkvScratchOut->SetStorageShape(qkvScratchOut->GetViewShape());
    qOut->SetStorageShape(qOut->GetViewShape());
    kOut->SetStorageShape(kOut->GetViewShape());
    vOut->SetStorageShape(vOut->GetViewShape());
    CHECK_RET(hC && wC && qwC && kwC && cosC && sinC, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::DgemmaFusedQkvProjNormRope(hC, wC, qwC, kwC, cosC, sinC,
                                                   qkvScratch, qkvScratchOut,
                                                   qOut, kOut, vOut,
                                                   epsilon, numQHeads, numKvHeads, headDim, hiddenSize,
                                                   syncBase,
                                                   uniqueExecutor.get());
    CHECK_RET(result.qkv_scratch_out && result.q_out && result.k_out && result.v_out, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDgemmaFusedQkvProjNormRope(void *workspace, uint64_t workspaceSize,
                                            aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDgemmaFusedQkvProjNormRope);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
