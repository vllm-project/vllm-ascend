/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_attn_res_fwd.cpp
 * \brief AttnResFwd ACLNN 两段式接口
 */
#include "aclnn_attn_res_fwd.h"
#include "attn_res_fwd.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
struct AttnResFwdParams {
    const aclTensor *prefixSum{nullptr};
    const aclTensor *blockResidual{nullptr};
    const aclTensor *projWeight{nullptr};
    const aclTensor *normWeight{nullptr};
    double normEps{1e-5};
    bool needBackward{false};
    const aclTensor *hiddenStates{nullptr};
    const aclTensor *invRms{nullptr};
    const aclTensor *probs{nullptr};
};

static const std::initializer_list<op::DataType> BF16_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> FLOAT_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const AttnResFwdParams &params)
{
    OP_CHECK_NULL(params.prefixSum, return false);
    OP_CHECK_NULL(params.blockResidual, return false);
    OP_CHECK_NULL(params.projWeight, return false);
    OP_CHECK_NULL(params.normWeight, return false);
    OP_CHECK_NULL(params.hiddenStates, return false);
    if (params.needBackward) {
        OP_CHECK_NULL(params.invRms, return false);
        OP_CHECK_NULL(params.probs, return false);
    }
    return true;
}

static inline bool CheckDtypeValid(const AttnResFwdParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.prefixSum, BF16_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.blockResidual, BF16_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.projWeight, BF16_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.normWeight, BF16_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.hiddenStates, BF16_TYPE_SUPPORT_LIST, return false);
    if (params.needBackward) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.invRms, FLOAT_TYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params.probs, FLOAT_TYPE_SUPPORT_LIST, return false);
    }
    return true;
}

static inline bool CheckSavedShape(const AttnResFwdParams &params)
{
    if (!params.needBackward) {
        return true;
    }
    const auto &prefixShape = params.prefixSum->GetViewShape();
    const auto &blockShape = params.blockResidual->GetViewShape();
    if (prefixShape.GetDimNum() != 2 || blockShape.GetDimNum() != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "prefix_sum must be 2D and block_residual must be 3D");
        return false;
    }
    const int64_t T = prefixShape.GetDim(0);
    const int64_t B = blockShape.GetDim(1) + 1;
    const auto &invShape = params.invRms->GetViewShape();
    const auto &probsShape = params.probs->GetViewShape();
    if (invShape.GetDimNum() != 2 || probsShape.GetDimNum() != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inv_rms/probs must be 2D [T,B]");
        return false;
    }
    if (invShape.GetDim(0) != T || invShape.GetDim(1) != B || probsShape.GetDim(0) != T ||
        probsShape.GetDim(1) != B) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inv_rms/probs shape must be [%ld,%ld]", T, B);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(AttnResFwdParams &params)
{
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSavedShape(params), ACLNN_ERR_PARAM_INVALID);
    if (params.normEps <= 0.0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "normEps must be > 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("AttnResFwd check params success.");
    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(AttnResFwdParams &params)
{
    params.prefixSum->SetOriginalShape(params.prefixSum->GetViewShape());
    params.blockResidual->SetOriginalShape(params.blockResidual->GetViewShape());
    params.projWeight->SetOriginalShape(params.projWeight->GetViewShape());
    params.normWeight->SetOriginalShape(params.normWeight->GetViewShape());
    params.hiddenStates->SetOriginalShape(params.hiddenStates->GetViewShape());
    if (params.needBackward) {
        params.invRms->SetOriginalShape(params.invRms->GetViewShape());
        params.probs->SetOriginalShape(params.probs->GetViewShape());
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnAttnResFwdGetWorkspaceSize(const aclTensor *prefixSum, const aclTensor *blockResidual,
                                            const aclTensor *projWeight, const aclTensor *normWeight, double normEps,
                                            bool needBackward, aclTensor *hiddenStates, aclTensor *invRms,
                                            aclTensor *probs, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnAttnResFwd,
                   DFX_IN(prefixSum, blockResidual, projWeight, normWeight, normEps, needBackward),
                   DFX_OUT(hiddenStates, invRms, probs));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    AttnResFwdParams params{prefixSum, blockResidual, projWeight, normWeight, normEps, needBackward,
                            hiddenStates, invRms, probs};
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (hiddenStates->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto prefixSum_ = l0op::Contiguous(prefixSum, uniqueExecutor.get());
    CHECK_RET(prefixSum_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto blockResidual_ = l0op::Contiguous(blockResidual, uniqueExecutor.get());
    CHECK_RET(blockResidual_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto projWeight_ = l0op::Contiguous(projWeight, uniqueExecutor.get());
    CHECK_RET(projWeight_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto normWeight_ = l0op::Contiguous(normWeight, uniqueExecutor.get());
    CHECK_RET(normWeight_ != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outRet =
        l0op::AttnResFwd(prefixSum_, blockResidual_, projWeight_, normWeight_, normEps, needBackward,
                         uniqueExecutor.get());
    if (outRet[0] == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyOut = l0op::ViewCopy(outRet[0], hiddenStates, uniqueExecutor.get());
    if (viewCopyOut == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (needBackward) {
        auto viewCopyInv = l0op::ViewCopy(outRet[1], invRms, uniqueExecutor.get());
        CHECK_RET(viewCopyInv != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyProbs = l0op::ViewCopy(outRet[2], probs, uniqueExecutor.get());
        CHECK_RET(viewCopyProbs != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAttnResFwd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAttnResFwd);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
