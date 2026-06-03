/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_fused_sigmoid_gating_delta_rule_update.cpp
 * \brief
 */
#include <dlfcn.h>
#include "aclnn_fused_sigmoid_gating_delta_rule_update.h"
#include "../fused_sigmoid_gating_delta_rule_update.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
struct FusedSigmoidGatingDeltaRuleUpdateParams {
    // mandatory
    const aclTensor *a_log {nullptr};
    const aclTensor *a {nullptr};
    const aclTensor *b {nullptr};
    const aclTensor *dt_bias {nullptr};
    const aclTensor *query {nullptr};
    const aclTensor *key {nullptr};
    const aclTensor *value {nullptr};
    const aclTensor *state {nullptr};
    const aclTensor *actual_seq_lengths {nullptr};
    const aclTensor *ssm_state_indices {nullptr};
    // optional
    const aclTensor *num_accepted_tokens {nullptr};
    // attrs
    float scale {1.0f};
    float softplus_beta {1.0f};
    float softplus_threshold {20.0f};
    //output
    const aclTensor *out {nullptr};
};

// support dtype
static const std::initializer_list<op::DataType> QKV_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16,op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> GATE_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> SEQ_LENS_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> SSM_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> HEAD_PARAM_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> ACC_TO_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};

static inline bool CheckNotNull(const FusedSigmoidGatingDeltaRuleUpdateParams &params)
{
    // 必选参数
    OP_CHECK_NULL(params.a_log, return false);
    OP_CHECK_NULL(params.a, return false);
    OP_CHECK_NULL(params.b, return false);
    OP_CHECK_NULL(params.dt_bias, return false);
    OP_CHECK_NULL(params.query, return false);
    OP_CHECK_NULL(params.key, return false);
    OP_CHECK_NULL(params.value, return false);
    OP_CHECK_NULL(params.state, return false);
    OP_CHECK_NULL(params.actual_seq_lengths, return false);
    OP_CHECK_NULL(params.ssm_state_indices, return false);
    OP_CHECK_NULL(params.out, return false);

    return true;
}

static inline bool CheckDtypeVaild(const FusedSigmoidGatingDeltaRuleUpdateParams &params)
{
    // 检查必选参数数据类型
    OP_CHECK_DTYPE_NOT_SUPPORT(params.a_log, HEAD_PARAM_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.a, GATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.b, GATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.dt_bias, HEAD_PARAM_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.query, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.key, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.value, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.state, STATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.actual_seq_lengths, SEQ_LENS_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.ssm_state_indices, SSM_TYPE_SUPPORT_LIST, return false);

    if (params.num_accepted_tokens != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.num_accepted_tokens, ACC_TO_TYPE_SUPPORT_LIST, return false);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(params.out, OUT_TYPE_SUPPORT_LIST, return false);
    return true;
}

static aclnnStatus CheckParams(FusedSigmoidGatingDeltaRuleUpdateParams &params)
{
    // 检查输入参数是否在支持的数据类型范围内
    CHECK_RET(CheckDtypeVaild(params), ACLNN_ERR_PARAM_INVALID);

    OP_LOGD("FusedSigmoidGatingDeltaRuleUpdate check params success.");

    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(FusedSigmoidGatingDeltaRuleUpdateParams &params)
{
    params.a_log->SetOriginalShape(params.a_log->GetViewShape());
    params.a->SetOriginalShape(params.a->GetViewShape());
    params.b->SetOriginalShape(params.b->GetViewShape());
    params.dt_bias->SetOriginalShape(params.dt_bias->GetViewShape());
    params.query->SetOriginalShape(params.query->GetViewShape());
    params.key->SetOriginalShape(params.key->GetViewShape());
    params.value->SetOriginalShape(params.value->GetViewShape());
    params.state->SetOriginalShape(params.state->GetViewShape());
    params.actual_seq_lengths->SetOriginalShape(params.actual_seq_lengths->GetViewShape());
    params.ssm_state_indices->SetOriginalShape(params.ssm_state_indices->GetViewShape());

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnFusedSigmoidGatingDeltaRuleUpdateGetWorkspaceSize(const aclTensor *aLog, const aclTensor *a,
                                                         const aclTensor *b, const aclTensor *dtBias,
                                                         const aclTensor *query, const aclTensor *key,
                                                         const aclTensor *value, aclTensor *stateRef,
                                                         const aclTensor *actualSeqLengths,
                                                         const aclTensor *ssmStateIndices,
                                                         const aclTensor *numAcceptedTokens,
                                                         float scaleValue, float softplusBeta,
                                                         float softplusThreshold, aclTensor *out,
                                                         uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFusedSigmoidGatingDeltaRuleUpdate,
                   DFX_IN(aLog, a, b, dtBias, query, key, value, stateRef, actualSeqLengths, ssmStateIndices,
                          numAcceptedTokens, scaleValue, softplusBeta, softplusThreshold),
                   DFX_OUT(out, stateRef));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    FusedSigmoidGatingDeltaRuleUpdateParams params {aLog, a, b, dtBias, query, key, value, stateRef,
                                                    actualSeqLengths, ssmStateIndices, numAcceptedTokens,
                                                    scaleValue, softplusBeta, softplusThreshold, out};

    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    auto ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto aLog_ = l0op::Contiguous(aLog, uniqueExecutor.get());
    auto a_ = l0op::Contiguous(a, uniqueExecutor.get());
    auto b_ = l0op::Contiguous(b, uniqueExecutor.get());
    auto dtBias_ = l0op::Contiguous(dtBias, uniqueExecutor.get());
    auto query_ = l0op::Contiguous(query, uniqueExecutor.get());
    auto key_ = l0op::Contiguous(key, uniqueExecutor.get());
    auto value_ = l0op::Contiguous(value, uniqueExecutor.get());
    auto actualSeqLengths_ = l0op::Contiguous(actualSeqLengths, uniqueExecutor.get());
    auto ssmStateIndices_ = l0op::Contiguous(ssmStateIndices, uniqueExecutor.get());
    if (numAcceptedTokens != nullptr) {
        numAcceptedTokens = l0op::Contiguous(numAcceptedTokens, uniqueExecutor.get());
    }

    auto out_ = l0op::Contiguous(out, uniqueExecutor.get());

    // 调用l0接口
    auto outRet =
        l0op::FusedSigmoidGatingDeltaRuleUpdate(aLog_, a_, b_, dtBias_, query_, key_, value_, stateRef,
                                                actualSeqLengths_, ssmStateIndices_, numAcceptedTokens, scaleValue,
                                                softplusBeta, softplusThreshold, uniqueExecutor.get());
    if (outRet == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto ViewCopyResult = l0op::ViewCopy(outRet, out_, uniqueExecutor.get());
    if (ViewCopyResult == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // 获取计算过程中需要使用的workspace大小。
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedSigmoidGatingDeltaRuleUpdate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedSigmoidGatingDeltaRuleUpdate);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
