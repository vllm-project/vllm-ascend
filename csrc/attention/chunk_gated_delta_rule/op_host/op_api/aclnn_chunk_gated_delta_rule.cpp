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
 * \file aclnn_chunk_gated_delta_rule.cpp
 * \brief ACLNN C-API (GetWorkspaceSize + Execute) for ChunkGatedDeltaRule.
 */

#include <dlfcn.h>
#include "aclnn_chunk_gated_delta_rule.h"
#include "chunk_gated_delta_rule.h"

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

struct ChunkGatedDeltaRuleParams {
    const aclTensor *query{nullptr};
    const aclTensor *key{nullptr};
    const aclTensor *value{nullptr};
    const aclTensor *beta{nullptr};
    const aclTensor *initialState{nullptr};
    const aclTensor *actualSeqLengths{nullptr};
    const aclTensor *g{nullptr};
    float scale{1.0f};
    aclTensor *out{nullptr};
    aclTensor *finalState{nullptr};
};

static const std::initializer_list<op::DataType> QKV_BETA_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> SEQ_LENS_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> G_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> FINAL_STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const ChunkGatedDeltaRuleParams &params)
{
    OP_CHECK_NULL(params.query, return false);
    OP_CHECK_NULL(params.key, return false);
    OP_CHECK_NULL(params.value, return false);
    OP_CHECK_NULL(params.beta, return false);
    OP_CHECK_NULL(params.initialState, return false);
    OP_CHECK_NULL(params.actualSeqLengths, return false);
    OP_CHECK_NULL(params.out, return false);
    OP_CHECK_NULL(params.finalState, return false);
    return true;
}

static inline bool CheckDtypeValid(const ChunkGatedDeltaRuleParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.query, QKV_BETA_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.key, QKV_BETA_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.value, QKV_BETA_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.beta, QKV_BETA_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.initialState, STATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.actualSeqLengths, SEQ_LENS_TYPE_SUPPORT_LIST, return false);
    if (params.g != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.g, G_TYPE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(params.out, OUT_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.finalState, FINAL_STATE_TYPE_SUPPORT_LIST, return false);
    return true;
}

static aclnnStatus CheckParams(const ChunkGatedDeltaRuleParams &params)
{
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("ChunkGatedDeltaRule check params success.");
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnChunkGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *beta, const aclTensor *initialState,
    const aclTensor *actualSeqLengths, const aclTensor *g,
    float scaleValue, aclTensor *out, aclTensor *finalState,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRule,
                   DFX_IN(query, key, value, beta, initialState, actualSeqLengths, g, scaleValue),
                   DFX_OUT(out, finalState));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    ChunkGatedDeltaRuleParams params{query, key, value, beta, initialState,
                                     actualSeqLengths, g, scaleValue, out, finalState};

    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto query_ = l0op::Contiguous(query, uniqueExecutor.get());
    auto key_ = l0op::Contiguous(key, uniqueExecutor.get());
    auto value_ = l0op::Contiguous(value, uniqueExecutor.get());
    auto beta_ = l0op::Contiguous(beta, uniqueExecutor.get());
    auto initialState_ = l0op::Contiguous(initialState, uniqueExecutor.get());
    auto actualSeqLengths_ = l0op::Contiguous(actualSeqLengths, uniqueExecutor.get());
    const aclTensor *g_ = nullptr;
    if (g != nullptr) {
        g_ = l0op::Contiguous(g, uniqueExecutor.get());
    }

    auto out_ = l0op::Contiguous(out, uniqueExecutor.get());
    auto finalState_ = l0op::Contiguous(finalState, uniqueExecutor.get());

    auto result = l0op::ChunkGatedDeltaRule(query_, key_, value_, beta_, initialState_,
                                            actualSeqLengths_, g_, scaleValue,
                                            uniqueExecutor.get());
    if (result.out == nullptr || result.final_state == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto vcOut = l0op::ViewCopy(result.out, out_, uniqueExecutor.get());
    if (vcOut == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto vcFinalState = l0op::ViewCopy(result.final_state, finalState_, uniqueExecutor.get());
    if (vcFinalState == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRule(void *workspace, uint64_t workspaceSize,
                                     aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRule);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
