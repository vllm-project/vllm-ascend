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
 * \brief
 */

#include "aclnn_chunk_gated_delta_rule.h"
#include "chunk_gated_delta_rule.h"

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
constexpr uint64_t INITIAL_STATE_DIMS_NUM = 4;

struct ChunkGatedDeltaRuleParams {
    // 必选输入
    const aclTensor *query{nullptr};
    const aclTensor *key{nullptr};
    const aclTensor *value{nullptr};
    const aclTensor *beta{nullptr};
    const aclTensor *initialState{nullptr};
    const aclTensor *actualSeqLengths{nullptr};
    // 可选输入
    const aclTensor *g{nullptr};
    // 常量输入
    float scale{1.0f};
    // 输出
    const aclTensor *out{nullptr};
    const aclTensor *finalState{nullptr};
};

// 校验相关：支持的数据类型
static const std::initializer_list<op::DataType> QKV_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> BETA_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> SEQ_LENS_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> G_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};

// 校验函数：做指针非空校验，gOptional 为可选输入，不做非空校验
static inline bool CheckNotNull(const ChunkGatedDeltaRuleParams &params)
{
    OP_CHECK_NULL(params.query, return false);
    OP_CHECK_NULL(params.key, return false);
    OP_CHECK_NULL(params.value, return false);
    OP_CHECK_NULL(params.initialState, return false);
    OP_CHECK_NULL(params.beta, return false);
    OP_CHECK_NULL(params.actualSeqLengths, return false);
    OP_CHECK_NULL(params.out, return false);
    OP_CHECK_NULL(params.finalState, return false);

    return true;
}

// 校验函数：做 dtype 校验
static inline bool CheckDtypeValid(const ChunkGatedDeltaRuleParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.query, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.key, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.value, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.initialState, STATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.beta, BETA_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.actualSeqLengths, SEQ_LENS_TYPE_SUPPORT_LIST, return false);

    // gOptional 仅在非空时校验 dtype
    if (params.g != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.g, G_TYPE_SUPPORT_LIST, return false);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(params.out, OUT_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.finalState, STATE_TYPE_SUPPORT_LIST, return false);

    return true;
}

// 校验函数：这里校验 dtype，shape/rank 约束由 tiling 侧保证
static aclnnStatus CheckParams(ChunkGatedDeltaRuleParams &params)
{
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("ChunkGatedDeltaRule check params success.");

    return ACLNN_SUCCESS;
}

// 预处理：同步 view 的原始 shape
static aclnnStatus PreProcess(ChunkGatedDeltaRuleParams &params)
{
    params.query->SetOriginalShape(params.query->GetViewShape());
    params.key->SetOriginalShape(params.key->GetViewShape());
    params.value->SetOriginalShape(params.value->GetViewShape());
    params.beta->SetOriginalShape(params.beta->GetViewShape());
    params.initialState->SetOriginalShape(params.initialState->GetViewShape());
    params.actualSeqLengths->SetOriginalShape(params.actualSeqLengths->GetViewShape());
    if (params.g != nullptr) {
        params.g->SetOriginalShape(params.g->GetViewShape());
    }

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnChunkGatedDeltaRuleGetWorkspaceSize(const aclTensor *query, const aclTensor *key,
                                                     const aclTensor *value, const aclTensor *beta,
                                                     const aclTensor *initialState, const aclTensor *actualSeqLengths,
                                                     const aclTensor *gOptional, float scaleValue, const aclTensor *out,
                                                     const aclTensor *finalState, uint64_t *workspaceSize,
                                                     aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRule,
                   DFX_IN(query, key, value, beta, initialState, actualSeqLengths, gOptional, scaleValue),
                   DFX_OUT(out, finalState));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    ChunkGatedDeltaRuleParams params{query,     key,        value, beta,      initialState, actualSeqLengths,
                                     gOptional, scaleValue, out,   finalState};
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 空输出时直接返回，避免无效执行
    if (out->IsEmpty() && finalState->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto query_ = l0op::Contiguous(query, uniqueExecutor.get());
    CHECK_RET(query_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto key_ = l0op::Contiguous(key, uniqueExecutor.get());
    CHECK_RET(key_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto value_ = l0op::Contiguous(value, uniqueExecutor.get());
    CHECK_RET(value_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto beta_ = l0op::Contiguous(beta, uniqueExecutor.get());
    CHECK_RET(beta_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 仅在Ascend950支持initialState 0或1轴非连续时传入非连续tensor，使用CreateView设置stride信息
    int64_t *stateShapeDims = nullptr;
    uint64_t stateShapeDimsNum = 0;
    int64_t *stateStrideDims = nullptr;
    uint64_t stateStrideDimsNum = 0;
    ret = aclGetViewShape(initialState, &stateShapeDims, &stateShapeDimsNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = aclGetViewStrides(initialState, &stateStrideDims, &stateStrideDimsNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(stateShapeDimsNum == INITIAL_STATE_DIMS_NUM && stateStrideDimsNum == INITIAL_STATE_DIMS_NUM,
        ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto initialState_ = initialState;

    const char *socName = aclrtGetSocName();
    CHECK_RET(socName != nullptr, ACLNN_ERR_INNER_NULLPTR);
    bool isAscend950 = std::strstr(socName, "Ascend950") != nullptr;
    if (isAscend950 &&
        stateStrideDims[stateStrideDimsNum - 1] == 1 &&
        stateStrideDims[stateStrideDimsNum - 2] == stateShapeDims[stateShapeDimsNum - 1]) {
        initialState_ = uniqueExecutor.get()->CreateView(
            initialState,
            initialState->GetViewShape(),
            initialState->GetStorageShape(),
            initialState->GetViewStrides(),
            initialState->GetViewOffset());
    } else {
        initialState_ = l0op::Contiguous(initialState, uniqueExecutor.get());
    }
    CHECK_RET(initialState_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    auto actualSeqLengths_ = l0op::Contiguous(actualSeqLengths, uniqueExecutor.get());
    CHECK_RET(actualSeqLengths_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // gOptional 为可选输入，非空时才做 contiguous
    if (gOptional != nullptr) {
        gOptional = l0op::Contiguous(gOptional, uniqueExecutor.get());
        CHECK_RET(gOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用 L0 接口，得到两个输出
    auto outRet = l0op::ChunkGatedDeltaRule(query_, key_, value_, beta_, initialState_, actualSeqLengths_, gOptional,
                                            scaleValue, uniqueExecutor.get());
    if (outRet[0] == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (outRet[1] == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto outRet0 = outRet[0];
    auto outRet1 = outRet[1];

    // 将 L0 输出回写到用户输出 tensor
    // 输出 tensor 不做 Contiguous：ViewCopy 可直接写回非连续输出 tensor，
    // 对输出在调用L0接口前调用 Contiguous 会将数据写入 workspace 而非用户 tensor， 会导致报错
    auto ViewCopyOut = l0op::ViewCopy(outRet0, out, uniqueExecutor.get());
    auto ViewCopyFinalState = l0op::ViewCopy(outRet1, finalState, uniqueExecutor.get());
    if (ViewCopyOut == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (ViewCopyFinalState == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // 获取计算所需 workspace 大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRule(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRule);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
