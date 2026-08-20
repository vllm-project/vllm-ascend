/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_dense_attention_score.h"

#include <acl/acl.h>
#include "aclnn_kernels/contiguous.h"
#include "dense_attention_score.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static aclnnStatus MakeDenseInputsContiguous(
    const aclTensor *&query,
    const aclTensor *&key,
    const aclTensor *&value,
    const aclTensor *&blockTable,
    const aclTensor *&actualSeqLengths,
    const aclTensor *&actualSeqLengthsKv,
    const aclTensor *&qDequantScale,
    const aclTensor *&kDequantScale,
    const aclTensor *&vDequantScale,
    aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    blockTable = l0op::Contiguous(blockTable, executor);
    CHECK_RET(blockTable != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    actualSeqLengths = l0op::Contiguous(actualSeqLengths, executor);
    CHECK_RET(actualSeqLengths != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    actualSeqLengthsKv = l0op::Contiguous(actualSeqLengthsKv, executor);
    CHECK_RET(actualSeqLengthsKv != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    qDequantScale = l0op::Contiguous(qDequantScale, executor);
    CHECK_RET(qDequantScale != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    kDequantScale = l0op::Contiguous(kDequantScale, executor);
    CHECK_RET(kDequantScale != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    vDequantScale = l0op::Contiguous(vDequantScale, executor);
    CHECK_RET(vDequantScale != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnDenseAttentionScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *blockTable,
    const aclTensor *actualSeqLengths,
    const aclTensor *actualSeqLengthsKv,
    const aclTensor *qDequantScale,
    const aclTensor *kDequantScale,
    const aclTensor *vDequantScale,
    int64_t numKeyValueHeads,
    double scaleValue,
    int64_t blockSize,
    int64_t innerPrecise,
    char *inputLayout,
    aclTensor *attentionOut,
    aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(blockTable != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualSeqLengths != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(actualSeqLengthsKv != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(qDequantScale != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(kDequantScale != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(vDequantScale != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(inputLayout != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    L2_DFX_PHASE_1(aclnnDenseAttentionScore,
                   DFX_IN(query, key, value, blockTable,
                          actualSeqLengths, actualSeqLengthsKv,
                          qDequantScale, kDequantScale, vDequantScale,
                          numKeyValueHeads, scaleValue, blockSize,
                          innerPrecise, inputLayout),
                   DFX_OUT(attentionOut, softmaxLseOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();
    aclnnStatus ret = MakeDenseInputsContiguous(
        query, key, value, blockTable, actualSeqLengths, actualSeqLengthsKv,
        qDequantScale, kDequantScale, vDequantScale, executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    auto outputs = l0op::DenseAttentionScore(
        query, key, value, blockTable,
        actualSeqLengths, actualSeqLengthsKv,
        qDequantScale, kDequantScale, vDequantScale,
        numKeyValueHeads, scaleValue, blockSize, innerPrecise,
        inputLayout, attentionOut, executorImpl);
    if (outputs[0] == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DenseAttentionScore returned nullptr output.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(outputs[0], attentionOut, executorImpl);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnDenseAttentionScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDenseAttentionScore);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
