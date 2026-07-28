/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sparse_attention_score.h"

#include "sparse_attention_score.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/common_types.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include <acl/acl.h>
#include <string>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static aclnnStatus MakeContiguous(const aclTensor *&query,
                                  const aclTensor *&key,
                                  const aclTensor *&value,
                                  const aclTensor *&selectIdx,
                                  const aclTensor *&blockTable,
                                  const aclTensor *&selectNumIdxOptional,
                                  const aclTensor *&actualSeqLengthsOptional,
                                  const aclTensor *&actualSeqLengthsKvOptional,
                                  aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    selectIdx = l0op::Contiguous(selectIdx, executor);
    CHECK_RET(selectIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    blockTable = l0op::Contiguous(blockTable, executor);
    CHECK_RET(blockTable != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    if (selectNumIdxOptional != nullptr) {
        selectNumIdxOptional = l0op::Contiguous(selectNumIdxOptional, executor);
        CHECK_RET(selectNumIdxOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (actualSeqLengthsOptional != nullptr) {
        actualSeqLengthsOptional = l0op::Contiguous(actualSeqLengthsOptional, executor);
        CHECK_RET(actualSeqLengthsOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (actualSeqLengthsKvOptional != nullptr) {
        actualSeqLengthsKvOptional = l0op::Contiguous(actualSeqLengthsKvOptional, executor);
        CHECK_RET(actualSeqLengthsKvOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnSparseAttentionScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *selectIdx,
    const aclTensor *blockTable,
    const aclTensor *selectNumIdxOptional,
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *qDequantScaleOptional,
    const aclTensor *kDequantScaleOptional,
    const aclTensor *vDequantScaleOptional,
    int64_t numKeyValueHeads,
    double scaleValue,
    int64_t blockSize,
    int64_t topK,
    int64_t innerPrecise,
    aclTensor *attentionOut,
    aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(selectIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(blockTable != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    L2_DFX_PHASE_1(aclnnSparseAttentionScore,
                   DFX_IN(query, key, value, selectIdx, blockTable, selectNumIdxOptional,
                          actualSeqLengthsOptional, actualSeqLengthsKvOptional,
                          qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional,
                          numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise),
                   DFX_OUT(attentionOut, softmaxLseOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();

    aclnnStatus ret = MakeContiguous(query, key, value, selectIdx, blockTable,
                                     selectNumIdxOptional,
                                     actualSeqLengthsOptional, actualSeqLengthsKvOptional,
                                     executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    auto outputs = l0op::SparseAttentionScore(
        query, key, value, selectIdx, blockTable,
        selectNumIdxOptional,
        actualSeqLengthsOptional, actualSeqLengthsKvOptional,
        qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional,
        numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise,
        executorImpl);

    if (outputs[0] == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SparseAttentionScore returned nullptr output.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(outputs[0], attentionOut, executorImpl);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // if (outputs[1] != nullptr) {
    //     auto viewCopyLseResult = l0op::ViewCopy(outputs[1], softmaxLseOptional, executorImpl);
    //     CHECK_RET(viewCopyLseResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // }

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnSparseAttentionScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSparseAttentionScore);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
