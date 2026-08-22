/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "aclnn_fused_infer_attention_score.h"
#include "fused_infer_attention_score_inner.h"
#include "aclnnInner_fused_infer_attention_score.h" // 该文件为自动生成，在build/autogen/inner路径下

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
aclnnStatus aclnnFusedInferAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1, const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blockTable, const aclTensor *queryPaddingSize, const aclTensor *kvPaddingSize, int64_t numHeads,
    double scaleValue, int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFusedInferAttentionScore,
                   DFX_IN(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKv,
                          deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale,
                          antiquantOffset, blockTable, queryPaddingSize, kvPaddingSize, numHeads, scaleValue,
                          preTokens, nextTokens, inputLayout, numKeyValueHeads, sparseMode, innerPrecise,
                          blockSize, antiquantMode, softmaxLseFlag),
                   DFX_OUT(attentionOut, softmaxLse));

    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Interface aclnnFusedInferAttentionScore versions V1 to V4 are no longer supported on Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    static bool isFirstCall = true;
    if (isFirstCall) {
        OP_LOGW("aclnnFusedInferAttentionScoreGetWorkspaceSize is scheduled to be deprecated in December 2026, "
                "and will be replaced by the aclnnFusedInferAttentionScoreV5GetWorkspaceSize. "
                "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.");
        isFirstCall = false;
    }
    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;
    FusedInferAttentionScoreProcessSoftmaxLse(softmaxLseFlag, softmaxLse, tempTensor, placeHolder);

    aclnnStatus ret = InnerFusedInferAttentionScoreGetWorkspaceSize(
        query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKv, deqScale1, quantScale1, deqScale2,
        quantScale2, quantOffset2, antiquantScale, antiquantOffset, blockTable, queryPaddingSize, kvPaddingSize,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        numHeads, scaleValue, preTokens, nextTokens,
        inputLayout, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, 0, 0, 0, 0, 0,
        attentionOut, placeHolder, workspaceSize, executor);
    if (softmaxLseFlag == false) {
        aclDestroyTensor(tempTensor);
    }
    return ret;
}

aclnnStatus aclnnFusedInferAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          const aclrtStream stream)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Interface aclnnFusedInferAttentionScore versions V1 to V4 are no longer supported on Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    static bool isFirstCall = true;
    if (isFirstCall) {
        OP_LOGW("aclnnFusedInferAttentionScore is scheduled to be deprecated in December 2026, "
                "and will be replaced by the aclnnFusedInferAttentionScoreV5. "
                "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ");
        isFirstCall = false;
    }
    return InnerFusedInferAttentionScore(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
