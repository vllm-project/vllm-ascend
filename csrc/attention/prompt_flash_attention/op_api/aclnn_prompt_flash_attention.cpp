/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_prompt_flash_attention.h"
#include "prompt_flash_attention_inner.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/slice.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
aclnnStatus aclnnPromptFlashAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask,
    const aclIntArray *actualSeqLengths,
    int64_t numHeads, // q_n
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    char *inputLayout,
    int64_t numKeyValueHeads,
    const aclTensor *attentionOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor) {
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Interface aclnnPromptFlashAttention versions V1 to V3 are no longer supported on Ascend950.");
            return ACLNN_ERR_RUNTIME_ERROR;
        }
        static bool isFirstCall = true;
        if (isFirstCall) {
            OP_LOGW("aclnnPromptFlashAttentionGetWorkspaceSize is scheduled to be deprecated in December 2026, "
                    "and will be replaced by the aclnnPromptFlashAttentionV3GetWorkspaceSize. "
                    "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.");
            isFirstCall = false;
        }   
        (void) pseShift;    
        const aclIntArray *actualSeqLengthsKv = nullptr;
        int64_t sparseMode = 0;
        int64_t innerPrecise = 1;
        const aclTensor *deqScale1 = nullptr;
        const aclTensor *quantScale1 = nullptr;
        const aclTensor *deqScale2 = nullptr;
        const aclTensor *quantScale2 = nullptr;
        const aclTensor *quantOffset2 = nullptr;
        return InnerPromptFlashAttentionGetWorkspaceSize(query, key, value, nullptr, attenMask,
                                                              actualSeqLengths, actualSeqLengthsKv,
                                                              deqScale1, quantScale1, deqScale2,
                                                              quantScale2, quantOffset2,
                                                              numHeads, scaleValue, preTokens, nextTokens,
                                                              inputLayout, numKeyValueHeads, sparseMode,
                                                              innerPrecise, attentionOut, workspaceSize, executor);
    }

aclnnStatus aclnnPromptFlashAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream) {
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Interface aclnnPromptFlashAttention versions V1 to V3 are no longer supported on Ascend950.");
            return ACLNN_ERR_RUNTIME_ERROR;
        }
        static bool isFirstCall = true;
        if (isFirstCall) {
            OP_LOGW("aclnnPromptFlashAttention is scheduled to be deprecated in December 2026, "
                    "and will be replaced by the aclnnPromptFlashAttentionV3. "
                    "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.");
            isFirstCall = false;
        }
        // perform attention computations.
        return InnerPromptFlashAttention(workspace, workspaceSize, executor, stream);
    }

}

#ifdef __cplusplus
}
#endif