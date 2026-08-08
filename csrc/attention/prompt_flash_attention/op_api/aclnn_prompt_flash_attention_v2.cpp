/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_prompt_flash_attention_v2.h"
#include "prompt_flash_attention_inner.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" { // Compile in C format
#endif

namespace {
aclnnStatus aclnnPromptFlashAttentionV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask,
    const aclIntArray *actualSeqLengths,
    const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1,
    const aclTensor *quantScale1, // quantScale1 of V2
    const aclTensor *deqScale2,
    const aclTensor *quantScale2,
    const aclTensor *quantOffset2,
    int64_t numHeads,
    double scaleValue,
    int64_t preTokens,  // Preceding tokens count
    int64_t nextTokens, // Subsequent tokens count
    char *inputLayout,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    const aclTensor *attentionOut,  // Attention output tensor
    uint64_t *workspaceSize,
    aclOpExecutor **executor) {
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Interface aclnnPromptFlashAttention versions V1 to V3 are no longer supported on Ascend950.");
            return ACLNN_ERR_RUNTIME_ERROR;
        }
        static bool isFirstCall = true;
        if (isFirstCall) {
            OP_LOGW("aclnnPromptFlashAttentionV2GetWorkspaceSize is scheduled to be deprecated in December 2026, "
                    "and will be replaced by the aclnnPromptFlashAttentionV3GetWorkspaceSize. "
                    "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.");
            isFirstCall = false;
        }
        (void) pseShift;
        int64_t innerPrecise = 1;
        return InnerPromptFlashAttentionGetWorkspaceSize(query, key, value, nullptr, attenMask,
                                                              actualSeqLengths, actualSeqLengthsKv,
                                                              deqScale1, quantScale1, deqScale2,
                                                              quantScale2, quantOffset2,
                                                              numHeads, scaleValue, preTokens, nextTokens,
                                                              inputLayout, numKeyValueHeads, sparseMode,
                                                              innerPrecise, attentionOut, workspaceSize, executor);
    }

aclnnStatus aclnnPromptFlashAttentionV2(
    void *workspace,    // pointer for storing temporary data
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream) { // V2 call aclnn inner
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Interface aclnnPromptFlashAttention versions V1 to V3 are no longer supported on Ascend950.");
            return ACLNN_ERR_RUNTIME_ERROR;
        }
        static bool isFirstCall = true;
        if (isFirstCall) {
            OP_LOGW("aclnnPromptFlashAttentionV2 is scheduled to be deprecated in December 2026, "
                    "and will be replaced by the aclnnPromptFlashAttentionV3. "
                    "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.");
            isFirstCall = false;
        }
        return InnerPromptFlashAttention(workspace, workspaceSize, executor, stream);
    }

}

#ifdef __cplusplus
}
#endif