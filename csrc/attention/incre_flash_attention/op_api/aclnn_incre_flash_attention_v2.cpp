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
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "aclnn_incre_flash_attention_v2.h"
#include "aclnnInner_incre_flash_attention.h" // 该文件是自动生成，在build/autogen/inner路径下

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnIncreFlashAttentionV2GetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclTensor *dequantScale1,
    const aclTensor *quantScale1, const aclTensor *dequantScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, int64_t numHeads, double scaleValue, char *inputLayout, int64_t numKeyValueHeads,
    const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Interface aclnnIncreFlashAttention versions V1 to V4 are no longer supported on Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    static bool isFirstCall = true;
    if (isFirstCall) {
        OP_LOGW("aclnnIncreFlashAttentionV2GetWorkspaceSize is scheduled to be deprecated in December 2026, "
                "and will be replaced by the aclnnIncreFlashAttentionV4GetWorkspaceSize. "
                "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.");
        isFirstCall = false;
    }
    (void) pseShift;
    aclnnStatus ret = aclnnInnerIncreFlashAttentionGetWorkspaceSize(
        query, key, value, nullptr, attenMask, actualSeqLengths, dequantScale1, quantScale1, dequantScale2, quantScale2,
        quantOffset2, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, inputLayout, numKeyValueHeads, 0, 1,
        attentionOut, workspaceSize, executor);

    return ret;
}

aclnnStatus aclnnIncreFlashAttentionV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       const aclrtStream stream)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Interface aclnnIncreFlashAttention versions V1 to V4 are no longer supported on Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    static bool isFirstCall = true;
    if (isFirstCall) {
        OP_LOGW("aclnnIncreFlashAttentionV2 is scheduled to be deprecated in December 2026, "
                "and will be replaced by the aclnnIncreFlashAttentionV4. "
                "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.");
        isFirstCall = false;
    }
    aclnnStatus ret = aclnnInnerIncreFlashAttention(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
