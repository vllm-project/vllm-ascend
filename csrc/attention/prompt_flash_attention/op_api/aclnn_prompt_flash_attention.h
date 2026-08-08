/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_PROMPT_FLASH_ATTENTION_H_
#define ACLNN_PROMPT_FLASH_ATTENTION_H_
#warning "aclnn_prompt_flash_attention.h is scheduled to be deprecated in December 2026, and will be replaced by the aclnn_prompt_flash_attention_v3.h. We apologize for any inconvenience caused and appreciate your timely migration to the new interface. "
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnPromptFlashAttention is used to calculate the workspace size based on the specific calculation process.
 * @domain aclnn_math
*/
__attribute__((deprecated("aclnnPromptFlashAttentionGetWorkspaceSize is scheduled to be deprecated in December 2026, and will be replaced by the aclnnPromptFlashAttentionV3GetWorkspaceSize. "
 	                      "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ")))
__attribute__ ((visibility("default"))) aclnnStatus aclnnPromptFlashAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask,
    const aclIntArray *actualSeqLengths,
    int64_t numHeads,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    char *inputLayout,
    int64_t numKeyValueHeads,
    const aclTensor *attentionOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnPromptFlashAttention is used to perform calculations.
*/
__attribute__((deprecated("aclnnPromptFlashAttention is scheduled to be deprecated in December 2026, and will be replaced by the aclnnPromptFlashAttentionV3. "
 	                      "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ")))
__attribute__ ((visibility("default"))) aclnnStatus aclnnPromptFlashAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif