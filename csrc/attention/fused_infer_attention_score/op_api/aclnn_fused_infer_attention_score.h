/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_FUSED_INFER_ATTENTION_SCORE_H_
#define ACLNN_FUSED_INFER_ATTENTION_SCORE_H_
#warning "aclnn_fused_infer_attention_score.h is scheduled to be deprecated in December 2026, and will be replaced by the aclnn_fused_infer_attention_score_v5.h. We apologize for any inconvenience caused and appreciate your timely migration to the new interface. "
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnFusedInferAttentionScore calculates the workspace size based on the specific calculation process.
 * @domain aclnn_ops_infer
 */
__attribute__((deprecated("aclnnFusedInferAttentionScoreGetWorkspaceSize is scheduled to be deprecated in December 2026, and will be replaced by the aclnnFusedInferAttentionScoreV5GetWorkspaceSize. "
                          "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ")))
__attribute__((visibility("default"))) aclnnStatus aclnnFusedInferAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1, const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blockTable, const aclTensor *queryPaddingSize, const aclTensor *kvPaddingSize, int64_t numHeads,
    double scaleValue, int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnFusedInferAttentionScore is used to perform calculations.
 */
__attribute__((deprecated("aclnnFusedInferAttentionScore is scheduled to be deprecated in December 2026, and will be replaced by the aclnnFusedInferAttentionScoreV5. "
                          "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ")))
__attribute__((visibility("default"))) aclnnStatus aclnnFusedInferAttentionScore(void *workspace,
                                                                                 uint64_t workspaceSize,
                                                                                 aclOpExecutor *executor,
                                                                                 const aclrtStream stream);


#ifdef __cplusplus
}
#endif

#endif