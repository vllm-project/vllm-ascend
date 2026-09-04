/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_FUSED_INFER_ATTENTION_SCORE_V3_H_
#define ACLNN_FUSED_INFER_ATTENTION_SCORE_V3_H_
#warning "aclnn_fused_infer_attention_score_v3.h is scheduled to be deprecated in December 2026, and will be replaced by the aclnn_fused_infer_attention_score_v5.h. We apologize for any inconvenience caused and appreciate your timely migration to the new interface. "
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnFusedInferAttentionScoreV3 calculates the workspace size based on the specific calculation process.
 * @domain aclnn_ops_infer
 */
__attribute__((deprecated("aclnnFusedInferAttentionScoreV3GetWorkspaceSize is scheduled to be deprecated in December 2026, and will be replaced by the aclnnFusedInferAttentionScoreV5GetWorkspaceSize. "
 	                      "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ")))
__attribute__((visibility("default"))) aclnnStatus aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional, const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional, const aclTensor *queryRopeOptional, 
    const aclTensor *keyRopeOptional, const aclTensor *keyRopeAntiquantScaleOptional,
    int64_t numHeads, double scaleValue, int64_t preTokens,
    int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode, int64_t valueAntiquantMode,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnFusedInferAttentionScoreV3 is used to perform calculations.
 */
__attribute__((deprecated("aclnnFusedInferAttentionScoreV3 is scheduled to be deprecated in December 2026, and will be replaced by the aclnnFusedInferAttentionScoreV5. "
 	                      "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ")))
__attribute__((visibility("default"))) aclnnStatus aclnnFusedInferAttentionScoreV3(void *workspace,
                                                                                   uint64_t workspaceSize,
                                                                                   aclOpExecutor *executor,
                                                                                   const aclrtStream stream);


#ifdef __cplusplus
}
#endif

#endif