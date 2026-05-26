/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_H
#define OP_API_ACLNN_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief FusedSigmoidGatingDeltaRuleUpdate 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @param [in] aLog: 数据类型支持：float32。
 * @param [in] a: 数据类型支持：bfloat16。
 * @param [in] b: 数据类型支持：bfloat16。
 * @param [in] dtBias: 数据类型支持：float32。
 * @param [in] query: 数据类型支持：bfloat16。
 * @param [in] key: 数据类型支持：bfloat16。
 * @param [in] value: 数据类型支持：bfloat16。
 * @param [in,out] stateRef: 数据类型支持：bfloat16、float32。
 * @param [in] actualSeqLengths: 数据类型支持：int32。
 * @param [in] ssmStateIndices: 数据类型支持：int32。
 * @param [in] numAcceptedTokens: 数据类型支持：int32。
 * @param [in] scaleValue: 数据类型支持：float32。
 * @param [in] softplusBeta: 数据类型支持：float32。
 * @param [in] softplusThreshold: 数据类型支持：float32。
 * @param [out] out: 数据类型支持：bfloat16。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedSigmoidGatingDeltaRuleUpdateGetWorkspaceSize(
    const aclTensor *aLog, const aclTensor *a, const aclTensor *b, const aclTensor *dtBias,
    const aclTensor *query, const aclTensor *key, const aclTensor *value, aclTensor *stateRef,
    const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices, const aclTensor *numAcceptedTokens,
    float scaleValue, float softplusBeta, float softplusThreshold, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief FusedSigmoidGatingDeltaRuleUpdate 的第二段接口，用于执行算子。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnFusedSigmoidGatingDeltaRuleUpdateGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedSigmoidGatingDeltaRuleUpdate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_H
