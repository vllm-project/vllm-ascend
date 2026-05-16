/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_RECURRENT_GETED_DELTA_RULE_H
#define OP_API_ACLNN_RECURRENT_GETED_DELTA_RULE_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief RecurrentGatedDeltaRule 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @param [in] query: 数据类型支持：bfloat16。
 * @param [in] key: 数据类型支持：bfloat16。
 * @param [in] value: 数据类型支持：bfloat16。
 * @param [in] beta: 数据类型支持：bfloat16。
 * @param [in] state: 数据类型支持：bfloat16。
 * @param [in] actualSeqLengths: 数据类型支持：int32。
 * @param [in] ssmStateIndices: 数据类型支持：int32。
 * @param [in] g: 数据类型支持：float32。
 * @param [in] gk: 数据类型支持：float32。
 * @param [in] numAcceptedTokens: 数据类型支持：int32。
 * @param [in] scaleValue: 数据类型支持：float32。
 * @param [out] out: 数据类型支持：bfloat16。
 * @param [out] 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta, aclTensor *stateRef,
    const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices, const aclTensor *g, const aclTensor *gk,
    const aclTensor *numAcceptedTokens, float scaleValue, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief 
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnRecurrentGatedDeltaRuleGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnRecurrentGatedDeltaRule(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_RECURRENT_GETED_DELTA_RULE_H