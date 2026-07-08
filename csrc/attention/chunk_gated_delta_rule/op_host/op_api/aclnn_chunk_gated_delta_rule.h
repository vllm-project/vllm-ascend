/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_CHUNK_GATED_DELTA_RULE_H
#define OP_API_ACLNN_CHUNK_GATED_DELTA_RULE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ChunkGatedDeltaRule phase-1: compute required workspace size.
 * @param [in]  query:            BF16, [T, Nk, Dk].
 * @param [in]  key:              BF16, [T, Nk, Dk].
 * @param [in]  value:            BF16, [T, Nv, Dv].
 * @param [in]  beta:             BF16, [T, Nk, Dk].
 * @param [in]  initialState:     FLOAT, [B, Nk, Dk, Dv].
 * @param [in]  actualSeqLengths: INT32, [B+1].
 * @param [in]  g:                FLOAT (optional).
 * @param [in]  scaleValue:       FLOAT.
 * @param [out] out:              BF16, [T, Nv, Dv].
 * @param [out] finalState:       FLOAT, [B, Nk, Dk, Dv].
 * @param [out] workspaceSize:    required workspace bytes on device.
 * @param [out] executor:         op executor handle.
 * @return aclnnStatus
 */
__attribute__((visibility("default"))) aclnnStatus aclnnChunkGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *beta, const aclTensor *initialState,
    const aclTensor *actualSeqLengths, const aclTensor *g,
    float scaleValue, aclTensor *out, aclTensor *finalState,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief ChunkGatedDeltaRule phase-2: launch the kernel.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnChunkGatedDeltaRule(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_CHUNK_GATED_DELTA_RULE_H
