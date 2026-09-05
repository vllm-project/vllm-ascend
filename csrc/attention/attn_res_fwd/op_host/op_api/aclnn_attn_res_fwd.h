/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_ATTN_RES_FWD_H
#define OP_API_ACLNN_ATTN_RES_FWD_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief AttnResFwd 第一段接口：计算 workspace 大小并构建执行器。
 * @param invRms nullable if !needBackward
 * @param probs  nullable if !needBackward
 */
ACLNN_API aclnnStatus aclnnAttnResFwdGetWorkspaceSize(const aclTensor *prefixSum, const aclTensor *blockResidual,
                                                      const aclTensor *projWeight, const aclTensor *normWeight,
                                                      double normEps, bool needBackward, aclTensor *hiddenStates,
                                                      aclTensor *invRms, aclTensor *probs, uint64_t *workspaceSize,
                                                      aclOpExecutor **executor);

/**
 * @brief AttnResFwd 第二段接口：在指定 stream 上执行计算。
 */
ACLNN_API aclnnStatus aclnnAttnResFwd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_ATTN_RES_FWD_H
