/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_FUSED_GDN_GATING_V310_H
#define OP_API_ACLNN_FUSED_GDN_GATING_V310_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// 310P 专属的 Phase-1 符号接口，后缀严格对齐 V310
__attribute__((visibility("default"))) aclnnStatus aclnnFusedGdnGatingV310GetWorkspaceSize(
    const aclTensor *aLog, const aclTensor *a, const aclTensor *b,
    const aclTensor *dtBias, float beta, float threshold,
    aclTensor *g, aclTensor *betaOutput,
    uint64_t *workspaceSize, aclOpExecutor **executor);

// 310P 专属的 Phase-2 符号接口
__attribute__((visibility("default"))) aclnnStatus aclnnFusedGdnGatingV310(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_FUSED_GDN_GATING_V310_H
