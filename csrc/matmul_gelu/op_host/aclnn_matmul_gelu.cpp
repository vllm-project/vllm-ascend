/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string.h>
#include "graph/types.h"
#include "aclnn/opdev/platform.h"
#include "aclnn_matmul_gelu.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnInnerMatmulGeluGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *bias,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMatmulGelu(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

aclnnStatus aclnnMatmulGeluGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *bias,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    aclnnStatus ret = aclnnInnerMatmulGeluGetWorkspaceSize(x, weight, bias, out, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnMatmulGelu(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerMatmulGelu(workspace, workspaceSize, executor, stream);
    return ret;
}
#ifdef __cplusplus
}
#endif