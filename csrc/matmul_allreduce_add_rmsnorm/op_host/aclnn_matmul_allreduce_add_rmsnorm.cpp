/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#include "aclnn_matmul_allreduce_add_rmsnorm.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern aclnnStatus aclnnInnerMatmulAllreduceAddRmsnormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *residual,
    const aclTensor *gamma,
    char *groupTp,
    int64_t tpRankSize,
    int64_t tpRankId,
    double epsilon,
    bool isTransB,
    bool isGatherAddOut,
    const aclTensor *yOut,
    const aclTensor *addOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMatmulAllreduceAddRmsnorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMatmulAllreduceAddRmsnormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *residual,
    const aclTensor *gamma,
    char *groupTp,
    int64_t tpRankSize,
    int64_t tpRankId,
    double epsilon,
    bool isTransB,
    bool isGatherAddOut,
    const aclTensor *y,
    const aclTensor *addOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerMatmulAllreduceAddRmsnormGetWorkspaceSize(x1, x2, residual,
        gamma, groupTp, tpRankSize, tpRankId, epsilon, isTransB, isGatherAddOut, y, addOut, workspaceSize, executor);
}

aclnnStatus aclnnMatmulAllreduceAddRmsnorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerMatmulAllreduceAddRmsnorm(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
