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
#include "aclnn_fuse_dense_allgather.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerFuseDenseAllgatherGetWorkspaceSize(
    const aclTensor *x,
    char *groupTp,
    int64_t tpRankSize,
    int64_t tpRankId,
    const aclTensor *yOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerFuseDenseAllgather(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

aclnnStatus aclnnFuseDenseAllgatherGetWorkspaceSize(
    const aclTensor *x,
    char *groupTp,
    int64_t tpRankSize,
    int64_t tpRankId,
    const aclTensor *y,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerFuseDenseAllgatherGetWorkspaceSize(x, groupTp, tpRankSize, tpRankId, y, workspaceSize, executor);
}

aclnnStatus aclnnFuseDenseAllgather(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerFuseDenseAllgather(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
