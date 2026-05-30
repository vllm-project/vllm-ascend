/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string.h>
#include "graph/types.h"
#include "aclnn_scatter_pa_kv_cache_vllm.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerScatterPaKvCacheVllmGetWorkspaceSize(
    const aclTensor *key,
    aclTensor *keyCacheRef,
    const aclTensor *slotMapping,
    const aclTensor *value,
    aclTensor *valueCacheRef,
    const aclTensor *compressLensOptional,
    const aclTensor *compressSeqOffsetOptional,
    const aclTensor *seqLensOptional,
    char *cacheModeOptional,
    char *scatterModeOptional,
    const aclIntArray *stridesOptional,
    const aclIntArray *offsetsOptional,
    const aclIntArray *kvCacheStridesOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerScatterPaKvCacheVllm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);



aclnnStatus aclnnScatterPaKvCacheVllmGetWorkspaceSize(
    const aclTensor *key,
    aclTensor *keyCacheRef,
    const aclTensor *slotMapping,
    const aclTensor *value,
    aclTensor *valueCacheRef,
    const aclTensor *compressLensOptional,
    const aclTensor *compressSeqOffsetOptional,
    const aclTensor *seqLensOptional,
    char *cacheModeOptional,
    char *scatterModeOptional,
    const aclIntArray *stridesOptional,
    const aclIntArray *offsetsOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    int64_t *keyCacheStridesValue = nullptr;
    uint64_t keyCacheStridesNum = 0;

    int64_t *valueCacheStridesValue = nullptr;
    uint64_t valueCacheStridesNum = 0;

    aclnnStatus ret = aclGetViewStrides(keyCacheRef, &keyCacheStridesValue, &keyCacheStridesNum);
    ret = aclGetViewStrides(valueCacheRef, &valueCacheStridesValue, &valueCacheStridesNum);

    std::vector<int64_t> sizeData = {0, 0};

    if (!op::IsContiguous(keyCacheRef) && !op::IsContiguous(valueCacheRef)){
        sizeData = {keyCacheStridesValue[0], valueCacheStridesValue[0]};
    }

    aclIntArray *kvCacheStridesOptional = aclCreateIntArray(sizeData.data(), sizeData.size());

    ret = aclnnInnerScatterPaKvCacheVllmGetWorkspaceSize(key, keyCacheRef, slotMapping, value, valueCacheRef,
                                                     compressLensOptional, compressSeqOffsetOptional, seqLensOptional,
                                                     cacheModeOptional, scatterModeOptional, stridesOptional,
                                                     offsetsOptional, kvCacheStridesOptional, workspaceSize, executor);

    aclDestroyIntArray(kvCacheStridesOptional);
    delete[] keyCacheStridesValue;
    delete[] valueCacheStridesValue;
    return ret;
}

aclnnStatus aclnnScatterPaKvCacheVllm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    return aclnnInnerScatterPaKvCacheVllm(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif