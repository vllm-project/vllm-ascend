/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_store_kv_block_metadata.cpp
 * \brief AClnn interface for StoreKvBlockMetadata operator
 */

#include "aclnn_store_kv_block_metadata.h"
#include "l0_store_kv_block_metadata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnStoreKvBlockMetadataGetWorkspaceSize(
    const aclTensor *slotMapping,
    const aclTensor *groupLen,
    const aclTensor *groupKeyIdx,
    const aclTensor *groupKeyCacheIdx,
    int64_t blockSize,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnStoreKvBlockMetadata,
                   DFX_IN(slotMapping, blockSize),
                   DFX_OUT(groupLen, groupKeyIdx, groupKeyCacheIdx));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // basic parameter checks
    CHECK_RET(slotMapping != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(groupLen != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(groupKeyIdx != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(groupKeyCacheIdx != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(blockSize > 0, ACLNN_ERR_PARAM_INVALID);

    auto slotMappingContiguous = l0op::Contiguous(slotMapping, uniqueExecutor.get());
    CHECK_RET(slotMappingContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto ret = l0op::StoreKvBlockMetadata(slotMappingContiguous,  groupLen, groupKeyIdx, groupKeyCacheIdx,blockSize,
                                          uniqueExecutor.get());
    CHECK_RET(ret != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnStoreKvBlockMetadata(void *workspace, uint64_t workspaceSize,
                                                                             aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnStoreKvBlockMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
