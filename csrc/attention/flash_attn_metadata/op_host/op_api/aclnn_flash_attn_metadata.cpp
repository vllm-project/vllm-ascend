/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_flash_attn_metadata.cpp
 * \brief
 */

#include "l0_flash_attn_metadata.h"
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

#include "../flash_attn_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACLNN_API __attribute__((visibility("default")))

ACLNN_API aclnnStatus aclnnFlashAttnMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ,
    int64_t numHeadsKv, int64_t headDim, int64_t headDimV, int64_t maskMode, int64_t winLeft, int64_t winRight,
    const char *layoutQ, const char *layoutKv, const char *layoutOut, bool isGradEnabled, const aclTensor *metaData,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFlashAttnMetadata,
                   DFX_IN(cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, batchSize,
                          maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, headDimV, maskMode, winLeft,
                          winRight, layoutQ, layoutKv, layoutOut, isGradEnabled),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = FlashAttnMetadataCheck::ParamsCheck(cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional,
                                                   sequsedKvOptional, batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ,
                                                   numHeadsKv, headDim, headDimV, maskMode, winLeft, winRight, layoutQ,
                                                   layoutKv, layoutOut, metaData, isGradEnabled);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();

    auto output = l0op::FlashAttnMetadata(cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional,
                                          batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, headDimV,
                                          maskMode, winLeft, winRight, layoutQ, layoutKv, layoutOut, isGradEnabled,
                                          socVersion, aicCoreNum, aivCoreNum, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus aclnnFlashAttnMetadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttnMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
