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
 * \file aclnn_flash_mla_with_kvcache_metadata.cpp
 * \brief
 */

#include "aclnn_flash_mla_with_kvcache_metadata.h"
#include "l0_flash_mla_with_kvcache_metadata.h"
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

#include "../flash_mla_with_kvcache_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus FlashMlaMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cacheSeqlensOptional, const aclTensor *sequsedQOptional,
    int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDimQk, int64_t headDimV,
    int64_t maskMode, const char *layoutQ, const aclTensor *metaData, uint64_t *workspaceSize, aclOpExecutor **executor,
    int64_t isC8)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 平台参数由 GetCurrentPlatformInfo（硬件）推导，先取出供容量校验和 l0 透传使用
    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();

    auto ret = FlashMlaWithKvcacheMetadataCheck::ParamsCheck(
        cuSeqlensQOptional, cacheSeqlensOptional, sequsedQOptional, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv,
        headDimQk, headDimV, maskMode, layoutQ, metaData, aicCoreNum, aivCoreNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 前置处理：seqlens 输入转连续（AICPU kernel 按连续内存读取原始数据）
    const aclTensor *cuSeqlensQContiguous = nullptr;
    if (cuSeqlensQOptional != nullptr) {
        cuSeqlensQContiguous = l0op::Contiguous(cuSeqlensQOptional, uniqueExecutor.get());
        if (cuSeqlensQContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_q contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *cacheSeqlensContiguous = nullptr;
    if (cacheSeqlensOptional != nullptr) {
        cacheSeqlensContiguous = l0op::Contiguous(cacheSeqlensOptional, uniqueExecutor.get());
        if (cacheSeqlensContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cache_seqlens contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *sequsedQContiguous = nullptr;
    if (sequsedQOptional != nullptr) {
        sequsedQContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
        if (sequsedQContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_q contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    // l0 函数输入槽位与 meta.txt 定义的 3 输入契约一致：
    //   slot0=cuSeqlensQ, slot1=cacheSeqlens, slot2=sequsedQ
    // 平台参数 socVersion/aicCoreNum/aivCoreNum 由 GetCurrentPlatformInfo 推导后透传。
    auto output =
        l0op::FlashMlaWithKvcacheMetadata(cuSeqlensQContiguous, cacheSeqlensContiguous, sequsedQContiguous, maxSeqlenQ,
                                          maxSeqlenKv, numHeadsQ, numHeadsKv, headDimQk, headDimV, maskMode, layoutQ,
                                          socVersion, aicCoreNum, aivCoreNum, isC8, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashMlaWithKvcacheMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cacheSeqlensOptional, const aclTensor *sequsedQOptional,
    int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDimQk, int64_t headDimV,
    int64_t maskMode, const char *layoutQ, const aclTensor *metaData, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFlashMlaWithKvcacheMetadata,
                  DFX_IN(cuSeqlensQOptional, cacheSeqlensOptional, sequsedQOptional, maxSeqlenQ, maxSeqlenKv,
                         numHeadsQ, numHeadsKv, headDimQk, headDimV, maskMode, layoutQ), DFX_OUT(metaData));
    return FlashMlaMetadataGetWorkspaceSize(cuSeqlensQOptional, cacheSeqlensOptional, sequsedQOptional,
        maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDimQk, headDimV, maskMode, layoutQ,
        metaData, workspaceSize, executor, 0);
}

aclnnStatus aclnnFlashMlaWithKvcacheMetadataC8GetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cacheSeqlensOptional, const aclTensor *sequsedQOptional,
    int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDimQk, int64_t headDimV,
    int64_t maskMode, const char *layoutQ, const aclTensor *metaData, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFlashMlaWithKvcacheMetadataC8,
                  DFX_IN(cuSeqlensQOptional, cacheSeqlensOptional, sequsedQOptional, maxSeqlenQ, maxSeqlenKv,
                         numHeadsQ, numHeadsKv, headDimQk, headDimV, maskMode, layoutQ), DFX_OUT(metaData));
    return FlashMlaMetadataGetWorkspaceSize(cuSeqlensQOptional, cacheSeqlensOptional, sequsedQOptional,
        maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDimQk, headDimV, maskMode, layoutQ,
        metaData, workspaceSize, executor, 1);
}

aclnnStatus aclnnFlashMlaWithKvcacheMetadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashMlaWithKvcacheMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}


aclnnStatus aclnnFlashMlaWithKvcacheMetadataC8(void *workspace, uint64_t workspaceSize,
                                           aclOpExecutor *executor, aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
