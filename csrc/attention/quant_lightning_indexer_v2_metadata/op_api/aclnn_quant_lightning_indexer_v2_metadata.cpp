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
 * \file aclnn_quant_lightning_indexer_v2_metadata.cpp
 * \brief
 */

#include "aclnn_quant_lightning_indexer_v2_metadata.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "quant_lightning_indexer_v2_metadata.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

// Host-side layout/seq checks live in quant_lightning_indexer_v2_metadata_check.h
// (op_common/log vs aclnn op_log). Keep them out of this TU so it matches the
// existing VA QLI metadata aclnn path.
static aclnnStatus ParamsCheckQliV2(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, char *, char *, int64_t, int64_t, const aclTensor *, uint32_t,
    uint32_t, const std::string &)
{
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional, int64_t numHeadsQ, int64_t numHeadsK,
    int64_t headDim, int64_t topk, int64_t quantMode, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK,
    char *layoutQOptional, char *layoutKOptional, int64_t maskMode, int64_t cmpRatio, const aclTensor *metadata,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (workspaceSize == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "workspaceSize is nullptr");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "executor is nullptr");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    L2_DFX_PHASE_1(aclnnQuantLightningIndexerV2Metadata,
                   DFX_IN(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                          cmpResidualKOptional, numHeadsQ, numHeadsK, headDim, topk, quantMode, batchSize, maxSeqlenQ,
                          maxSeqlenK, layoutQOptional, layoutKOptional, maskMode, cmpRatio),
                   DFX_OUT(metadata));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const std::string socVersion = npuInfo.GetSocLongVersion();

    auto ret = ParamsCheckQliV2(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                                cmpResidualKOptional, numHeadsQ, numHeadsK, headDim, topk, quantMode, batchSize,
                                maxSeqlenQ, maxSeqlenK, layoutQOptional, layoutKOptional, maskMode, cmpRatio, metadata,
                                aicCoreNum, aivCoreNum, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *cuSeqlensQOptionalContiguous = nullptr;
    if (cuSeqlensQOptional != nullptr) {
        cuSeqlensQOptionalContiguous = l0op::Contiguous(cuSeqlensQOptional, uniqueExecutor.get());
        if (cuSeqlensQOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_q contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *cuSeqlensKOptionalContiguous = nullptr;
    if (cuSeqlensKOptional != nullptr) {
        cuSeqlensKOptionalContiguous = l0op::Contiguous(cuSeqlensKOptional, uniqueExecutor.get());
        if (cuSeqlensKOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_k contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *sequsedQOptionalContiguous = nullptr;
    if (sequsedQOptional != nullptr) {
        sequsedQOptionalContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
        if (sequsedQOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_q contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *sequsedKOptionalContiguous = nullptr;
    if (sequsedKOptional != nullptr) {
        sequsedKOptionalContiguous = l0op::Contiguous(sequsedKOptional, uniqueExecutor.get());
        if (sequsedKOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_k contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *cmpResidualLKOptionalContiguous = nullptr;
    if (cmpResidualKOptional != nullptr) {
        cmpResidualLKOptionalContiguous = l0op::Contiguous(cmpResidualKOptional, uniqueExecutor.get());
        if (cmpResidualLKOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cmp_residual_k contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    auto output = l0op::QuantLightningIndexerV2Metadata(
        cuSeqlensQOptionalContiguous, cuSeqlensKOptionalContiguous, sequsedQOptionalContiguous,
        sequsedKOptionalContiguous, cmpResidualLKOptionalContiguous, numHeadsQ, numHeadsK, headDim, topk, quantMode,
        batchSize, maxSeqlenQ, maxSeqlenK, layoutQOptional, layoutKOptional, maskMode, cmpRatio, aicCoreNum, aivCoreNum,
        socVersion.c_str(), metadata, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantLightningIndexerV2Metadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantLightningIndexerV2Metadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
