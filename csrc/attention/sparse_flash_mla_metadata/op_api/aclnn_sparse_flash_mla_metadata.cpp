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
 * \file aclnn_sparse_flash_mla_metadata.cpp
 * \brief
 */

#include "aclnn_sparse_flash_mla_metadata.h"
#include "sparse_flash_mla_metadata.h"
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
#include "../op_host/sparse_flash_mla_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional, const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional, const aclTensor *cmpTopkLengthOptional, int64_t numHeadsQ,
    int64_t numHeadsKv, int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv,
    int64_t maxSeqlenCmpKv, int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode,
    int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, const char *layoutQOptional,
    const char *layoutKvOptional, bool hasOriKv, bool hasCmpKv, const aclTensor *metaData, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    if (workspaceSize == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "workspaceSize is nullptr");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "executor is nullptr");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    L2_DFX_PHASE_1(aclnnSparseFlashMlaMetadata,
                   DFX_IN(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional,
                          sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional,
                          cmpTopkLengthOptional, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenOriKv,
                          maxSeqlenCmpKv, oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
                          layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    std::string socVersionStr = npuInfo.GetSocLongVersion();
    const char *socVersion = socVersionStr.c_str();

    auto ret = ParamsCheck(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional,
                           sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional,
                           cmpTopkLengthOptional, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenOriKv,
                           maxSeqlenCmpKv, oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft,
                           oriWinRight, layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv, aicCoreNum, aivCoreNum,
                           socVersion, metaData);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *cuSeqlensQOptionalContiguous = nullptr;
    if (cuSeqlensQOptional != nullptr) {
        cuSeqlensQOptionalContiguous = l0op::Contiguous(cuSeqlensQOptional, uniqueExecutor.get());
        if (cuSeqlensQOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_q contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *cuSeqlensOriKvOptionalContiguous = nullptr;
    if (cuSeqlensOriKvOptional != nullptr) {
        cuSeqlensOriKvOptionalContiguous = l0op::Contiguous(cuSeqlensOriKvOptional, uniqueExecutor.get());
        if (cuSeqlensOriKvOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_ori_kv contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *cuSeqlensCmpKvOptionalContiguous = nullptr;
    if (cuSeqlensCmpKvOptional != nullptr) {
        cuSeqlensCmpKvOptionalContiguous = l0op::Contiguous(cuSeqlensCmpKvOptional, uniqueExecutor.get());
        if (cuSeqlensCmpKvOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_cmp_kv contiguous is null");
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
    const aclTensor *sequsedOriKvOptionalContiguous = nullptr;
    if (sequsedOriKvOptional != nullptr) {
        sequsedOriKvOptionalContiguous = l0op::Contiguous(sequsedOriKvOptional, uniqueExecutor.get());
        if (sequsedOriKvOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_ori_kv contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *sequsedCmpKvOptionalContiguous = nullptr;
    if (sequsedCmpKvOptional != nullptr) {
        sequsedCmpKvOptionalContiguous = l0op::Contiguous(sequsedCmpKvOptional, uniqueExecutor.get());
        if (sequsedCmpKvOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_cmp_kv contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *cmpResidualKvOptionalContiguous = nullptr;
    if (cmpResidualKvOptional != nullptr) {
        cmpResidualKvOptionalContiguous = l0op::Contiguous(cmpResidualKvOptional, uniqueExecutor.get());
        if (cmpResidualKvOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cmp_residual_kv contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *oriTopkLengthOptionalContiguous = nullptr;
    if (oriTopkLengthOptional != nullptr) {
        oriTopkLengthOptionalContiguous = l0op::Contiguous(oriTopkLengthOptional, uniqueExecutor.get());
        if (oriTopkLengthOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ori_topk_length contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *cmpTopkLengthOptionalContiguous = nullptr;
    if (cmpTopkLengthOptional != nullptr) {
        cmpTopkLengthOptionalContiguous = l0op::Contiguous(cmpTopkLengthOptional, uniqueExecutor.get());
        if (cmpTopkLengthOptionalContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cmp_topk_length contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    auto output = l0op::SparseFlashMlaMetadata(
        cuSeqlensQOptionalContiguous, cuSeqlensOriKvOptionalContiguous, cuSeqlensCmpKvOptionalContiguous,
        sequsedQOptionalContiguous, sequsedOriKvOptionalContiguous, sequsedCmpKvOptionalContiguous,
        cmpResidualKvOptionalContiguous, oriTopkLengthOptionalContiguous, cmpTopkLengthOptionalContiguous, numHeadsQ,
        numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv, oriTopk, cmpTopk, cmpRatio,
        oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv,
        socVersion, aicCoreNum, aivCoreNum, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnSparseFlashMlaMetadata(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSparseFlashMlaMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif