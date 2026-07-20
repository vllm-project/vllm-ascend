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
 * \file aclnn_sparse_attn_sharedkv_metadata.cpp
 * \brief
 */

#include "aclnn_sparse_attn_sharedkv_metadata.h"
#include "l0_sparse_attn_sharedkv_metadata.h"
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
#include "../sparse_attn_sharedkv_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnSparseAttnSharedkvMetadataGetWorkspaceSize(
    const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensOriKvOptional,
    const aclTensor* cuSeqLensCmpKvOptional, const aclTensor* sequsedQOptional, const aclTensor* sequsedKvOptional,
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
    int64_t oriTopK, int64_t cmpTopK, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft,
    int64_t oriWinRight, char *layoutQOptional, char *layoutKvOptional, bool hasOriKv, bool hasCmpKv,
    const aclTensor* metadata, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSparseAttnSharedkvMetadata,
                   DFX_IN(cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional,
                          sequsedKvOptional, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenKv,
                          oriTopK, cmpTopK, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
                          layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv),
                   DFX_OUT(metadata));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional,
                           sequsedKvOptional, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenKv,
                           oriTopK, cmpTopK, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
                           layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();

    auto cuSeqLensQOptionalContiguous = l0op::Contiguous(cuSeqLensQOptional, uniqueExecutor.get());
    if (cuSeqLensQOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_q contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto cuSeqLensOriKvOptionalContiguous = l0op::Contiguous(cuSeqLensOriKvOptional, uniqueExecutor.get());
    if (cuSeqLensOriKvOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_ori_kv contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto cuSeqLensCmpKvOptionalContiguous = l0op::Contiguous(cuSeqLensCmpKvOptional, uniqueExecutor.get());
    if (cuSeqLensCmpKvOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_cmp_kv contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto sequsedQOptionalContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
    if (sequsedQOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_q contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto sequsedKvOptionalContiguous = l0op::Contiguous(sequsedKvOptional, uniqueExecutor.get());
    if (sequsedKvOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_kv contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto output = l0op::SparseAttnSharedkvMetadata(
        cuSeqLensQOptionalContiguous, cuSeqLensOriKvOptionalContiguous, cuSeqLensCmpKvOptionalContiguous,
        sequsedQOptionalContiguous, sequsedKvOptionalContiguous, numHeadsQ, numHeadsKv, headDim, batchSize,
        maxSeqlenQ, maxSeqlenKv, oriTopK, cmpTopK, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
        layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv, socVersion, aicCoreNum, aivCoreNum, metadata,
        uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnSparseAttnSharedkvMetadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSparseAttnSharedkvMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
