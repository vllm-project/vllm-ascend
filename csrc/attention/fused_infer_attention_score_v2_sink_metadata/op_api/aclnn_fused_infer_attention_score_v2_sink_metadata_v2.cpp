/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_fused_infer_attention_score_v2_sink_metadata_v2.cpp
 * \brief
 */

#include "aclnn_fused_infer_attention_score_v2_sink_metadata_v2.h"
#include "l0_fused_infer_attention_score_v2_sink_metadata.h"
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

namespace {
static aclnnStatus ParamsCheck(const aclTensor *actualSeqLengthsOptional,
                               const aclTensor *actualSeqLengthsKvOptional,
                               int64_t numHeadsQ,
                               int64_t numHeadsKv,
                               int64_t headDimQk,
                               int64_t headDimV,
                               int64_t batchSizeOptional,
                               int64_t sparseModeOptional,
                               int64_t preTokensOptional,
                               int64_t nextTokensOptional,
                               char *inputLayoutOptional,
                               char *inputLayoutKvOptional,
                               int64_t sinkNumOptional,
                               int64_t kSinkNumOptional,
                               bool batchInvariantOptional,
                               int64_t ropeHeadDimOptional,
                               int64_t blockSizeOptional,
                               int64_t aicCoreNumOptional,
                               int64_t aivCoreNumOptional,
                               const aclTensor* metaData)
{
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedInferAttentionScoreV2SinkMetadataV2GetWorkspaceSize(const aclTensor *actualSeqLengthsOptional,
                                                                          const aclTensor *actualSeqLengthsKvOptional,
                                                                          int64_t numHeadsQ,
                                                                          int64_t numHeadsKv,
                                                                          int64_t headDimQk,
                                                                          int64_t headDimV,
                                                                          int64_t batchSizeOptional,
                                                                          int64_t sparseModeOptional,
                                                                          int64_t preTokensOptional,
                                                                          int64_t nextTokensOptional,
                                                                          char *inputLayoutOptional,
                                                                          char *inputLayoutKvOptional,
                                                                          int64_t sinkNumOptional,
                                                                          int64_t kSinkNumOptional,
                                                                          bool batchInvariantOptional,
                                                                          int64_t ropeHeadDimOptional,
                                                                          int64_t blockSizeOptional,
                                                                          int64_t aicCoreNumOptional,
                                                                          int64_t aivCoreNumOptional,
                                                                          const aclTensor* metaData,
                                                                          uint64_t* workspaceSize,
                                                                          aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFusedInferAttentionScoreV2SinkMetadataV2,
                   DFX_IN(actualSeqLengthsOptional,
                          actualSeqLengthsKvOptional,
                          numHeadsQ,
                          numHeadsKv,
                          headDimQk,
                          headDimV,
                          batchSizeOptional,
                          sparseModeOptional,
                          preTokensOptional,
                          nextTokensOptional,
                          inputLayoutOptional,
                          inputLayoutKvOptional,
                          sinkNumOptional,
                          kSinkNumOptional,
                          batchInvariantOptional,
                          ropeHeadDimOptional,
                          blockSizeOptional,
                          aicCoreNumOptional,
                          aivCoreNumOptional),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(actualSeqLengthsOptional,
                           actualSeqLengthsKvOptional,
                           numHeadsQ,
                           numHeadsKv,
                           headDimQk,
                           headDimV,
                           batchSizeOptional,
                           sparseModeOptional,
                           preTokensOptional,
                           nextTokensOptional,
                           inputLayoutOptional,
                           inputLayoutKvOptional,
                           sinkNumOptional,
                           kSinkNumOptional,
                           batchInvariantOptional,
                           ropeHeadDimOptional,
                           blockSizeOptional,
                           aicCoreNumOptional,
                           aivCoreNumOptional,
                           metaData);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();

    if (aicCoreNumOptional <= 0 || aicCoreNumOptional > static_cast<int64_t>(aicCoreNum)) {
        aicCoreNumOptional = static_cast<int64_t>(aicCoreNum);
    }
    if (aivCoreNumOptional <= 0 || aivCoreNumOptional > static_cast<int64_t>(aivCoreNum)) {
        aivCoreNumOptional = static_cast<int64_t>(aivCoreNum);
    }
    auto output = FusedInferAttentionScoreV2SinkMetadatal0op::FusedInferAttentionScoreV2SinkMetadata(
        actualSeqLengthsOptional,
        actualSeqLengthsKvOptional,
        numHeadsQ,
        numHeadsKv,
        headDimQk,
        headDimV,
        batchSizeOptional,
        sparseModeOptional,
        preTokensOptional,
        nextTokensOptional,
        inputLayoutOptional,
        inputLayoutKvOptional,
        sinkNumOptional,
        kSinkNumOptional,
        batchInvariantOptional,
        ropeHeadDimOptional,
        blockSizeOptional,
        socVersion,
        aicCoreNumOptional,
        aivCoreNumOptional,
        metaData,
        uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnFusedInferAttentionScoreV2SinkMetadataV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
  L2_DFX_PHASE_2(aclnnFusedInferAttentionScoreV2SinkMetadataV2);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}

#ifdef __cplusplus
}
#endif