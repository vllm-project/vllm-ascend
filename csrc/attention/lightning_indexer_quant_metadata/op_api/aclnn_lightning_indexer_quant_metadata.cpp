/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_lightning_indexer_quant_metadata.cpp
 * \brief
 */

#include "aclnn_lightning_indexer_quant_metadata.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "l0_lightning_indexer_quant_metadata.h"
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

static aclnnStatus ParamsCheck(const aclTensor* actualSeqLengthsQueryOptional,
                               const aclTensor* actualSeqLengthsKeyOptional,
                               int64_t numHeadsQ,
                               int64_t numHeadsK,
                               int64_t headDim,
                               int64_t queryQuantMode,
                               int64_t keyQuantMode,
                               int64_t batchSizeOptional,
                               int64_t maxSeqlenQOptional,
                               int64_t maxSeqlenKOptional,
                               char* layoutQueryOptional,
                               char* layoutKeyOptional,
                               int64_t sparseCountOptional,
                               int64_t sparseModeOptional,
                               bool isFdOptional,
                               int64_t preTokensOptional,
                               int64_t nextTokensOptional,
                               int64_t cmpRatioOptional,
                               const aclTensor* metaData) {
  return ACLNN_SUCCESS;
}

__attribute__((visibility("default")))
aclnnStatus aclnnLightningIndexerQuantMetadataGetWorkspaceSize(
    const aclTensor* actualSeqLengthsQueryOptional,
    const aclTensor* actualSeqLengthsKeyOptional,
    int64_t numHeadsQ,
    int64_t numHeadsK,
    int64_t headDim,
    int64_t queryQuantMode,
    int64_t keyQuantMode,
    int64_t batchSizeOptional,
    int64_t maxSeqlenQOptional,
    int64_t maxSeqlenKOptional,
    char* layoutQueryOptional,
    char* layoutKeyOptional,
    int64_t sparseCountOptional,
    int64_t sparseModeOptional,
    bool isFdOptional,
    int64_t preTokensOptional,
    int64_t nextTokensOptional,
    int64_t cmpRatioOptional,
    const aclTensor* metaData,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) {
  L2_DFX_PHASE_1(
      aclnnLightningIndexerQuantMetadata,
      DFX_IN(actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, numHeadsQ, numHeadsK, headDim, queryQuantMode,
             keyQuantMode, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKOptional, layoutQueryOptional, layoutKeyOptional,
             sparseCountOptional, sparseModeOptional, isFdOptional, preTokensOptional, nextTokensOptional, cmpRatioOptional),
      DFX_OUT(metaData));

  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  auto ret = ParamsCheck(actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, numHeadsQ, numHeadsK, headDim, queryQuantMode,
             keyQuantMode, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKOptional, layoutQueryOptional, layoutKeyOptional,
             sparseCountOptional, sparseModeOptional, isFdOptional, preTokensOptional, nextTokensOptional, cmpRatioOptional, metaData);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  
  const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
  uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
  uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
  const char* socVersion = npuInfo.GetSocLongVersion().c_str();

  auto output = l0op::LightningIndexerQuantMetadata(
                         actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, aicCoreNum, aivCoreNum, socVersion,
                         numHeadsQ, numHeadsK, headDim, queryQuantMode, keyQuantMode, batchSizeOptional, maxSeqlenQOptional,  
                         maxSeqlenKOptional, layoutQueryOptional, layoutKeyOptional, sparseCountOptional, sparseModeOptional,
                         isFdOptional, preTokensOptional, nextTokensOptional, cmpRatioOptional, metaData, uniqueExecutor.get());
  CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = 0;
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnLightningIndexerQuantMetadata(void* workspace,
                              uint64_t workspaceSize,
                              aclOpExecutor* executor,
                              aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnLightningIndexerQuantMetadata);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
