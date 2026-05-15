/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_kv_quant_sparse_attn_sharedkv_metadata.cpp
 * \brief
 */

#include "aclnn_kv_quant_sparse_attn_sharedkv_metadata.h"
#include "l0_kv_quant_sparse_attn_sharedkv_metadata.h"
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

static aclnnStatus ParamsCheck(const aclTensor* cuSeqLensQOptional,
                               const aclTensor* cuSeqLensOriKvOptional,
                               const aclTensor* cuSeqLensCmpKvOptional,
                               const aclTensor* sequsedQOptional,
                               const aclTensor* sequsedKvOptional,
                               int64_t numHeadsQ,
                               int64_t numHeadsKv,
                               int64_t headDim,
                               int64_t batchSizeOptional,
                               int64_t maxSeqlenQOptional,
                               int64_t maxSeqlenKvOptional,
                               int64_t oriTopKOptional,
                               int64_t cmpTopKOptional,
                               int64_t kvQuantMode,
                               int64_t tileSizeOptional,
                               int64_t ropeHeadDimOptional,
                               int64_t cmpRatioOptional,
                               int64_t oriMaskModeOptional,
                               int64_t cmpMaskModeOptional,
                               int64_t oriWinLeftOptional,
                               int64_t oriWinRightOptional,
                               char *layoutQOptional,
                               char *layoutKvOptional,
                               bool hasOriKvOptional,
                               bool hasCmpKvOptional,
                               const aclTensor* metaData) {
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnKvQuantSparseAttnSharedkvMetadataGetWorkspaceSize(
    const aclTensor* cuSeqLensQOptional,
    const aclTensor* cuSeqLensOriKvOptional,
    const aclTensor* cuSeqLensCmpKvOptional,
    const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKvOptional,
    int64_t numHeadsQ,
    int64_t numHeadsKv,
    int64_t headDim,
    int64_t batchSizeOptional,
    int64_t maxSeqlenQOptional,
    int64_t maxSeqlenKvOptional,
    int64_t oriTopKOptional,
    int64_t cmpTopKOptional,
    int64_t kvQuantMode,
    int64_t tileSizeOptional,
    int64_t ropeHeadDimOptional,
    int64_t cmpRatioOptional,
    int64_t oriMaskModeOptional,
    int64_t cmpMaskModeOptional,
    int64_t oriWinLeftOptional,
    int64_t oriWinRightOptional,
    char *layoutQOptional,
    char *layoutKvOptional,
    bool hasOriKvOptional,
    bool hasCmpKvOptional,
    const aclTensor* metaData,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnKvQuantSparseAttnSharedkvMetadata,
                 DFX_IN(cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional, sequsedKvOptional, numHeadsQ, numHeadsKv, headDim, batchSizeOptional, 
                        maxSeqlenQOptional, maxSeqlenKvOptional, oriTopKOptional, cmpTopKOptional, kvQuantMode, tileSizeOptional, ropeHeadDimOptional, cmpRatioOptional, oriMaskModeOptional, 
                        cmpMaskModeOptional, oriWinLeftOptional, oriWinRightOptional, layoutQOptional, layoutKvOptional, 
                        hasOriKvOptional, hasCmpKvOptional),
                 DFX_OUT(metaData));

  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  auto ret = ParamsCheck(
      cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional, sequsedKvOptional, numHeadsQ, numHeadsKv, headDim, batchSizeOptional, 
      maxSeqlenQOptional, maxSeqlenKvOptional, oriTopKOptional, cmpTopKOptional, kvQuantMode, tileSizeOptional, ropeHeadDimOptional, cmpRatioOptional, oriMaskModeOptional, 
      cmpMaskModeOptional, oriWinLeftOptional, oriWinRightOptional, layoutQOptional, layoutKvOptional, 
      hasOriKvOptional, hasCmpKvOptional, metaData);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
  uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
  uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
  const char *socVersion = npuInfo.GetSocLongVersion().c_str();
  auto output = l0op::KvQuantSparseAttnSharedkvMetadata(
      cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional, sequsedKvOptional, numHeadsQ, numHeadsKv, headDim, batchSizeOptional, 
      maxSeqlenQOptional, maxSeqlenKvOptional, oriTopKOptional, cmpTopKOptional, kvQuantMode, tileSizeOptional, ropeHeadDimOptional, cmpRatioOptional, oriMaskModeOptional, 
      cmpMaskModeOptional, oriWinLeftOptional, oriWinRightOptional, layoutQOptional, layoutKvOptional, 
      hasOriKvOptional, hasCmpKvOptional, socVersion, aicCoreNum, aivCoreNum, metaData, 
      uniqueExecutor.get());
  CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = 0;
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnKvQuantSparseAttnSharedkvMetadata(void *workspace, uint64_t workspaceSize,
                                aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnKvQuantSparseAttnSharedkvMetadata);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
