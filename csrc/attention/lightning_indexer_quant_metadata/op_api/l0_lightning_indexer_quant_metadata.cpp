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
 * \file l0_lightning_indexer_quant_metadata.cpp
 * \brief
 */

#include "l0_lightning_indexer_quant_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(LightningIndexerQuantMetadata);

const aclTensor* LightningIndexerQuantMetadata(
    const aclTensor* actualSeqLengthsQueryOptional,
    const aclTensor* actualSeqLengthsKeyOptional,
    int64_t aicCoreNum,
    int64_t aivCoreNum,
    const char* socVersion,
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
    aclOpExecutor* executor) {
  L0_DFX(LightningIndexerQuantMetadata, actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, aicCoreNum, aivCoreNum, socVersion,
                         numHeadsQ, numHeadsK, headDim, queryQuantMode, keyQuantMode, batchSizeOptional, maxSeqlenQOptional,  
                         maxSeqlenKOptional, layoutQueryOptional, layoutKeyOptional, sparseCountOptional, sparseModeOptional,
                         isFdOptional, preTokensOptional, nextTokensOptional, cmpRatioOptional, metaData);

  static internal::AicpuTaskSpace space("LightningIndexerQuantMetadata");

  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
      LightningIndexerQuantMetadata,
      OP_ATTR_NAMES({"aic_core_num", "aiv_core_num", "soc_version", "num_heads_q", "num_heads_k", "head_dim", "query_quant_mode",
                     "key_quant_mode", "batch_size", "max_seqlen_q", "max_seqlen_k", "layout_query", "layout_key", "sparse_count",
                     "sparse_mode", "is_fd", "pre_tokens", "next_tokens", "cmp_ratio"}),
      OP_INPUT(actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional), OP_OUTPUT(metaData),
      OP_ATTR(aicCoreNum, aivCoreNum, socVersion, numHeadsQ, numHeadsK, headDim, queryQuantMode, keyQuantMode,
              batchSizeOptional, maxSeqlenQOptional, maxSeqlenKOptional, layoutQueryOptional, layoutKeyOptional, 
              sparseCountOptional, sparseModeOptional, isFdOptional, preTokensOptional, nextTokensOptional, cmpRatioOptional));
  OP_CHECK(ret == ACL_SUCCESS,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                   "LightningIndexerQuantMetadata"
                   " ADD_TO_LAUNCHER_LIST_AICPU failed."),
           return nullptr);
  return metaData;
}
}  // namespace l0op
