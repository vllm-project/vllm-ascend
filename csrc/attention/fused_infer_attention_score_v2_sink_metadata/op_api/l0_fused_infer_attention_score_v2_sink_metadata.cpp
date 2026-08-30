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
 * \file l0_fused_infer_attention_score_v2_sink_metadata.cpp
 * \brief
 */

#include "l0_fused_infer_attention_score_v2_sink_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace FusedInferAttentionScoreV2SinkMetadatal0op {
OP_TYPE_REGISTER(FusedInferAttentionScoreV2SinkMetadata);

const aclTensor *FusedInferAttentionScoreV2SinkMetadata(const aclTensor *actualSeqLengthsOptional,
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
                                                        const char *socVersion,
                                                        int64_t aicCoreNum,
                                                        int64_t aivCoreNum,
                                                        const aclTensor *metaData,
                                                        aclOpExecutor *executor)
{
    L0_DFX(FusedInferAttentionScoreV2SinkMetadata,
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
           aicCoreNum,
           aivCoreNum,
           metaData);

    static internal::AicpuTaskSpace space("FusedInferAttentionScoreV2SinkMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        FusedInferAttentionScoreV2SinkMetadata,
        OP_ATTR_NAMES({"num_heads_q", "num_heads_kv", "head_dim_qk", "head_dim_v", "batch_size", "sparse_mode",
                    "pre_tokens", "next_tokens", "input_layout", "input_layout_kv", "sink_num", "k_sink_num",
                    "batch_invariant", "rope_head_dim", "block_size", "soc_version", "aic_core_num", "aiv_core_num"}),
        OP_INPUT(actualSeqLengthsOptional, actualSeqLengthsKvOptional),
        OP_OUTPUT(metaData),
        OP_ATTR(numHeadsQ, numHeadsKv, headDimQk, headDimV, batchSizeOptional, sparseModeOptional, preTokensOptional,
                nextTokensOptional, inputLayoutOptional, inputLayoutKvOptional, sinkNumOptional, kSinkNumOptional,
                batchInvariantOptional, ropeHeadDimOptional, blockSizeOptional, socVersion, aicCoreNum, aivCoreNum));
        OP_CHECK(ret == ACL_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                   "FusedInferAttentionScoreV2SinkMetadata"
                   " ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return metaData;
}

} // namespace l0op
