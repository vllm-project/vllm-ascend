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
 * \file l0_sparse_attn_sharedkv_metadata.cpp
 * \brief
 */

#include "l0_sparse_attn_sharedkv_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(SparseAttnSharedkvMetadata);

const aclTensor* SparseAttnSharedkvMetadata(
    const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensOriKvOptional,
    const aclTensor* cuSeqLensCmpKvOptional, const aclTensor* sequsedQOptional, const aclTensor* sequsedKvOptional,
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
    int64_t oriTopK, int64_t cmpTopK, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft,
    int64_t oriWinRight, char *layoutQOptional, char *layoutKvOptional, bool hasOriKv, bool hasCmpKv,
    const char *socVersion, int64_t aicCoreNum, int64_t aivCoreNum, const aclTensor* metadata, aclOpExecutor* executor)
{
    L0_DFX(SparseAttnSharedkvMetadata, cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional,
           sequsedQOptional, sequsedKvOptional, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenKv,
           oriTopK, cmpTopK, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional,
           layoutKvOptional, hasOriKv, hasCmpKv, socVersion, aicCoreNum, aivCoreNum, metadata);

    static internal::AicpuTaskSpace space(
        "SparseAttnSharedkvMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        SparseAttnSharedkvMetadata,
        OP_ATTR_NAMES({"num_heads_q", "num_heads_kv", "head_dim", "batch_size", "max_seqlen_q", "max_seqlen_kv",
                       "ori_topk", "cmp_topk", "cmp_ratio", "ori_mask_mode", "cmp_mask_mode", "ori_win_left",
                       "ori_win_right", "layout_q", "layout_kv", "has_ori_kv", "has_cmp_kv", "soc_version",
                       "aic_core_num", "aiv_core_num"}),
        OP_INPUT(cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional,
                 sequsedKvOptional),
        OP_OUTPUT(metadata),
        OP_ATTR(numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenKv, oriTopK, cmpTopK, cmpRatio,
                oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional, hasOriKv,
                hasCmpKv, socVersion, aicCoreNum, aivCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SparseAttnSharedkvMetadata ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return metadata;
}

}  // namespace l0op
