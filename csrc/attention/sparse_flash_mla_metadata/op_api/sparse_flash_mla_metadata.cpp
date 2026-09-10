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
 * \file sparse_flash_mla_metadata.cpp
 * \brief
 */

#include "sparse_flash_mla_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(SparseFlashMlaMetadata);

const aclTensor *SparseFlashMlaMetadata(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional, const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional, const aclTensor *cmpTopkLengthOptional, int64_t numHeadsQ,
    int64_t numHeadsKv, int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv,
    int64_t maxSeqlenCmpKv, int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode,
    int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, const char *layoutQOptional,
    const char *layoutKvOptional, bool hasOriKv, bool hasCmpKv, const char *socVersion, int64_t aicCoreNum,
    int64_t aivCoreNum, const aclTensor *metaData, aclOpExecutor *executor)
{
    L0_DFX(SparseFlashMlaMetadata, cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional,
           sequsedQOptional, sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional,
           cmpTopkLengthOptional, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv,
           oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional,
           layoutKvOptional, hasOriKv, hasCmpKv, socVersion, aicCoreNum, aivCoreNum, metaData);

    static internal::AicpuTaskSpace space("SparseFlashMlaMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        SparseFlashMlaMetadata,
        OP_ATTR_NAMES({"num_heads_q", "num_heads_kv", "head_dim", "batch_size", "max_seqlen_q", "max_seqlen_ori_kv",
                       "max_seqlen_cmp_kv", "ori_topk", "cmp_topk", "cmp_ratio", "ori_mask_mode", "cmp_mask_mode",
                       "ori_win_left", "ori_win_right", "layout_q", "layout_kv", "has_ori_kv", "has_cmp_kv",
                       "soc_version", "aic_core_num", "aiv_core_num"}),
        OP_INPUT(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional,
                 sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional,
                 cmpTopkLengthOptional),
        OP_OUTPUT(metaData),
        OP_ATTR(numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv, oriTopk, cmpTopk,
                cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional,
                hasOriKv, hasCmpKv, socVersion, aicCoreNum, aivCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SparseFlashMlaMetadata ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return metaData;
}

} // namespace l0op