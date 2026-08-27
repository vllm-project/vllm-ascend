/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "sparse_attention_score_metadata.h"

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GenericBlockSparseAttentionMetadata);

const aclTensor *GenericBlockSparseAttentionMetadata(
    const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount, const aclTensor *cuSeqLengthsOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *seqUsedQOptional, const aclTensor *seqUsedKvOptional,
    int64_t maxQSeqLen, int64_t numQHeads, int64_t numKvHeads, int64_t headDim, int64_t blockShapeX,
    int64_t blockShapeY, int64_t isPackedGQA, const char *qInputLayout, int64_t aicCoreNum, const aclTensor *metadata,
    aclOpExecutor *executor)
{
    static internal::AicpuTaskSpace space("GenericBlockSparseAttentionMetadata");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        GenericBlockSparseAttentionMetadata,
        OP_ATTR_NAMES({"max_q_seq_len", "num_q_heads", "num_kv_heads", "head_dim", "block_shape_x",
                       "block_shape_y", "is_packed_gqa", "q_input_layout", "aic_core_num"}),
        OP_INPUT(sparseBlockIdx, sparseBlockCount, cuSeqLengthsOptional, cuSeqLengthsKvOptional, seqUsedQOptional,
                 seqUsedKvOptional),
        OP_OUTPUT(metadata),
        OP_ATTR(maxQSeqLen, numQHeads, numKvHeads, headDim, blockShapeX, blockShapeY, isPackedGQA, qInputLayout,
                aicCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GenericBlockSparseAttentionMetadata AICPU launch registration failed."),
             return nullptr);
    return metadata;
}

} // namespace l0op
