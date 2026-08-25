/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef L0_GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_H
#define L0_GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_H

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor *GenericBlockSparseAttentionMetadata(
    const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount, const aclTensor *cuSeqLengthsOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *seqUsedQOptional, const aclTensor *seqUsedKvOptional,
    int64_t maxQSeqLen, int64_t numQHeads, int64_t numKvHeads, int64_t headDim, int64_t blockShapeX,
    int64_t blockShapeY, int64_t isPackedGQA, const char *qInputLayout, int64_t aicCoreNum, const aclTensor *metadata,
    aclOpExecutor *executor);

} // namespace l0op

#endif // L0_GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_H
