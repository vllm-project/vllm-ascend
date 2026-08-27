/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_H_
#define GENERIC_BLOCK_SPARSE_ATTENTION_H_

#include <array>
#include "opdev/op_executor.h"

namespace l0op {

const std::array<const aclTensor *, 2> GenericBlockSparseAttention(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseBlockIdx,
    const aclTensor *sparseBlockCount,
    const aclTensor *metadataOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *qDequantScaleOptional,
    const aclTensor *kDequantScaleOptional,
    const aclTensor *vDequantScaleOptional,
    const aclTensor *pQuantScaleOptional,
    const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    const aclTensor *blockTableOptional,
    const aclIntArray *blockShape,
    int64_t isPackedGQA,
    const char *layoutQ,
    const char *layoutKv,
    double scaleValue,
    int64_t maskType,
    int64_t quantType,
    double dstTypeMax,
    int64_t softmaxPrecision,
    int64_t winLeft,
    int64_t winRight,
    int64_t returnSoftmaxlse,
    const aclTensor *attentionOut,
    aclOpExecutor *executor);

}  // namespace l0op

#endif  // GENERIC_BLOCK_SPARSE_ATTENTION_H_
