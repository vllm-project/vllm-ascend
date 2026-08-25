/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generic_block_sparse_attention.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GenericBlockSparseAttention);

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
    aclOpExecutor *executor)
{
    L0_DFX(GenericBlockSparseAttention, query, key, value, sparseBlockIdx, sparseBlockCount,
           metadataOptional, attenMaskOptional, qDequantScaleOptional, kDequantScaleOptional,
           vDequantScaleOptional, pQuantScaleOptional, cuSeqLengthsQOptional, cuSeqLengthsKvOptional,
           sequsedQOptional, sequsedKvOptional, blockTableOptional,
           blockShape, isPackedGQA, layoutQ, layoutKv, scaleValue,
           maskType, quantType, dstTypeMax, softmaxPrecision, winLeft, winRight, returnSoftmaxlse);

    DataType outDtype = (query->GetDataType() == DataType::DT_FLOAT8_E4M3FN)
                            ? attentionOut->GetDataType()
                            : query->GetDataType();
    auto attentionOutTensor = executor->AllocTensor(outDtype, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLseTensor = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(GenericBlockSparseAttention,
                           OP_INPUT(query, key, value, sparseBlockIdx, sparseBlockCount,
                                    metadataOptional, attenMaskOptional, qDequantScaleOptional,
                                    kDequantScaleOptional, vDequantScaleOptional, pQuantScaleOptional,
                                    cuSeqLengthsQOptional, cuSeqLengthsKvOptional,
                                    sequsedQOptional, sequsedKvOptional, blockTableOptional),
                           OP_OUTPUT(attentionOutTensor, softmaxLseTensor),
                           OP_ATTR(blockShape,
                                   static_cast<int64_t>(isPackedGQA),
                                   layoutQ,
                                   layoutKv,
                                   static_cast<float>(scaleValue),
                                   static_cast<int64_t>(maskType),
                                   static_cast<int64_t>(quantType),
                                   static_cast<float>(dstTypeMax),
                                   static_cast<int64_t>(softmaxPrecision),
                                   static_cast<int64_t>(winLeft),
                                   static_cast<int64_t>(winRight),
                                   static_cast<int64_t>(returnSoftmaxlse)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GenericBlockSparseAttention infer shape failed.");
        return {nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(GenericBlockSparseAttention,
                                OP_INPUT(query, key, value, sparseBlockIdx, sparseBlockCount,
                                         metadataOptional, attenMaskOptional, qDequantScaleOptional,
                                         kDequantScaleOptional, vDequantScaleOptional, pQuantScaleOptional,
                                         cuSeqLengthsQOptional, cuSeqLengthsKvOptional,
                                         sequsedQOptional, sequsedKvOptional, blockTableOptional),
                                OP_OUTPUT(attentionOutTensor, softmaxLseTensor),
                                OP_ATTR(blockShape,
                                        static_cast<int64_t>(isPackedGQA),
                                        layoutQ,
                                        layoutKv,
                                        static_cast<float>(scaleValue),
                                        static_cast<int64_t>(maskType),
                                        static_cast<int64_t>(quantType),
                                        static_cast<float>(dstTypeMax),
                                        static_cast<int64_t>(softmaxPrecision),
                                        static_cast<int64_t>(winLeft),
                                        static_cast<int64_t>(winRight),
                                        static_cast<int64_t>(returnSoftmaxlse)));

    return {attentionOutTensor, softmaxLseTensor};
}

}  // namespace l0op
