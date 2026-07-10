/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sparse_attention_score.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SparseAttentionScore);

const std::array<const aclTensor *, 2> SparseAttentionScore(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *selectIdx,
    const aclTensor *blockTable,
    const aclTensor *selectNumIdxOptional,
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *qDequantScaleOptional,
    const aclTensor *kDequantScaleOptional,
    const aclTensor *vDequantScaleOptional,
    int64_t numKeyValueHeads,
    double scaleValue,
    int64_t blockSize,
    int64_t topK,
    int64_t innerPrecise,
    aclOpExecutor *executor)
{
    L0_DFX(SparseAttentionScore, query, key, value, selectIdx, blockTable,
           selectNumIdxOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional,
           qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional,
           numKeyValueHeads, scaleValue, blockSize, topK, innerPrecise);

    DataType attentionOutDtype = query->GetDataType() == DataType::DT_FLOAT8_E4M3FN ?
        DataType::DT_FLOAT16 : query->GetDataType();
    auto attentionOutTensor = executor->AllocTensor(attentionOutDtype, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLseTensor = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(SparseAttentionScore,
                           OP_INPUT(query, key, value, selectIdx, blockTable,
                                    selectNumIdxOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional,
                                    qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional),
                           OP_OUTPUT(attentionOutTensor, softmaxLseTensor),
                           OP_ATTR(static_cast<int64_t>(numKeyValueHeads),
                                   static_cast<float>(scaleValue),
                                   static_cast<int64_t>(blockSize),
                                   static_cast<int64_t>(topK),
                                   static_cast<int64_t>(innerPrecise)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SparseAttentionScore infer shape failed.");
        return {nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(SparseAttentionScore,
                                OP_INPUT(query, key, value, selectIdx, blockTable,
                                         selectNumIdxOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional,
                                         qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional),
                                OP_OUTPUT(attentionOutTensor, softmaxLseTensor),
                                OP_ATTR(static_cast<int64_t>(numKeyValueHeads),
                                        static_cast<float>(scaleValue),
                                        static_cast<int64_t>(blockSize),
                                        static_cast<int64_t>(topK),
                                        static_cast<int64_t>(innerPrecise)));

    return {attentionOutTensor, softmaxLseTensor};
}

} // namespace l0op
