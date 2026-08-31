/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_mla_with_kvcache.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FlashMlaWithKvcache);

const std::array<const aclTensor *, 2> FlashMlaWithKvcache(
    const aclTensor *q, const aclTensor *kCache,
    const aclTensor *blockTableOptional,
    const aclTensor *cacheSeqlensOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *attnMaskOptional,
    const aclTensor *metadataOptional,
    int32_t headDimV,
    double softmaxScale,
    int32_t maskMode,
    int32_t maxSeqlenQ,
    int32_t maxSeqlenKV,
    const char *layoutQ, const char *layoutKv, const char *layoutOut,
    int32_t returnSoftmaxLse,
    aclOpExecutor *executor)
{
    L0_DFX(FlashMlaWithKvcache, q, kCache, blockTableOptional, cacheSeqlensOptional, cuSeqlensQOptional,
           sequsedQOptional, attnMaskOptional, metadataOptional,
           headDimV, softmaxScale, maskMode, maxSeqlenQ, maxSeqlenKV, layoutQ, layoutKv, layoutOut,
           returnSoftmaxLse);

    if (blockTableOptional == nullptr) {
        blockTableOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (cacheSeqlensOptional == nullptr) {
        cacheSeqlensOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (cuSeqlensQOptional == nullptr) {
        cuSeqlensQOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (sequsedQOptional == nullptr) {
        sequsedQOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (attnMaskOptional == nullptr) {
        attnMaskOptional = executor->AllocTensor(DataType::DT_INT8, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (metadataOptional == nullptr) {
        metadataOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto attentionOutAlloc = executor->AllocTensor(q->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLseAlloc = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(FlashMlaWithKvcache,
                           OP_INPUT(q, kCache, blockTableOptional, cacheSeqlensOptional, cuSeqlensQOptional,
                                    sequsedQOptional, attnMaskOptional, metadataOptional),
                           OP_OUTPUT(attentionOutAlloc, softmaxLseAlloc),
                           OP_ATTR(headDimV, softmaxScale, maskMode, maxSeqlenQ, maxSeqlenKV,
                                   layoutQ, layoutKv, layoutOut, returnSoftmaxLse));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FlashMlaWithKvcache InferShape failed.");
        return {nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        FlashMlaWithKvcache,
        OP_INPUT(q, kCache, blockTableOptional, cacheSeqlensOptional, cuSeqlensQOptional,
                 sequsedQOptional, attnMaskOptional, metadataOptional),
        OP_OUTPUT(attentionOutAlloc, softmaxLseAlloc),
        OP_ATTR(headDimV, softmaxScale, maskMode, maxSeqlenQ, maxSeqlenKV,
                layoutQ, layoutKv, layoutOut, returnSoftmaxLse));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FlashMlaWithKvcache launch kernel failed.");
        return {nullptr, nullptr};
    }

    return {attentionOutAlloc, softmaxLseAlloc};
}

}  // namespace l0op