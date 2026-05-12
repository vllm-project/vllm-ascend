/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "chunk_gated_delta_rule_fwd_h.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkGatedDeltaRuleFwdH);

const std::array<const aclTensor *, 3> ChunkGatedDeltaRuleFwdH(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *g,
    const aclTensor *initalStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool outputFinalState,
    int64_t chunkSize,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkGatedDeltaRuleFwdH, k, w, u, g, initalStateOptional, cuSeqlensOptional, chunkIndicesOptional, outputFinalState, chunkSize, hOut, vNewOut, finalStateOut);

    const aclTensor *actualCuSeqlens = nullptr;
    if (cuSeqlensOptional) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualCuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualCuSeqlens = nullptr;
    }

    const aclTensor *actualChunkIndices = nullptr;
    if (chunkIndicesOptional) {
        actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualChunkIndices = nullptr;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ChunkGatedDeltaRuleFwdH,
        OP_INPUT(k, w, u, g, initalStateOptional, actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(hOut, vNewOut, finalStateOut),
        OP_ATTR(outputFinalState, chunkSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {hOut, vNewOut, finalStateOut};
}

} // namespace l0op
