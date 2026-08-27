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
#include "chunk_fwd_o.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkFwdO);

const std::array<const aclTensor *, 1> ChunkFwdO(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *h,
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkOffsetsOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *oOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkFwdO, q, k, v, h, g, cuSeqlensOptional, chunkOffsetsOptional, scale, chunkSize, oOut);

    const aclTensor *actualCuSeqlens = nullptr;
    if (cuSeqlensOptional) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualCuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualCuSeqlens = nullptr;
    }

    const aclTensor *actualChunkOffsets = nullptr;
    if (chunkOffsetsOptional) {
        actualChunkOffsets = executor->ConvertToTensor(chunkOffsetsOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualChunkOffsets)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkOffsets)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkOffsets)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualChunkOffsets = nullptr;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ChunkFwdO,
        OP_INPUT(q, k, v, h, g, actualCuSeqlens, actualChunkOffsets),
        OP_OUTPUT(oOut),
        OP_ATTR(scale, chunkSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr};
    }
    return {oOut};
}

} // namespace l0op
