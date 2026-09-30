/**
 * Copyright (c) 2026 Tianjin University, Ltd.
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
#include "chunk_fwd_h.h"

#include <initializer_list>

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkFwdH);

namespace {

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

} // namespace

const std::array<const aclTensor *, 3> ChunkFwdH(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *g,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool outputFinalState,
    int64_t chunkSize,
    bool saveNewValue,
    bool useExp2,
    bool stateVFirst,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkFwdH, k, w, u, g, gkOptional, initialStateOptional, cuSeqlensOptional,
           chunkIndicesOptional, outputFinalState, chunkSize, saveNewValue, useExp2, stateVFirst,
           hOut, vNewOut, finalStateOut);

    const aclTensor *actualCuSeqlens = nullptr;
    if (cuSeqlensOptional) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        if (actualCuSeqlens == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert cuSeqlens to tensor failed.");
            return {nullptr, nullptr, nullptr};
        }
        const_cast<aclTensor *>(actualCuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualCuSeqlens = nullptr;
    }

    const aclTensor *actualChunkIndices = nullptr;
    if (chunkIndicesOptional) {
        actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        if (actualChunkIndices == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert chunkIndices to tensor failed.");
            return {nullptr, nullptr, nullptr};
        }
        const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    } else {
        actualChunkIndices = nullptr;
    }

    const auto &kShape = k->GetViewShape();
    const auto &uShape = u->GetViewShape();
    const int64_t logicalBatch = kShape.GetDim(0);
    const int64_t logicalKHeads = kShape.GetDim(1);
    const int64_t logicalSeqlen = kShape.GetDim(2);
    const int64_t logicalKDim = kShape.GetDim(3);
    const int64_t logicalVHeads = uShape.GetDim(1);
    const int64_t logicalVDim = uShape.GetDim(3);
    const aclTensor *finalStateOutKernel = finalStateOut;
    if (finalStateOutKernel == nullptr) {
        const DataType stateType = initialStateOptional == nullptr
                                       ? DataType::DT_FLOAT
                                       : initialStateOptional->GetDataType();
        finalStateOutKernel = executor->AllocTensor(MakeShape({0}), stateType, Format::FORMAT_ND);
        if (finalStateOutKernel == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Alloc finalStateOut placeholder failed.");
            return {nullptr, nullptr, nullptr};
        }
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ChunkFwdH,
        OP_INPUT(k, w, u, g, gkOptional, initialStateOptional, actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(hOut, vNewOut, finalStateOutKernel),
        OP_ATTR(outputFinalState, chunkSize, saveNewValue, useExp2, stateVFirst,
                logicalBatch, logicalSeqlen,
                logicalKHeads, logicalVHeads, logicalKDim, logicalVDim));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {hOut, vNewOut, finalStateOut};
}

} // namespace l0op
