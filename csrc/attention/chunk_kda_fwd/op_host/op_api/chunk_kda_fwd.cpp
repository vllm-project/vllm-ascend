/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "chunk_kda_fwd.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaFwd);

KdaCoreOutputs KdaChunkForward(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    bool safeGate,
    bool inputSequenceMajor,
    bool useGateInKernel,
    double lowerBound,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    const aclTensor *qgScaledOut,
    const aclTensor *uSeedOut,
    int64_t stage,
    aclOpExecutor *executor)
{
    L0_DFX(KdaChunkForward, q, k, v, g, beta, aLogOptional, dtBiasOptional,
           initialStateOptional, cuSeqlensOptional, chunkIndicesOptional,
           scale, chunkSize, safeGate, inputSequenceMajor, useGateInKernel,
           lowerBound, attnOut, finalStateOut, gkOut, aqkOut, akkOut,
           wOut, uOut, qgOut, kgOut, vNewOut, hOut, qgScaledOut, uSeedOut,
           stage);

    const aclTensor *actualCuSeqlens = nullptr;
    if (cuSeqlensOptional != nullptr) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualCuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const aclTensor *actualChunkIndices = nullptr;
    if (chunkIndicesOptional != nullptr) {
        actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        if (actualChunkIndices == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "failed to convert chunk metadata to tensor.");
            return {};
        }
        const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaFwd,
        OP_INPUT(q, k, v, g, beta, aLogOptional, dtBiasOptional,
                 initialStateOptional, actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(attnOut, finalStateOut, gkOut, aqkOut, akkOut, wOut, uOut,
                  qgOut, kgOut, vNewOut, hOut, qgScaledOut, uSeedOut),
        OP_ATTR(inputSequenceMajor ? "BSND" : "BNSD", scale, chunkSize,
                safeGate, static_cast<float>(lowerBound), useGateInKernel,
                false, stage));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaFwd failed.");
        return {};
    }
    return {attnOut, finalStateOut, gkOut, aqkOut, akkOut, wOut, uOut,
            qgOut, kgOut, vNewOut, hOut, qgScaledOut, uSeedOut};
}

} // namespace l0op
