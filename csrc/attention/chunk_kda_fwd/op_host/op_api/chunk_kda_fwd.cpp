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

#include <vector>

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaFwd);

namespace {
const aclIntArray *BuildPackedChunkMetadata(const aclIntArray *cuSeqlens,
                                            const aclIntArray *chunkIndices,
                                            int64_t chunkSize,
                                            int64_t totalChunks,
                                            aclOpExecutor *executor)
{
    if (cuSeqlens == nullptr || cuSeqlens->Size() < 2 || chunkSize <= 0 || totalChunks <= 0) {
        return nullptr;
    }

    const aclIntArray &cu = *cuSeqlens;
    std::vector<int64_t> packed;
    packed.reserve(static_cast<size_t>(totalChunks) * 4);
    auto appendChunk = [&](int64_t seq, int64_t localChunk) -> bool {
        if (seq < 0 || static_cast<size_t>(seq + 1) >= cu.Size() || localChunk < 0) {
            return false;
        }
        int64_t seqStart = cu[static_cast<size_t>(seq)];
        int64_t seqEnd = cu[static_cast<size_t>(seq + 1)];
        int64_t start = seqStart + localChunk * chunkSize;
        if (start < seqStart || start >= seqEnd) {
            return false;
        }
        int64_t end = start + chunkSize;
        if (end > seqEnd) {
            end = seqEnd;
        }
        packed.insert(packed.end(), {seq, start, end, 0});
        return true;
    };

    if (chunkIndices != nullptr) {
        if (chunkIndices->Size() != static_cast<size_t>(totalChunks) * 2) {
            return nullptr;
        }
        for (size_t idx = 0; idx < chunkIndices->Size(); idx += 2) {
            if (!appendChunk((*chunkIndices)[idx], (*chunkIndices)[idx + 1])) {
                return nullptr;
            }
        }
    } else {
        for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
            int64_t seqLength = cu[seq + 1] - cu[seq];
            int64_t chunkCount = (seqLength + chunkSize - 1) / chunkSize;
            for (int64_t localChunk = 0; localChunk < chunkCount; ++localChunk) {
                if (!appendChunk(static_cast<int64_t>(seq), localChunk)) {
                    return nullptr;
                }
            }
        }
    }
    if (packed.size() != static_cast<size_t>(totalChunks) * 4) {
        return nullptr;
    }
    return executor->AllocIntArray(packed.data(), packed.size());
}
} // namespace

const std::array<const aclTensor *, 10> ChunkKdaFwd(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *gk,
    const aclTensor *beta,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const aclTensor *stageQGInputOptional,
    const aclTensor *stageAqkInputOptional,
    const aclTensor *stageVNewInputOptional,
    const aclTensor *stageHInputOptional,
    double scale,
    int64_t chunkSize,
    bool outputFinalState,
    int64_t totalChunks,
    int64_t stage,
    const aclTensor *oOut,
    const aclTensor *finalStateOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaFwd, q, k, v, gk, beta, initialStateOptional, cuSeqlensOptional, chunkIndicesOptional,
           stageQGInputOptional, stageAqkInputOptional, stageVNewInputOptional, stageHInputOptional, scale, chunkSize,
           outputFinalState, totalChunks, stage, oOut, finalStateOut, aqkOut, akkOut, wOut, uOut, qgOut, kgOut,
           vNewOut, hOut);

    const aclTensor *actualCuSeqlens = nullptr;
    if (cuSeqlensOptional != nullptr) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualCuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const aclTensor *actualChunkIndices = nullptr;
    if (cuSeqlensOptional != nullptr) {
        const aclIntArray *packedChunkMetadata = BuildPackedChunkMetadata(
            cuSeqlensOptional, chunkIndicesOptional, chunkSize, totalChunks, executor);
        if (packedChunkMetadata == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "failed to build packed chunk metadata.");
            return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        }
        actualChunkIndices = executor->ConvertToTensor(packedChunkMetadata, DataType::DT_INT64);
        if (actualChunkIndices == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "failed to convert packed chunk metadata to tensor.");
            return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        }
        const_cast<aclTensor *>(actualChunkIndices)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualChunkIndices)->SetOriginalFormat(Format::FORMAT_ND);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaFwd,
        OP_INPUT(q, k, v, gk, beta, initialStateOptional, actualCuSeqlens, actualChunkIndices,
                 stageQGInputOptional, stageAqkInputOptional, stageVNewInputOptional, stageHInputOptional),
        OP_OUTPUT(oOut, finalStateOut, aqkOut, akkOut, wOut, uOut, qgOut, kgOut, vNewOut, hOut),
        OP_ATTR(scale, chunkSize, outputFinalState, totalChunks, stage));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaFwd failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {oOut, finalStateOut, aqkOut, akkOut, wOut, uOut, qgOut, kgOut, vNewOut, hOut};
}

} // namespace l0op
