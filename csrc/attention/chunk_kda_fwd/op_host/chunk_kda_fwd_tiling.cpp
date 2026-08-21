/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "chunk_kda_fwd_tiling.h"
#include <algorithm>
#include <array>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_Q_IDX = 0;
constexpr size_t INPUT_V_IDX = 2;
constexpr size_t INPUT_GK_IDX = 3;
constexpr size_t INPUT_INITIAL_IDX = 5;
constexpr size_t INPUT_CU_SEQLENS_IDX = 6;
constexpr size_t INPUT_CHUNK_INDICES_IDX = 7;
constexpr size_t ATTR_SCALE_IDX = 0;
constexpr size_t ATTR_CHUNK_SIZE_IDX = 1;
constexpr size_t ATTR_OUTPUT_FINAL_STATE_IDX = 2;
constexpr size_t ATTR_TOTAL_CHUNKS_IDX = 3;
constexpr size_t ATTR_STAGE_IDX = 4;
constexpr uint64_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint64_t KDA_SCORE_QUEUE_SLOTS = 2;
constexpr uint64_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint64_t KDA_FP32_BYTES = sizeof(float);
constexpr uint64_t KDA_WORKSPACE_ALIGN = 512;

constexpr size_t DIM_B = 0;
constexpr size_t DIM_H = 1;
constexpr size_t DIM_T = 2;
constexpr size_t DIM_D = 3;

int64_t DTypeCode(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return 1;
    }
    if (dtype == ge::DT_FLOAT) {
        return 2;
    }
    return 0;
}
} // namespace

ge::graphStatus Tiling4ChunkKdaFwd(gert::TilingContext *context)
{
    ChunkKdaFwdTilingData tiling;

    auto qShape = context->GetOptionalInputShape(INPUT_Q_IDX)->GetStorageShape();
    auto vShape = context->GetOptionalInputShape(INPUT_V_IDX)->GetStorageShape();
    auto qDesc = context->GetInputDesc(INPUT_Q_IDX);
    auto gDesc = context->GetInputDesc(INPUT_GK_IDX);
    if (qDesc == nullptr || gDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrPtr = context->GetAttrs();
    if (attrPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    float scale = static_cast<float>(*(attrPtr->GetAttrPointer<double>(ATTR_SCALE_IDX)));
    int64_t chunkSize = *(attrPtr->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX));
    bool outputFinalState = *(attrPtr->GetAttrPointer<bool>(ATTR_OUTPUT_FINAL_STATE_IDX));
    int64_t totalChunks = *(attrPtr->GetAttrPointer<int64_t>(ATTR_TOTAL_CHUNKS_IDX));
    int64_t stage = 0;
    const int64_t *stagePtr = attrPtr->GetAttrPointer<int64_t>(ATTR_STAGE_IDX);
    if (stagePtr != nullptr) {
        stage = *stagePtr;
    }

    bool isVarLen = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX) != nullptr;
    int64_t batch = qShape.GetDim(DIM_B);
    int64_t seqNum = batch;
    std::array<int64_t, KDA_MAX_TILING_SEQUENCES> seqStart{};
    std::array<int64_t, KDA_MAX_TILING_SEQUENCES> seqEnd{};
    std::array<int64_t, KDA_MAX_TILING_SEQUENCE_OFFSETS> seqChunkOffset{};
    if (isVarLen) {
        auto cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX);
        seqNum = cuTensor->GetStorageShape().GetDim(0) - 1;
        auto chunkMetadata = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
        if (seqNum <= 0 || seqNum > KDA_MAX_TILING_SEQUENCES || chunkMetadata == nullptr ||
            chunkMetadata->GetStorageShape().GetShapeSize() != totalChunks * 4) {
            return ge::GRAPH_FAILED;
        }
        const int64_t *cu = cuTensor->GetData<int64_t>();
        if (cu == nullptr) {
            return ge::GRAPH_FAILED;
        }
        int64_t chunkOffset = 0;
        for (int64_t seq = 0; seq < seqNum; ++seq) {
            if (cu[seq] < 0 || cu[seq + 1] < cu[seq]) {
                return ge::GRAPH_FAILED;
            }
            seqStart[seq] = cu[seq];
            seqEnd[seq] = cu[seq + 1];
            seqChunkOffset[seq] = chunkOffset;
            const int64_t seqLength = cu[seq + 1] - cu[seq];
            chunkOffset += (seqLength + chunkSize - 1) / chunkSize;
        }
        seqChunkOffset[seqNum] = chunkOffset;
        if (chunkOffset != totalChunks) {
            return ge::GRAPH_FAILED;
        }
    }
    bool hasInitialState = context->GetOptionalInputTensor(INPUT_INITIAL_IDX) != nullptr;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();
    int64_t taskNum = seqNum * vShape.GetDim(DIM_H);
    if (stage == 1 || stage == 2 || stage == 3) {
        taskNum = (isVarLen ? totalChunks : batch * totalChunks) * vShape.GetDim(DIM_H);
    }
    uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(taskNum, coreNum));
    if (stage == 1 || stage == 2 || stage == 3 ||
        (qDesc->GetDataType() != ge::DT_FLOAT && qShape.GetDim(DIM_D) >= 16)) {
        blockDim = coreNum;
    }
    context->SetBlockDim(blockDim == 0 ? 1 : blockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    uint64_t kernelScratch = 0;
    if (stage == 1) {
        const uint64_t usedCoreNum = static_cast<uint64_t>(blockDim == 0 ? 1 : blockDim);
        const uint64_t solveScratch = usedCoreNum * KDA_SOLVE_SCRATCH_SLOTS *
                                      static_cast<uint64_t>(chunkSize) * static_cast<uint64_t>(chunkSize) *
                                      KDA_FP32_BYTES;
        const uint64_t alignedSolveScratch =
            (solveScratch + KDA_WORKSPACE_ALIGN - 1) / KDA_WORKSPACE_ALIGN * KDA_WORKSPACE_ALIGN;
        const uint64_t scoreElementBytes = qDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
        const uint64_t scoreScratch = usedCoreNum * KDA_SCORE_QUEUE_SLOTS * KDA_SCORE_SCRATCH_PLANES *
                                      static_cast<uint64_t>(chunkSize) *
                                      static_cast<uint64_t>(qShape.GetDim(DIM_D)) * scoreElementBytes;
        kernelScratch = alignedSolveScratch + scoreScratch;
    } else if (stage == 2) {
        const uint64_t outputElements = static_cast<uint64_t>(batch) *
                                        static_cast<uint64_t>(vShape.GetDim(DIM_H)) *
                                        static_cast<uint64_t>(qShape.GetDim(DIM_T)) *
                                        static_cast<uint64_t>(vShape.GetDim(DIM_D));
        kernelScratch = 2 * outputElements * KDA_FP32_BYTES;
    }
    kernelScratch = (kernelScratch + KDA_WORKSPACE_ALIGN - 1) / KDA_WORKSPACE_ALIGN * KDA_WORKSPACE_ALIGN;
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + kernelScratch;

    tiling.set_batch(batch);
    tiling.set_seqNum(seqNum);
    tiling.set_qHeadNum(qShape.GetDim(DIM_H));
    tiling.set_vHeadNum(vShape.GetDim(DIM_H));
    tiling.set_seqlen(qShape.GetDim(DIM_T));
    tiling.set_kHeadDim(qShape.GetDim(DIM_D));
    tiling.set_vHeadDim(vShape.GetDim(DIM_D));
    tiling.set_chunkSize(chunkSize);
    tiling.set_totalChunks(totalChunks);
    tiling.set_scale(scale);
    tiling.set_hasInitialState(hasInitialState);
    tiling.set_outputFinalState(outputFinalState);
    tiling.set_isVarLen(isVarLen);
    tiling.set_dataType(DTypeCode(qDesc->GetDataType()));
    tiling.set_gateDataType(DTypeCode(gDesc->GetDataType()));
    tiling.set_usedCoreNum(blockDim == 0 ? 1 : blockDim);
    tiling.set_stage(stage);
    tiling.set_seqStart(seqStart.data());
    tiling.set_seqEnd(seqEnd.data());
    tiling.set_seqChunkOffset(seqChunkOffset.data());

    if (qDesc->GetDataType() == ge::DT_FLOAT) {
        context->SetTilingKey(0);
    } else if (qShape.GetDim(DIM_D) < 16) {
        context->SetTilingKey(2);
    } else {
        context->SetTilingKey(1);
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaFwd(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaFwd)
    .Tiling(Tiling4ChunkKdaFwd)
    .TilingParse<ChunkKdaFwdCompileInfo>(TilingPrepare4ChunkKdaFwd);

} // namespace optiling
