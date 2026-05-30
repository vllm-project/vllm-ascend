/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_fwd_o_tiling.cpp
 * \brief
 */

#include "chunk_fwd_o_tiling.h"
#include <register/op_impl_registry.h>
#include "../tiling_base/data_copy_transpose_tiling.h"
#include "../tiling_base/tiling_templates_registry.h"

namespace optiling {
static constexpr size_t INPUT_Q_IDX = 0;
static constexpr size_t INPUT_K_IDX = 1;
static constexpr size_t INPUT_V_IDX = 2;
static constexpr size_t INPUT_H_IDX = 3;
static constexpr size_t INPUT_G_IDX = 4;
static constexpr size_t INPUT_SEQLENS_IDX = 5;
static constexpr size_t INPUT_CHUNK_INDICES_IDX = 6;

static constexpr size_t ATTR_SCALE_IDX = 0;
static constexpr size_t ATTR_CHUNK_SIZE_IDX = 1;
static constexpr size_t ATTR_K_STRIDE_IDX = 2;
static constexpr size_t ATTR_V_STRIDE_IDX = 3;

static constexpr size_t DIM_BATCH = 0;
static constexpr size_t DIM_HEAD_NUM = 1;
static constexpr size_t DIM_SEQLEN = 2;
static constexpr size_t DIM_HEAD_DIM = 3;


static void ChunkFwdOTilingDataPrint(gert::TilingContext *context, ChunkFwdOTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print ChunkFwdO tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== batch: %ld", tiling.get_shapeBatch());
    OP_LOGD(nodeName, "=== seqlen: %ld", tiling.get_seqlen());
    OP_LOGD(nodeName, "=== kNumHead: %ld", tiling.get_kNumHead());
    OP_LOGD(nodeName, "=== vNumHead: %ld", tiling.get_vNumHead());
    OP_LOGD(nodeName, "=== kHeadDim: %ld", tiling.get_kHeadDim());
    OP_LOGD(nodeName, "=== vHeadDim: %ld", tiling.get_vHeadDim());
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.get_chunkSize());
    OP_LOGD(nodeName, "=== dataType: %ld", tiling.get_dataType());
    OP_LOGD(nodeName, "=== isVariedLen: %ld", tiling.get_isVariedLen());
    OP_LOGD(nodeName, "=== tokenBatch: %f", tiling.get_tokenBatch());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print ChunkFwdO tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus Tiling4ChunkFwdO(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkFwdO start.");
    ChunkFwdOTilingData tiling;
    
    gert::Shape qStorageShape = context->GetOptionalInputShape(INPUT_Q_IDX)->GetStorageShape();
    gert::Shape vStorageShape = context->GetOptionalInputShape(INPUT_V_IDX)->GetStorageShape();

    int64_t seqlen = qStorageShape.GetDim(DIM_SEQLEN);
    int64_t kNumHead = qStorageShape.GetDim(DIM_HEAD_NUM);
    int64_t vNumHead = vStorageShape.GetDim(DIM_HEAD_NUM);
    int64_t kHeadDim = qStorageShape.GetDim(DIM_HEAD_DIM);
    int64_t vHeadDim = vStorageShape.GetDim(DIM_HEAD_DIM);
    int64_t isVariedLen, shapeBatch, tokenBatch;
    
    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_SEQLENS_IDX);
    if (cuSeqlensTensor == nullptr) {
        isVariedLen = false;
        shapeBatch = qStorageShape.GetDim(DIM_BATCH);
        tokenBatch = 1;
    } else {
        isVariedLen = true;
        shapeBatch = 1;
        tokenBatch = cuSeqlensTensor->GetStorageShape().GetDim(DIM_BATCH) - 1;
    }
    
    auto attrPtr = context->GetAttrs();
    float scale = *(attrPtr->GetAttrPointer<double>(ATTR_SCALE_IDX));
    int64_t kStride0 = *(attrPtr->GetAttrPointer<int64_t>(ATTR_K_STRIDE_IDX));
    int64_t vStride0 = *(attrPtr->GetAttrPointer<int64_t>(ATTR_V_STRIDE_IDX));
    int64_t chunkSize = *(attrPtr->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX));

    auto dtype = context->GetInputTensor(0)->GetDataType();
    uint64_t dataType = dtype == ge::DT_BF16 ? 1 : 0;

    auto gDType = context->GetOptionalInputTensor(INPUT_G_IDX)->GetDataType();
    int64_t gDataType = 2;
    if (gDType == ge::DT_BF16) {
        gDataType = 1;
    } else if (gDType == ge::DT_FLOAT16) {
        gDataType = 0;
    }
   
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
    context->SetBlockDim(aicCoreNum);

    constexpr size_t WORKSPACE_RSV_BYTE = 16 * 1024 * 1024;
    constexpr size_t GM_ALIGN = 512;
    int64_t pingpongStages = 2;

    size_t workspaceOffset = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceOffset += WORKSPACE_RSV_BYTE;
    
    tiling.set_vWorkspaceOffset(workspaceOffset);
    workspaceOffset += (aicCoreNum * chunkSize * vHeadDim * sizeof(float) * pingpongStages + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tiling.set_hWorkspaceOffset(workspaceOffset);
    workspaceOffset += (aicCoreNum * chunkSize * vHeadDim * sizeof(float) * pingpongStages + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tiling.set_attnWorkspaceOffset(workspaceOffset);
    workspaceOffset += (aicCoreNum * chunkSize * chunkSize * sizeof(float) * pingpongStages + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tiling.set_aftermaskWorkspaceOffset(workspaceOffset);
    workspaceOffset += (aicCoreNum * chunkSize * chunkSize * sizeof(float) * pingpongStages + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    tiling.set_maskWorkspaceOffset(workspaceOffset);
    workspaceOffset += (chunkSize * chunkSize + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    workspaceOffset += WORKSPACE_RSV_BYTE;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = (workspaceOffset - 0);

    tiling.set_shapeBatch(shapeBatch);
    tiling.set_seqlen(seqlen);
    tiling.set_kNumHead(kNumHead);
    tiling.set_vNumHead(vNumHead);
    tiling.set_kHeadDim(kHeadDim);
    tiling.set_vHeadDim(vHeadDim);
    tiling.set_scale(scale);
    tiling.set_kStride0(kStride0);
    tiling.set_vStride0(vStride0);
    tiling.set_chunkSize(chunkSize);
    tiling.set_isVariedLen(isVariedLen);
    tiling.set_tokenBatch(tokenBatch);
    tiling.set_dataType(dataType);
    tiling.set_gDataType(gDataType);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    ChunkFwdOTilingDataPrint(context, tiling);
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkFwdO end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkFwdO(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkFwdO)
    .Tiling(Tiling4ChunkFwdO)
    .TilingParse<ChunkFwdOCompileInfo>(TilingPrepareForChunkFwdO);

} // namespace optiling
