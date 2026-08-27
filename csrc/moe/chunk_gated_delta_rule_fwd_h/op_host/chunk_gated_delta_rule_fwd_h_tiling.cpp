/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_fwd_h_tiling.cpp
 * \brief
 */

#include "chunk_gated_delta_rule_fwd_h_tiling.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "chunk_gated_delta_rule_fwd_h_tiling_processor.h"

namespace optiling {

// Maps a ge::DataType to the {fp16:0, bf16:1, fp32:2} convention shared with the kernel.
static int64_t GdnFwdHDtypeToEnum(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return GDN_FWD_H_DTYPE_BF16;
    }
    if (dtype == ge::DT_FLOAT16) {
        return GDN_FWD_H_DTYPE_FP16;
    }
    return GDN_FWD_H_DTYPE_FP32;
}
static constexpr size_t INPUT_K_IDX = 0;
static constexpr size_t INPUT_W_IDX = 1;
static constexpr size_t INPUT_U_IDX = 2;
static constexpr size_t INPUT_G_IDX = 3;
static constexpr size_t INPUT_INITIAL_STATE_IDX = 4;
static constexpr size_t INPUT_SEQLENS_IDX = 5;
static constexpr size_t INPUT_CHUNK_INDICES_IDX = 6;

static constexpr size_t ATTR_STORE_FINAL_STATE_IDX = 0;
static constexpr size_t ATTR_CHUNK_SIZE_IDX = 1;

static constexpr size_t DIM_BATCH = 0;
static constexpr size_t DIM_HEAD_NUM = 1;
static constexpr size_t DIM_SEQLEN = 2;
static constexpr size_t DIM_HEAD_DIM = 3;


static void ChunkGatedDeltaRuleFwdHTilingDataPrint(gert::TilingContext *context, ChunkGatedDeltaRuleFwdHTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print ChunkGatedDeltaRuleFwdH tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== batch: %ld", tiling.get_batch());
    OP_LOGD(nodeName, "=== seqlen: %ld", tiling.get_seqlen());
    OP_LOGD(nodeName, "=== kNumHead: %ld", tiling.get_kNumHead());
    OP_LOGD(nodeName, "=== vNumHead: %ld", tiling.get_vNumHead());
    OP_LOGD(nodeName, "=== kHeadDim: %ld", tiling.get_kHeadDim());
    OP_LOGD(nodeName, "=== vHeadDim: %ld", tiling.get_vHeadDim());
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.get_chunkSize());
    OP_LOGD(nodeName, "=== useInitialState: %ld", tiling.get_useInitialState());
    OP_LOGD(nodeName, "=== storeFinalState: %ld", tiling.get_storeFinalState());
    OP_LOGD(nodeName, "=== dataType: %ld", tiling.get_dataType());
    OP_LOGD(nodeName, "=== isVariedLen: %ld", tiling.get_isVariedLen());
    OP_LOGD(nodeName, "=== shapeBatch: %ld", tiling.get_shapeBatch());
    OP_LOGD(nodeName, "=== tokenBatch: %f", tiling.get_tokenBatch());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print ChunkGatedDeltaRuleFwdH tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdH(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkGatedDeltaRuleFwdH start.");
    ChunkGatedDeltaRuleFwdHTilingData tiling;

    gert::Shape kStorageShape = context->GetOptionalInputShape(INPUT_K_IDX)->GetStorageShape();
    gert::Shape uStorageShape = context->GetOptionalInputShape(INPUT_U_IDX)->GetStorageShape();

    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_SEQLENS_IDX);
    auto initialStateTensor = context->GetOptionalInputTensor(INPUT_INITIAL_STATE_IDX);
    bool useInitialState = initialStateTensor != nullptr;

    auto attrPtr = context->GetAttrs();
    bool storeFinalState = *(attrPtr->GetAttrPointer<bool>(ATTR_STORE_FINAL_STATE_IDX));
    int64_t chunkSize = *(attrPtr->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX));

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    ChunkGatedDeltaRuleFwdHTilingContext tilingCtx{};
    tilingCtx.seqlen = kStorageShape.GetDim(DIM_SEQLEN);
    tilingCtx.kNumHead = kStorageShape.GetDim(DIM_HEAD_NUM);
    tilingCtx.kHeadDim = kStorageShape.GetDim(DIM_HEAD_DIM);
    tilingCtx.vNumHead = uStorageShape.GetDim(DIM_HEAD_NUM);
    tilingCtx.vHeadDim = uStorageShape.GetDim(DIM_HEAD_DIM);
    tilingCtx.shapeBatchDim = kStorageShape.GetDim(DIM_BATCH);
    tilingCtx.hasCuSeqlens = cuSeqlensTensor != nullptr;
    tilingCtx.cuSeqlensDim0 =
        cuSeqlensTensor != nullptr ? cuSeqlensTensor->GetStorageShape().GetDim(DIM_BATCH) : 0;
    tilingCtx.dataType = GdnFwdHDtypeToEnum(context->GetInputTensor(0)->GetDataType());
    tilingCtx.gDataType = GdnFwdHDtypeToEnum(context->GetOptionalInputTensor(INPUT_G_IDX)->GetDataType());
    tilingCtx.useInitialState = useInitialState;
    tilingCtx.stateDataType =
        useInitialState ? GdnFwdHDtypeToEnum(initialStateTensor->GetDataType()) : GDN_FWD_H_DTYPE_FP32;
    tilingCtx.storeFinalState = storeFinalState;
    tilingCtx.chunkSize = chunkSize;
    tilingCtx.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    tilingCtx.libApiWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    ::ChunkGatedDeltaRuleFwdHTilingData plainTiling{};
    uint32_t blockDim = 0;
    size_t workspaceSize = 0;
    ChunkGatedDeltaRuleFwdHTilingProcessor processor(tilingCtx);
    processor.Process(plainTiling, blockDim, workspaceSize);

    context->SetBlockDim(blockDim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize;

    tiling.set_batch(plainTiling.batch);
    tiling.set_seqlen(plainTiling.seqlen);
    tiling.set_kNumHead(plainTiling.kNumHead);
    tiling.set_vNumHead(plainTiling.vNumHead);
    tiling.set_kHeadDim(plainTiling.kHeadDim);
    tiling.set_vHeadDim(plainTiling.vHeadDim);
    tiling.set_chunkSize(plainTiling.chunkSize);
    tiling.set_useInitialState(plainTiling.useInitialState);
    tiling.set_storeFinalState(plainTiling.storeFinalState);
    tiling.set_dataType(plainTiling.dataType);
    tiling.set_stateDataType(plainTiling.stateDataType);
    tiling.set_gDataType(plainTiling.gDataType);
    tiling.set_isVariedLen(plainTiling.isVariedLen);
    tiling.set_shapeBatch(plainTiling.shapeBatch);
    tiling.set_tokenBatch(plainTiling.tokenBatch);
    tiling.set_vWorkspaceOffset(plainTiling.vWorkspaceOffset);
    tiling.set_vUpdateWorkspaceOffset(plainTiling.vUpdateWorkspaceOffset);
    tiling.set_hWorkspaceOffset(plainTiling.hWorkspaceOffset);
    tiling.set_numSeqWorkspaceOffset(plainTiling.numSeqWorkspaceOffset);
    tiling.set_numChunksWorkspaceOffset(plainTiling.numChunksWorkspaceOffset);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    ChunkGatedDeltaRuleFwdHTilingDataPrint(context, tiling);
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkGatedDeltaRuleFwdH end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkGatedDeltaRuleFwdH(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleFwdH)
    .Tiling(Tiling4ChunkGatedDeltaRuleFwdH)
    .TilingParse<ChunkGatedDeltaRuleFwdHCompileInfo>(TilingPrepareForChunkGatedDeltaRuleFwdH);

} // namespace optiling
