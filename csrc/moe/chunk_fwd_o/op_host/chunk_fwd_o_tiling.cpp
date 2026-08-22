/**
 * Copyright (c) 2026 Tianjin University, Ltd.
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
#include "chunk_fwd_o_tiling_processor.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

static void ChunkFwdOTilingDataPrint(gert::TilingContext *context, const ChunkFwdOTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print ChunkFwdO tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== batch: %ld", tiling.shapeBatch);
    OP_LOGD(nodeName, "=== seqlen: %ld", tiling.seqlen);
    OP_LOGD(nodeName, "=== kNumHead: %ld", tiling.kNumHead);
    OP_LOGD(nodeName, "=== vNumHead: %ld", tiling.vNumHead);
    OP_LOGD(nodeName, "=== kHeadDim: %ld", tiling.kHeadDim);
    OP_LOGD(nodeName, "=== vHeadDim: %ld", tiling.vHeadDim);
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.chunkSize);
    OP_LOGD(nodeName, "=== dataType: %ld", tiling.dataType);
    OP_LOGD(nodeName, "=== gDataType: %ld", tiling.gDataType);
    OP_LOGD(nodeName, "=== isVariedLen: %ld", tiling.isVariedLen);
    OP_LOGD(nodeName, "=== tokenBatch: %ld", tiling.tokenBatch);
    OP_LOGD(nodeName, "=== scale: %f", tiling.scale);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print ChunkFwdO tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus Tiling4ChunkFwdO(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkFwdO start.");
    ChunkFwdOTilingData *tiling = context->GetTilingData<ChunkFwdOTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);

    auto qInputDesc = context->GetInputDesc(CHUNK_FWD_O_INPUT_Q_IDX);
    auto gInputDesc = context->GetInputDesc(CHUNK_FWD_O_INPUT_G_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qInputDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gInputDesc);

    ge::DataType qDtype = qInputDesc->GetDataType();
    ge::DataType gDtype = gInputDesc->GetDataType();
    int64_t dataType = (qDtype == ge::DT_BF16) ? CHUNK_FWD_O_DTYPE_BF16 : CHUNK_FWD_O_DTYPE_FP16;
    int64_t gDataType = CHUNK_FWD_O_DTYPE_FP32;
    if (gDtype == ge::DT_BF16) {
        gDataType = CHUNK_FWD_O_DTYPE_BF16;
    } else if (gDtype == ge::DT_FLOAT16) {
        gDataType = CHUNK_FWD_O_DTYPE_FP16;
    }

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    ChunkFwdOTilingContext ctx{
        context->GetNodeName(),
        context->GetOptionalInputShape(CHUNK_FWD_O_INPUT_Q_IDX),
        context->GetOptionalInputShape(CHUNK_FWD_O_INPUT_K_IDX),
        context->GetOptionalInputShape(CHUNK_FWD_O_INPUT_V_IDX),
        context->GetOptionalInputShape(CHUNK_FWD_O_INPUT_H_IDX),
        context->GetOptionalInputShape(CHUNK_FWD_O_INPUT_G_IDX),
        context->GetOptionalInputShape(CHUNK_FWD_O_INPUT_SEQLENS_IDX),
        context->GetOptionalInputShape(CHUNK_FWD_O_INPUT_CHUNK_OFFSETS_IDX),
        *(attrPtr->GetAttrPointer<double>(CHUNK_FWD_O_ATTR_SCALE_IDX)),
        *(attrPtr->GetAttrPointer<int64_t>(CHUNK_FWD_O_ATTR_CHUNK_SIZE_IDX)),
        dataType,
        gDataType,
        aicCoreNum,
        sysWorkspaceSize,
    };

    ChunkFwdOTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    context->SetBlockDim(aicCoreNum);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = processor.GetWorkspaceSize();

    ChunkFwdOTilingDataPrint(context, *tiling);
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
