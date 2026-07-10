/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_wu_fwd_tiling.cpp
 * \brief
 */

#include "recompute_wu_fwd_tiling.h"
#include "recompute_wu_fwd_tiling_processor.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

static void RecomputeWUFwdTilingDataPrint(gert::TilingContext *context, const GDN::RecomputeWUFwdTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print RecomputeWUFwd tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== B: %ld", tiling.B);
    OP_LOGD(nodeName, "=== Hk: %ld", tiling.Hk);
    OP_LOGD(nodeName, "=== Hv: %ld", tiling.Hv);
    OP_LOGD(nodeName, "=== hvPerHk: %ld", tiling.hvPerHk);
    OP_LOGD(nodeName, "=== T: %ld", tiling.T);
    OP_LOGD(nodeName, "=== K: %ld", tiling.K);
    OP_LOGD(nodeName, "=== V: %ld", tiling.V);
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.chunkSize);
    OP_LOGD(nodeName, "=== vbVecRow: %ld", tiling.vbVecRow);
    OP_LOGD(nodeName, "=== kbgExpVecRow: %ld", tiling.kbgExpVecRow);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print RecomputeWUFwd tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus Tiling4RecomputeWUFwd(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4RecomputeWUFwd start.");
    GDN::RecomputeWUFwdTilingData tiling{};

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);

    auto kDesc = context->GetDynamicInputDesc(RECOMPUTE_WU_FWD_INPUT_K_IDX, 0);
    auto betaDesc = context->GetDynamicInputDesc(RECOMPUTE_WU_FWD_INPUT_BETA_IDX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, kDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    auto cuSeqlensTensor = context->GetOptionalInputTensor(RECOMPUTE_WU_FWD_INPUT_SEQLENS_IDX);
    auto chunkIndicesTensor = context->GetOptionalInputTensor(RECOMPUTE_WU_FWD_INPUT_CHUNK_INDICES_IDX);
    const int64_t *cuSeqlensData = cuSeqlensTensor != nullptr ? cuSeqlensTensor->GetData<int64_t>() : nullptr;
    const int64_t *chunkIndicesData =
        chunkIndicesTensor != nullptr ? chunkIndicesTensor->GetData<int64_t>() : nullptr;

    RecomputeWUFwdTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(RECOMPUTE_WU_FWD_INPUT_K_IDX),
        context->GetRequiredInputShape(RECOMPUTE_WU_FWD_INPUT_V_IDX),
        context->GetRequiredInputShape(RECOMPUTE_WU_FWD_INPUT_BETA_IDX),
        context->GetRequiredInputShape(RECOMPUTE_WU_FWD_INPUT_A_IDX),
        context->GetRequiredInputShape(RECOMPUTE_WU_FWD_INPUT_G_IDX),
        context->GetOptionalInputShape(RECOMPUTE_WU_FWD_INPUT_SEQLENS_IDX),
        context->GetOptionalInputShape(RECOMPUTE_WU_FWD_INPUT_CHUNK_INDICES_IDX),
        cuSeqlensData,
        chunkIndicesData,
        *(attrPtr->GetAttrPointer<int32_t>(0)),
        kDesc->GetDataType(),
        betaDesc->GetDataType(),
        ubSize,
        sysWorkspaceSize,
    };

    RecomputeWUFwdTilingProcessor processor(ctx, tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    if (tiling.V == RECOMPUTE_WU_FWD_V_DIM_256) {
        context->SetTilingKey(2);
    } else {
        context->SetTilingKey(1);
    }
    OP_LOGD(context->GetNodeName(), "tilingKey: %d (V=%ld)", context->GetTilingKey(), tiling.V);
    RecomputeWUFwdTilingDataPrint(context, tiling);

    RecomputeWUFwdTilingData tilingData;
    tilingData.set_B(tiling.B);
    tilingData.set_Hk(tiling.Hk);
    tilingData.set_Hv(tiling.Hv);
    tilingData.set_hvPerHk(tiling.hvPerHk);
    tilingData.set_T(tiling.T);
    tilingData.set_K(tiling.K);
    tilingData.set_V(tiling.V);
    tilingData.set_chunkNum(tiling.chunkNum);
    tilingData.set_chunkSize(tiling.chunkSize);
    tilingData.set_vbVecRow(tiling.vbVecRow);
    tilingData.set_kbgExpVecRow(tiling.kbgExpVecRow);
    tilingData.set_isVariable(tiling.isVariable);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    context->SetBlockDim(ascendcPlatform.GetCoreNumAic());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);
    OP_LOGD(context->GetNodeName(), "Tiling4RecomputeWUFwd end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingRecomputeForRecomputeWUFwd(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RecomputeWUFwd)
    .Tiling(Tiling4RecomputeWUFwd)
    .TilingParse<RecomputeWUFwdCompileInfo>(TilingRecomputeForRecomputeWUFwd);

} // namespace optiling
