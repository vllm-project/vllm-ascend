/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "kda_layout_swap12_tiling.h"
#include <algorithm>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t OUTPUT_Y_IDX = 0;
constexpr size_t DIM_B = 0;
constexpr size_t DIM_FIRST = 1;
constexpr size_t DIM_SECOND = 2;

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

ge::graphStatus Tiling4KdaLayoutSwap12(gert::TilingContext *context)
{
    KdaLayoutSwap12TilingData tiling;
    auto xShape = context->GetOptionalInputShape(INPUT_X_IDX)->GetStorageShape();
    auto yShape = context->GetOutputShape(OUTPUT_Y_IDX)->GetStorageShape();
    auto xDesc = context->GetInputDesc(INPUT_X_IDX);
    if (xDesc == nullptr || xShape.GetDimNum() < 3) {
        return ge::GRAPH_FAILED;
    }

    int64_t batch = xShape.GetDim(DIM_B);
    int64_t firstDim = xShape.GetDim(DIM_FIRST);
    int64_t secondDim = xShape.GetDim(DIM_SECOND);
    int64_t tailDim = 1;
    for (size_t idx = 3; idx < xShape.GetDimNum(); ++idx) {
        tailDim *= xShape.GetDim(idx);
    }

    if (xShape.GetDimNum() == 3 && yShape.GetDimNum() == 3 &&
        yShape.GetDim(0) == xShape.GetDim(1) &&
        yShape.GetDim(1) == xShape.GetDim(0) &&
        yShape.GetDim(2) == xShape.GetDim(2)) {
        batch = 1;
        firstDim = xShape.GetDim(0);
        secondDim = xShape.GetDim(1);
        tailDim = xShape.GetDim(2);
    }

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    int64_t rowCount = batch * firstDim * secondDim;
    uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(rowCount, coreNum));
    context->SetBlockDim(blockDim == 0 ? 1 : blockDim);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    tiling.set_batch(batch);
    tiling.set_firstDim(firstDim);
    tiling.set_secondDim(secondDim);
    tiling.set_tailDim(tailDim);
    tiling.set_dataType(DTypeCode(xDesc->GetDataType()));
    tiling.set_usedCoreNum(blockDim == 0 ? 1 : blockDim);

    if (xDesc->GetDataType() == ge::DT_FLOAT) {
        context->SetTilingKey(0);
    } else if (xDesc->GetDataType() == ge::DT_BF16) {
        context->SetTilingKey(1);
    } else {
        context->SetTilingKey(2);
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4KdaLayoutSwap12(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KdaLayoutSwap12)
    .Tiling(Tiling4KdaLayoutSwap12)
    .TilingParse<KdaLayoutSwap12CompileInfo>(TilingPrepare4KdaLayoutSwap12);

} // namespace optiling
