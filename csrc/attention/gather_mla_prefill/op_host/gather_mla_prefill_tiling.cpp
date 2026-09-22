// SPDX-License-Identifier: Apache-2.0
#include "gather_mla_prefill_tiling.h"
#include <algorithm>
#include <limits>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
ge::graphStatus TilingGatherMlaPrefill(gert::TilingContext *context)
{
    const auto *attrs = context->GetAttrs();
    const auto *latent = context->GetInputShape(0);
    const auto *table = context->GetInputShape(2);
    if (attrs == nullptr || latent == nullptr || table == nullptr) return ge::GRAPH_FAILED;
    const auto &shape = latent->GetOriginShape();
    const auto &tableShape = table->GetOriginShape();
    if (shape.GetDimNum() != 4 || tableShape.GetDimNum() != 2) return ge::GRAPH_FAILED;
    int64_t values[6];
    for (uint32_t index = 0; index < 6; ++index) {
        const auto *value = attrs->GetAttrPointer<int64_t>(index);
        if (value == nullptr) return ge::GRAPH_FAILED;
        values[index] = *value;
    }
    const int64_t requests = tableShape.GetDim(0);
    if (values[0] < 0 || values[1] <= 0 || requests <= 0 || tableShape.GetDim(1) <= 0 ||
        values[2] <= 0 || values[3] < 512 || values[4] <= 0 || values[5] < 64 ||
        values[3] - 512 > std::numeric_limits<uint32_t>::max() ||
        values[5] - 64 > std::numeric_limits<uint32_t>::max() / 2) return ge::GRAPH_FAILED;
    const int64_t tilesPerRequest = (values[1] + 15) / 16;
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int64_t cores = std::max<int64_t>(1, std::min<int64_t>(
        requests * tilesPerRequest, platform.GetCoreNumAiv()));
    GatherMlaPrefillTilingData tiling;
    tiling.set_requests(requests);
    tiling.set_pages(shape.GetDim(0));
    tiling.set_tableColumns(tableShape.GetDim(1));
    tiling.set_numTokens(values[0]);
    tiling.set_tilesPerRequest(tilesPerRequest);
    tiling.set_latentPageStride(values[2]);
    tiling.set_latentRowStride(values[3]);
    tiling.set_ropePageStride(values[4]);
    tiling.set_ropeRowStride(values[5]);
    tiling.set_usedCoreNum(cores);
    context->SetBlockDim(static_cast<uint32_t>(cores));
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareGatherMlaPrefill(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(GatherMlaPrefill)
    .Tiling(TilingGatherMlaPrefill)
    .TilingParse<GatherMlaPrefillCompileInfo>(PrepareGatherMlaPrefill);
} // namespace optiling
