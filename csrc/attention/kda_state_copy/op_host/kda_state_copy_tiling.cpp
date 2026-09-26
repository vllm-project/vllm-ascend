// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/kda_state_copy_tiling_data.h"
namespace optiling {
static ge::graphStatus TilingKdaStateCopy(gert::TilingContext *ctx)
{
    const auto *attrs = ctx->GetAttrs();
    if (!attrs || !ctx->GetPlatformInfo() || !ctx->GetInputDesc(1)) return ge::GRAPH_FAILED;
    auto *data = ctx->GetTilingData<KdaStateCopyTilingData>();
    data->cacheRows = *attrs->GetAttrPointer<int64_t>(0);
    data->selectedRows = *attrs->GetAttrPointer<int64_t>(1);
    data->cacheStrideBytes = *attrs->GetAttrPointer<int64_t>(2);
    data->payloadBytes = *attrs->GetAttrPointer<int64_t>(3);
    data->toCache = *attrs->GetAttrPointer<bool>(4) ? 1 : 0;
    data->hasInitialState = ctx->GetOptionalInputShape(2) ? 1 : 0;
    if (data->cacheRows <= 0 || data->selectedRows <= 0 || data->payloadBytes <= 0 ||
        data->cacheStrideBytes < data->payloadBytes) return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    const int64_t tiles = (data->payloadBytes + KDA_STATE_COPY_TILE_BYTES - 1) / KDA_STATE_COPY_TILE_BYTES;
    // Tiny test/custom layouts can share a 32-byte store unit between states.
    // Keep those copies serial; the model's 768 KiB states use all AIV cores.
    const bool aligned = data->payloadBytes % 32 == 0 && data->cacheStrideBytes % 32 == 0;
    ctx->SetBlockDim(aligned ? std::min<int64_t>(data->selectedRows * tiles, platform.GetCoreNumAiv()) : 1);
    ctx->SetTilingKey(ctx->GetInputDesc(1)->GetDataType() == ge::DT_INT64 ? 1 : 0);
    ctx->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(KdaStateCopy).Tiling(TilingKdaStateCopy);
}
