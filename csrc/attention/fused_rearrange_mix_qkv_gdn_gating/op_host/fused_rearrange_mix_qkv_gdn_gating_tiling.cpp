// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vllm-ascend project

#include "fused_rearrange_mix_qkv_gdn_gating_tiling.h"

#include <algorithm>
#include <cstdint>

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
// One 32-byte DMA block holds 16 bf16 elements.
constexpr uint64_t DMA_TILE_ELEMENTS = 160 * 1024 / sizeof(uint16_t);
constexpr uint32_t MAX_DMA_BLOCK_COUNT = 4095;
// rows * heads handled by a single gating tile, bounded by the UB budget of
// the vector stage (18 bytes per element).
constexpr uint32_t MAX_GATING_TILE_ELEMENTS = 4096;
constexpr uint32_t MAX_GATING_TILE_ROWS = 512;
}  // namespace

static ge::graphStatus FusedRearrangeMixQkvGdnGatingTilingFunc(gert::TilingContext* context)
{
    const auto& xStorageShape = context->GetInputShape(0)->GetStorageShape();
    const auto& aStorageShape = context->GetInputShape(1)->GetStorageShape();
    const auto* attrs = context->GetAttrs();

    const int64_t qDim = *attrs->GetInt(0);
    const int64_t kDim = *attrs->GetInt(1);
    const int64_t vDim = *attrs->GetInt(2);
    const float beta = *attrs->GetFloat(3);
    const float threshold = *attrs->GetFloat(4);

    const int64_t tokens = xStorageShape.GetDim(0);
    const int64_t rowDim = xStorageShape.GetDim(1);
    const int64_t numHeads = aStorageShape.GetDim(1);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
    const auto availableCoreNum = static_cast<uint64_t>(aicCoreNum);
    const uint64_t activeCoreNum =
        std::min<uint64_t>(static_cast<uint64_t>(std::max<int64_t>(tokens, 0)), availableCoreNum);
    const uint32_t dmaCoreNum = static_cast<uint32_t>(std::max<uint64_t>(activeCoreNum, 1));
    const uint64_t rowsPerTile =
        DMA_TILE_ELEMENTS / static_cast<uint64_t>(std::max<int64_t>(rowDim, 1));
    const uint32_t dmaTileRows =
        static_cast<uint32_t>(std::min<uint64_t>(rowsPerTile, MAX_DMA_BLOCK_COUNT));

    uint32_t gatingTileRows = 1;
    if (numHeads > 0) {
        const int64_t headsPerTile = std::max<int64_t>(1, numHeads);
        gatingTileRows = static_cast<uint32_t>(std::max<int64_t>(
            1, std::min<int64_t>(MAX_GATING_TILE_ELEMENTS / headsPerTile, MAX_GATING_TILE_ROWS)));
    }

    FusedRearrangeMixQkvGdnGatingTilingData tilingData;
    tilingData.set_tokens(static_cast<uint64_t>(tokens));
    tilingData.set_qDim(static_cast<uint64_t>(qDim));
    tilingData.set_kDim(static_cast<uint64_t>(kDim));
    tilingData.set_vDim(static_cast<uint64_t>(vDim));
    tilingData.set_rowDim(static_cast<uint64_t>(rowDim));
    tilingData.set_dmaTileRows(dmaTileRows);
    tilingData.set_dmaCoreNum(dmaCoreNum);
    tilingData.set_numHeads(static_cast<uint32_t>(numHeads));
    tilingData.set_gatingTileRows(gatingTileRows);
    tilingData.set_beta(beta);
    tilingData.set_threshold(threshold);

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    // blockDim counts MIX task groups.  Each group contains one AIC and two
    // AIV tasks, so the number of active AIC DMA workers is the required
    // group count.  Do not launch all physical cores for a short token batch.
    const uint32_t blockDim = dmaCoreNum;
    // The second AIV in the last group is filtered by the device-side guard.
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FusedRearrangeMixQkvGdnGatingTilingParse(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedRearrangeMixQkvGdnGating)
    .Tiling(FusedRearrangeMixQkvGdnGatingTilingFunc)
    .TilingParse<FusedRearrangeMixQkvGdnGatingCompileInfo>(FusedRearrangeMixQkvGdnGatingTilingParse);

}  // namespace optiling
