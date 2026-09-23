// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vllm-ascend project

#include "rearrange_qkv_dma_tiling.h"

#include <algorithm>
#include <cstdint>

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint64_t TILE_ELEMENTS = 160 * 1024 / sizeof(uint16_t);
constexpr uint32_t MAX_DMA_BLOCK_COUNT = 4095;
}  // namespace

static ge::graphStatus RearrangeQkvDmaTilingFunc(gert::TilingContext* context)
{
    const auto& xStorageShape = context->GetInputShape(0)->GetStorageShape();
    const auto attrs = context->GetAttrs();
    const int64_t qDim = *attrs->GetInt(0);
    const int64_t kDim = *attrs->GetInt(1);
    const int64_t vDim = *attrs->GetInt(2);
    const int64_t tokens = xStorageShape.GetDim(0);
    const int64_t rowDim = xStorageShape.GetDim(1);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(tokens, aicCoreNum));
    const uint32_t tileRows = static_cast<uint32_t>(
        std::min<uint64_t>(TILE_ELEMENTS / static_cast<uint64_t>(rowDim), MAX_DMA_BLOCK_COUNT));

    RearrangeQkvDmaTilingData tilingData;
    tilingData.set_tokens(static_cast<uint64_t>(tokens));
    tilingData.set_qDim(static_cast<uint64_t>(qDim));
    tilingData.set_kDim(static_cast<uint64_t>(kDim));
    tilingData.set_vDim(static_cast<uint64_t>(vDim));
    tilingData.set_rowDim(static_cast<uint64_t>(rowDim));
    tilingData.set_tileRows(tileRows);
    tilingData.set_usedCoreNum(usedCoreNum);

    context->GetWorkspaceSizes(1)[0] = 0;
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(0);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RearrangeQkvDmaTilingParse(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RearrangeQkvDma)
    .Tiling(RearrangeQkvDmaTilingFunc)
    .TilingParse<RearrangeQkvDmaCompileInfo>(RearrangeQkvDmaTilingParse);
}  // namespace optiling
