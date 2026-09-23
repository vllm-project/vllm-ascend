// SPDX-License-Identifier: Apache-2.0
#include "flash_mla_bf16_prepare_tiling.h"
#include <algorithm>
#include <limits>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
ge::graphStatus TilingFlashMlaBf16Prepare(gert::TilingContext *context)
{
    const auto *attrs = context->GetAttrs();
    const auto *key = context->GetInputShape(0);
    const auto *rope = context->GetInputShape(2);
    if (attrs == nullptr || key == nullptr || rope == nullptr) return ge::GRAPH_FAILED;
    const auto &ks = key->GetOriginShape();
    const auto &rs = rope->GetOriginShape();
    if (ks.GetDimNum() != 3 || rs.GetDimNum() != 3 || ks.GetDim(0) <= 0 ||
        ks.GetDim(1) < 1 || ks.GetDim(1) > 128 || ks.GetDim(2) != 128 ||
        rs.GetDim(0) != ks.GetDim(0) || rs.GetDim(2) != 64 ||
        (rs.GetDim(1) != 1 && rs.GetDim(1) != ks.GetDim(1))) return ge::GRAPH_FAILED;
    int64_t strides[6];
    for (uint32_t i = 0; i < 6; ++i) {
        const auto *value = attrs->GetAttrPointer<int64_t>(i);
        if (value == nullptr || *value < (i < 4 ? 128 : 64) ||
            *value > std::numeric_limits<uint32_t>::max() / sizeof(uint16_t)) return ge::GRAPH_FAILED;
        strides[i] = *value;
    }
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    constexpr uint64_t UB_RESERVE_BYTES = 32 * 1024;
    const uint64_t bytesPerToken = 2 * sizeof(uint16_t) * (ks.GetDim(1) * 320 + rs.GetDim(1) * 64);
    if (ubBytes <= UB_RESERVE_BYTES || ubBytes - UB_RESERVE_BYTES < bytesPerToken) return ge::GRAPH_FAILED;
    const uint32_t tileTokens = static_cast<uint32_t>(std::min<uint64_t>(
        std::min<int64_t>(16, 128 / ks.GetDim(1)), (ubBytes - UB_RESERVE_BYTES) / bytesPerToken));
    const int64_t tiles = (ks.GetDim(0) + tileTokens - 1) / tileTokens;
    const uint32_t cores = static_cast<uint32_t>(std::max<int64_t>(1,
        std::min<int64_t>(tiles, platform.GetCoreNumAiv())));
    FlashMlaBf16PrepareTilingData data;
    data.set_tokens(ks.GetDim(0));
    data.set_heads(ks.GetDim(1));
    data.set_ropeHeads(rs.GetDim(1));
    data.set_keyTokenStride(strides[0]);
    data.set_keyHeadStride(strides[1]);
    data.set_valueTokenStride(strides[2]);
    data.set_valueHeadStride(strides[3]);
    data.set_ropeTokenStride(strides[4]);
    data.set_ropeHeadStride(strides[5]);
    data.set_tileTokens(tileTokens);
    data.set_usedCores(cores);
    context->SetBlockDim(cores);
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus PrepareFlashMlaBf16Prepare(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(FlashMlaBf16Prepare)
    .Tiling(TilingFlashMlaBf16Prepare)
    .TilingParse<FlashMlaBf16PrepareCompileInfo>(PrepareFlashMlaBf16Prepare);
} // namespace optiling
