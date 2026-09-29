// SPDX-License-Identifier: Apache-2.0
#include "vllm_mhc_expand_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <limits>

namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* context)
{
    const auto* shape = context->GetInputShape(0);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetPlatformInfo());
    const auto* mult = attrs->GetInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, mult);
    const auto& x = shape->GetStorageShape();
    if (x.GetDimNum() != 2 || x.GetDim(0) <= 0 || x.GetDim(1) <= 0 || *mult <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t s = x.GetDim(0), d = x.GetDim(1), m = *mult;
    constexpr uint64_t MAX_ELEMENTS = std::numeric_limits<int64_t>::max() / 2;
    if (s > MAX_ELEMENTS / d || s * d > MAX_ELEMENTS / m) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint32_t availableCores = platform.GetCoreNumAiv();
    constexpr uint64_t RESERVE = 4096;
    // Retain the source implementation's conservative tile budget.
    constexpr uint64_t BYTES_PER_ELEMENT = 12;
    constexpr uint64_t ALIGNMENT = 32;
    constexpr uint64_t MAX_TILE = 8192;
    constexpr uint64_t BYTES_PER_CORE = 32768;
    constexpr uint64_t MIN_OUTPUT_PARTITION_WIDTH = 1024;
    if (availableCores == 0 || ubBytes < RESERVE + BYTES_PER_ELEMENT * ALIGNMENT) {
        return ge::GRAPH_FAILED;
    }
    uint64_t tile = std::min(MAX_TILE, (ubBytes - RESERVE) / (BYTES_PER_ELEMENT * ALIGNMENT) * ALIGNMENT);
    const bool partitionOutput = d % 16 != 0 && d >= MIN_OUTPUT_PARTITION_WIDTH;
    const uint64_t expandedBytes = s * d * m * 2;
    const uint64_t computeCores = std::min<uint64_t>(availableCores,
        std::max(s, (expandedBytes - 1) / BYTES_PER_CORE + 1));
    const uint64_t columnsPerCore = (computeCores - 1) / s + 1;
    const uint64_t target = (d - 1) / columnsPerCore + 1;
    const bool bulk = d % 64 == 0 && d * m <= 2040 && tile >= d && tile * 2 >= (m + 1) * d;
    const uint64_t useful = bulk || partitionOutput ? tile : std::min(target, tile);
    tile = (useful + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    const uint64_t perRow = (d - 1) / tile + 1;
    // Output tiles own disjoint aligned GM ranges even when row boundaries
    // are unaligned. Their input mapping is resolved inside the kernel.
    const uint64_t total = partitionOutput ? (s * d * m - 1) / tile + 1 : s * perRow;
    const uint64_t rows = bulk ? std::min<uint64_t>(16, tile * 2 / ((m + 1) * d)) : 1;
    const uint64_t tasks = bulk ? (s - 1) / rows + 1 : total;
    uint32_t cores = std::min<uint64_t>(bulk || partitionOutput ? availableCores : computeCores, tasks);
    // An unaligned row boundary can share a 32-byte output block with its neighbor.
    if (d % 16 != 0 && !partitionOutput) {
        cores = 1;
    }
    VllmMhcExpandTilingData params;
    params.set_tokens(s);
    params.set_hidden(d);
    params.set_mhcMult(m);
    params.set_tilesPerRow(perRow);
    params.set_totalTiles(total);
    params.set_tileLength(static_cast<uint32_t>(tile));
    context->SetTilingKey(partitionOutput ? 2 : 1);
    context->SetBlockDim(cores);
    params.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(params.GetDataSize());
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(VllmMhcExpand).Tiling(Tiling);
}  // namespace optiling
