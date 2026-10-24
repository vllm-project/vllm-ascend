// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <algorithm>
#include <limits>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "engram_hash_ids_tiling.h"
#include "../op_kernel/engram_hash_ids_constants.h"

namespace optiling {
namespace {
using namespace EngramHashIds;

// Prefer the largest tile that supplies at least one task per available AIV.
// Dimensions and core count are validated by the host before calling this helper.
uint32_t SelectTokenTileSize(int64_t num_tokens, int64_t num_layers, uint32_t cores)
{
    const uint32_t sizes[]{TOKEN_TILE_ELEMENTS, MEDIUM_TOKEN_TILE_ELEMENTS, SMALL_TOKEN_TILE_ELEMENTS};
    for (const uint32_t size : sizes) {
        const int64_t tiles = num_tokens / size + (num_tokens % size != 0);
        if (tiles * num_layers >= cores) return size;
    }
    return SMALL_TOKEN_TILE_ELEMENTS;
}

ge::graphStatus Tiling(gert::TilingContext *context)
{
    const auto *attrs = context->GetAttrs();
    const auto *parameters = attrs == nullptr ? nullptr : attrs->GetListInt(PARAMETERS);
    if (parameters == nullptr || parameters->GetSize() != PARAMETER_COUNT) {
        return ge::GRAPH_FAILED;
    }
    const auto *p = static_cast<const int64_t *>(parameters->GetData());
    if (p[NUM_TOKENS] < 0 || p[NUM_LAYERS] <= 0 || p[MAX_NGRAM] < 1 || p[NUM_HEADS] <= 0 ||
        p[LOOKBACK_DEPTH] < 0 || p[NUM_TOKENS] > std::numeric_limits<int64_t>::max() / p[NUM_LAYERS] ||
        p[MAX_NGRAM] - 1 > std::numeric_limits<int64_t>::max() / p[NUM_HEADS]) {
        return ge::GRAPH_FAILED;
    }
    for (const auto index : {INPUT_STRIDE, MASK_STRIDE, POSITION_STRIDE, TABLE_STRIDE, TABLE_COL_STRIDE,
                              QUERY_STRIDE, LOOKBACK_ROW_STRIDE, LOOKBACK_COL_STRIDE,
                              LOOKBACK_MASK_ROW_STRIDE, LOOKBACK_MASK_COL_STRIDE}) {
        if (p[index] < 0) return ge::GRAPH_FAILED;
    }
    const bool has_cache = context->GetOptionalInputShape(CACHE) != nullptr;
    if (has_cache && (context->GetOptionalInputShape(BLOCK_TABLE) == nullptr ||
                     p[NUM_SLOTS] <= 0 || p[NUM_TABLE_ROWS] <= 0 || p[MAX_BLOCKS] <= 0 ||
                     p[CACHE_BLOCK_SIZE] <= 0)) {
        return ge::GRAPH_FAILED;
    }
    EngramHashIdsTilingData tiling;
    tiling.set_numTokens(p[NUM_TOKENS]);
    tiling.set_numLayers(p[NUM_LAYERS]);
    tiling.set_maxNgram(p[MAX_NGRAM]);
    tiling.set_numHeads(p[NUM_HEADS]);
    tiling.set_numSlots(p[NUM_SLOTS]);
    tiling.set_numTableRows(p[NUM_TABLE_ROWS]);
    tiling.set_maxBlocks(p[MAX_BLOCKS]);
    tiling.set_lookbackDepth(p[LOOKBACK_DEPTH]);
    tiling.set_padId(p[PAD_ID]);
    tiling.set_deadId(p[DEAD_ID]);
    tiling.set_cacheBlockSize(p[CACHE_BLOCK_SIZE]);
    tiling.set_inputStride(p[INPUT_STRIDE]);
    tiling.set_maskStride(p[MASK_STRIDE]);
    tiling.set_positionStride(p[POSITION_STRIDE]);
    tiling.set_tableRowStride(p[TABLE_STRIDE]);
    tiling.set_tableColStride(p[TABLE_COL_STRIDE]);
    tiling.set_queryStride(p[QUERY_STRIDE]);
    tiling.set_lookbackRowStride(p[LOOKBACK_ROW_STRIDE]);
    tiling.set_lookbackColStride(p[LOOKBACK_COL_STRIDE]);
    tiling.set_lookbackMaskRowStride(p[LOOKBACK_MASK_ROW_STRIDE]);
    tiling.set_lookbackMaskColStride(p[LOOKBACK_MASK_COL_STRIDE]);
    tiling.set_hasCache(has_cache);
    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    if (platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
        // Cover the fixed history/arithmetic buffers while leaving UB for SIMT DCache.
        constexpr uint32_t SIMT_LOCAL_MEMORY_BYTES = 64U * 1024U;
        if (context->SetLocalMemorySize(SIMT_LOCAL_MEMORY_BYTES) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    const uint32_t cores = std::max<uint32_t>(platform.GetCoreNumAiv(), 1);
    const uint32_t tile_size = SelectTokenTileSize(p[NUM_TOKENS], p[NUM_LAYERS], cores);
    tiling.set_tokenTileSize(tile_size);
    const int64_t token_tiles = p[NUM_TOKENS] / tile_size + (p[NUM_TOKENS] % tile_size != 0);
    const int64_t tasks = std::max<int64_t>(token_tiles * p[NUM_LAYERS], 1);
    const auto used_cores = static_cast<uint32_t>(std::min<int64_t>(tasks, cores));
    tiling.set_usedCoreNum(used_cores);
    context->SetBlockDim(used_cores);
    context->SetTilingKey(TILING_KEY);
    context->GetWorkspaceSizes(1)[0] = 0;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParse(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(EngramHashIds).Tiling(Tiling).TilingParse<EngramHashIdsCompileInfo>(TilingParse);
} // namespace optiling
