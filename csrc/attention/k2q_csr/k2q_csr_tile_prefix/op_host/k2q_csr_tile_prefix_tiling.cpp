/**
 * K2qCsrTilePrefix Host tiling.
 */
#include "../../k2q_csr_common/op_host/k2q_csr_tiling_common.h"

namespace optiling {
struct K2qCsrTilePrefixCompileInfo {};

static ge::graphStatus K2qCsrTilePrefixTilingFunc(gert::TilingContext *context)
{
    return k2q_csr_tiling::FillTiling(context, k2q_csr_tiling::Stage::TilePrefix);
}

static ge::graphStatus TilingParseForK2qCsrTilePrefix([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(K2qCsrTilePrefix)
    .Tiling(K2qCsrTilePrefixTilingFunc)
    .TilingParse<K2qCsrTilePrefixCompileInfo>(TilingParseForK2qCsrTilePrefix);
} // namespace optiling
