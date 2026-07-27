/**
 * K2qCsrRowPrefix Host tiling.
 */
#include "../../k2q_csr_common/op_host/k2q_csr_tiling_common.h"

namespace optiling {
struct K2qCsrRowPrefixCompileInfo {};

static ge::graphStatus K2qCsrRowPrefixTilingFunc(gert::TilingContext *context)
{
    return k2q_csr_tiling::FillTiling(context, k2q_csr_tiling::Stage::RowPrefix);
}

static ge::graphStatus TilingParseForK2qCsrRowPrefix([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(K2qCsrRowPrefix)
    .Tiling(K2qCsrRowPrefixTilingFunc)
    .TilingParse<K2qCsrRowPrefixCompileInfo>(TilingParseForK2qCsrRowPrefix);
} // namespace optiling
