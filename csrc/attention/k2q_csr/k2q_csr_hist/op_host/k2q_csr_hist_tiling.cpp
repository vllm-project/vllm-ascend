/**
 * K2qCsrHist Host tiling.
 */
#include "../../k2q_csr_common/op_host/k2q_csr_tiling_common.h"

namespace optiling {
struct K2qCsrHistCompileInfo {};

static ge::graphStatus K2qCsrHistTilingFunc(gert::TilingContext *context)
{
    return k2q_csr_tiling::FillTiling(context, k2q_csr_tiling::Stage::Hist);
}

static ge::graphStatus TilingParseForK2qCsrHist([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(K2qCsrHist)
    .Tiling(K2qCsrHistTilingFunc)
    .TilingParse<K2qCsrHistCompileInfo>(TilingParseForK2qCsrHist);
} // namespace optiling
