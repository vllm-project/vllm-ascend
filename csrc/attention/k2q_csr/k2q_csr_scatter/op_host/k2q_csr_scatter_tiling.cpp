/**
 * K2qCsrScatter Host tiling.
 */
#include "../../k2q_csr_common/op_host/k2q_csr_tiling_common.h"

namespace optiling {
struct K2qCsrScatterCompileInfo {};

static ge::graphStatus K2qCsrScatterTilingFunc(gert::TilingContext *context)
{
    return k2q_csr_tiling::FillTiling(context, k2q_csr_tiling::Stage::Scatter);
}

static ge::graphStatus TilingParseForK2qCsrScatter([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(K2qCsrScatter)
    .Tiling(K2qCsrScatterTilingFunc)
    .TilingParse<K2qCsrScatterCompileInfo>(TilingParseForK2qCsrScatter);
} // namespace optiling
