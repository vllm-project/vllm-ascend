/**
 * K2qCsrMeta Host tiling.
 */
#include "../../k2q_csr_common/op_host/k2q_csr_tiling_common.h"

namespace optiling {
struct K2qCsrMetaCompileInfo {};

static ge::graphStatus K2qCsrMetaTilingFunc(gert::TilingContext *context)
{
    return k2q_csr_tiling::FillTiling(context, k2q_csr_tiling::Stage::Meta);
}

static ge::graphStatus TilingParseForK2qCsrMeta([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(K2qCsrMeta)
    .Tiling(K2qCsrMetaTilingFunc)
    .TilingParse<K2qCsrMetaCompileInfo>(TilingParseForK2qCsrMeta);
} // namespace optiling
