// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../mla_prolog_v3/op_host/mla_prolog_tiling.h"

namespace optiling {
static ge::graphStatus PrepareMlaPrologDcpC8(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(MlaPrologDcpC8).Tiling(TilingMlaProlog)
    .TilingParse<MlaPrologCompileInfo>(PrepareMlaPrologDcpC8);
}  // namespace optiling
