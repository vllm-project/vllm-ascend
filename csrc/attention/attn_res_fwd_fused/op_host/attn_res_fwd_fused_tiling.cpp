// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../attn_res_fwd/op_host/attn_res_fwd_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
namespace optiling {
REGISTER_OPS_TILING_TEMPLATE(AttnResFwdFused, AttnResFwdTiling, 0);
static ge::graphStatus Tiling(gert::TilingContext *context) {
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}
static ge::graphStatus Parse(gert::TilingParseContext *context) {
    return context != nullptr && context->GetPlatformInfo() != nullptr ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}
IMPL_OP_OPTILING(AttnResFwdFused).Tiling(Tiling).TilingParse<AttnResFwdCompileInfo>(Parse);
}
