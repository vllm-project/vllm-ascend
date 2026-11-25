/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace optiling {

static ge::graphStatus Tiling4WinProbe(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetBlockDim(1);
    context->SetTilingKey(0);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(WinProbe).Tiling(Tiling4WinProbe);

}  // namespace optiling
