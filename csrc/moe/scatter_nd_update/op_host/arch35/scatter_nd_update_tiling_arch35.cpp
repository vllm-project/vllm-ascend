/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_update.cc
 * \brief
 */
#include "scatter_nd_update_tiling_regbase.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"

namespace optiling {

static ge::graphStatus Tiling4ScatterNdUpdate(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "ScatterNdUpdateTiling running begin");
    OP_LOGD(context->GetNodeName(), "Tiling4ScatterNdAdd running ScatterNdUpdate tiling.");
    ScatterNdUpdateTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareScatterNdUpdateForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init ScatterNdUpdate Tiling.");
    auto compile_info = context->GetCompiledInfo<ScatterNdUpdateCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compile_info->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compile_info->core_num <= 0), OP_LOGE(context->GetNodeName(), "Failed to core num."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compile_info->ub_size = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compile_info->ub_size <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ScatterNdUpdate(gert::TilingParseContext* context)
{
    TilingPrepareScatterNdUpdateForAscendC(context);
    OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4ScatterNdUpdate success.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the ScatterNdUpdate op.
IMPL_OP_OPTILING(ScatterNdUpdate)
    .Tiling(Tiling4ScatterNdUpdate)
    .TilingParse<ScatterNdUpdateCompileInfo>(TilingPrepare4ScatterNdUpdate);
} // namespace optiling
