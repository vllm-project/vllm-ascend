/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_custom_tiling_base.cpp
 * \brief
 */
#include "moe_init_routing_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_templates_registry.h"

#define unlikely(x) __builtin_expect((x), 0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                           \
    do {                                                                                                   \
        if (unlikely((ptr) == nullptr)) {                                                                  \
            const char* name = (unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr)) ? \
                                   "nil" :                                                                 \
                                   (context)->GetNodeName();                                               \
            OPS_LOG_E(name, "%s is nullptr!", #ptr);                                                       \
            return ge::GRAPH_FAILED;                                                                       \
        }                                                                                                  \
    } while (0)

namespace optiling {
static ge::graphStatus TilingForMoeInitRoutingCustom(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMoeInitRountingCustom(gert::TilingParseContext* context)
{   
    OPS_LOG_D(context, "TilingPrepareForMoeInitRountingCustom enter.");

    auto compileInfo = context->GetCompiledInfo<MoeInitRoutingCustomCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->aivNum <= 0) {
        OPS_LOG_E(context, "TilingPrepareForMoeInitRountingCustom fail to get core num.");
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    if (compileInfo->ubSize <= 0) {
        OPS_LOG_E(context, "TilingPrepareForMoeInitRountingCustom fail to get ub size.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeInitRoutingCustom)
    .Tiling(TilingForMoeInitRoutingCustom)
    .TilingParse<MoeInitRoutingCustomCompileInfo>(TilingPrepareForMoeInitRountingCustom);
}  // namespace optiling