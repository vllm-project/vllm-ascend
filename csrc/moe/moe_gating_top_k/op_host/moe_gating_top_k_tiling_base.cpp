/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_gating_top_k_tiling_base.cpp
 * \brief
 */
#include "moe_gating_top_k_tiling.h"
#include "register/op_def_registry.h"
#include "../tiling_base/tiling_base.h"
#include "../tiling_base/tiling_templates_registry.h"
#include "error_log.h"
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {
static ge::graphStatus TilingForMoeGatingTopK(gert::TilingContext *context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMoeGatingTopK(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeGatingTopK)
    .Tiling(TilingForMoeGatingTopK)
    .TilingParse<MoeGatingTopKCompileInfo>(TilingPrepareForMoeGatingTopK);

} // namespace optiling