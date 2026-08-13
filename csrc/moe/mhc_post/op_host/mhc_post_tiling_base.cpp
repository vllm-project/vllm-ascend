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
 * \file mhc_post_tiling_base.cpp
 * \brief
 */

#include "mhc_post_tiling.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {

static ge::graphStatus TilingPrepareForMhcPost([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForMhcPost(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MhcPost", "context is null"), return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

IMPL_OP_OPTILING(MhcPost)
    .Tiling(TilingForMhcPost)
    .TilingParse<MhcPostCompileInfo>(TilingPrepareForMhcPost);

}  // namespace optiling