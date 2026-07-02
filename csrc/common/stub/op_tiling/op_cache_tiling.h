/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cop_ache_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_OP_CACHE_TILING_H
#define OPS_BUILT_IN_OP_TILING_OP_CACHE_TILING_H

#include <array>
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "op_cache_def_tiling.h"

namespace optiling {

bool TilingPrepareForOpCache(gert::TilingContext* context);
bool TilingPrepareForOpCache(gert::TilingParseContext* context);

bool GenTiling(
    const std::string& op_type, const BatchmatmulCompileParas& compile_params, BatchmatmulRunParas& run_params,
    CacheTilingData& tiling, gert::TilingContext* context);
} // namespace optiling

#endif