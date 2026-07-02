/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gdn_gating_tiling.h
 * \brief Function-style tiling declaration for FusedGdnGating.
 */

#ifndef FUSED_GDN_GATING_TILING_H
#define FUSED_GDN_GATING_TILING_H

#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>

namespace optiling {

// Required by CANN tiling framework.
struct FusedGdnGatingCompileInfo {};

ge::graphStatus FusedGdnGatingTilingFunc(gert::TilingContext *context);
ge::graphStatus TilingPrepareForFusedGdnGating(gert::TilingParseContext *context);

} // namespace optiling

#endif // FUSED_GDN_GATING_TILING_H
