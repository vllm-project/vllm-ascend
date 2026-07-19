/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef DGEMMA_APPLY_ROUTER_SCALE_TILING_H
#define DGEMMA_APPLY_ROUTER_SCALE_TILING_H
#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>
namespace optiling {
struct DgemmaApplyRouterScaleCompileInfo {};
ge::graphStatus DgemmaApplyRouterScaleTilingFunc(gert::TilingContext *context);
ge::graphStatus TilingPrepareForDgemmaApplyRouterScale(gert::TilingParseContext *context);
}
#endif
