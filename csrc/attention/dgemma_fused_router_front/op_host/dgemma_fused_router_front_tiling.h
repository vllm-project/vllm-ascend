/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_router_front_tiling.h */
#ifndef DGEMMA_FUSED_ROUTER_FRONT_TILING_H
#define DGEMMA_FUSED_ROUTER_FRONT_TILING_H
#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>
namespace optiling {
struct DgemmaFusedRouterFrontCompileInfo {};
ge::graphStatus DgemmaFusedRouterFrontTilingFunc(gert::TilingContext *context);
ge::graphStatus TilingPrepareForDgemmaFusedRouterFront(gert::TilingParseContext *context);
}
#endif
