/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_qkv_proj_norm_rope_tiling.h */
#ifndef DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_TILING_H
#define DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_TILING_H
#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>
namespace optiling {
struct DgemmaFusedQkvProjNormRopeCompileInfo {};
ge::graphStatus DgemmaFusedQkvProjNormRopeTilingFunc(gert::TilingContext *context);
ge::graphStatus TilingPrepareForDgemmaFusedQkvProjNormRope(gert::TilingParseContext *context);
}
#endif
