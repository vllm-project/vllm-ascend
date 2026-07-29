/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

namespace ops {
namespace {
constexpr int32_t DST_CACHE_0_INPUT = 2;
constexpr int32_t DST_CACHE_1_INPUT = 3;
constexpr int32_t DST_CACHE_0_OUTPUT = 0;
constexpr int32_t DST_CACHE_1_OUTPUT = 1;
}  // namespace

static ge::graphStatus InferShapeKvCacheFullBlockDump(
    gert::InferShapeContext* context)
{
    const gert::Shape* cache0Shape = context->GetInputShape(DST_CACHE_0_INPUT);
    const gert::Shape* cache1Shape = context->GetInputShape(DST_CACHE_1_INPUT);
    OPS_LOG_E_IF_NULL(context, cache0Shape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cache1Shape, return ge::GRAPH_FAILED);
    gert::Shape* cache0OutputShape = context->GetOutputShape(
        DST_CACHE_0_OUTPUT);
    gert::Shape* cache1OutputShape = context->GetOutputShape(
        DST_CACHE_1_OUTPUT);
    OPS_LOG_E_IF_NULL(context, cache0OutputShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cache1OutputShape, return ge::GRAPH_FAILED);
    *cache0OutputShape = *cache0Shape;
    *cache1OutputShape = *cache1Shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeKvCacheFullBlockDump(
    gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(
        DST_CACHE_0_OUTPUT, context->GetInputDataType(DST_CACHE_0_INPUT));
    context->SetOutputDataType(
        DST_CACHE_1_OUTPUT, context->GetInputDataType(DST_CACHE_1_INPUT));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvCacheFullBlockDump)
    .InferShape(InferShapeKvCacheFullBlockDump)
    .InferDataType(InferDtypeKvCacheFullBlockDump);
}  // namespace ops
