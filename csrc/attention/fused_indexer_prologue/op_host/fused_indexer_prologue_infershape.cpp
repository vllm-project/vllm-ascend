/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <register/op_impl_registry.h>

#include "error/ops_error.h"

namespace ops {
namespace {
constexpr uint32_t K_CACHE_INPUT = 6;
constexpr uint32_t SCALE_CACHE_INPUT = 7;
}  // namespace

static ge::graphStatus InferShapeFusedIndexerPrologue(
    gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("FusedIndexerPrologue", "InferShapeContext is null."),
               return ge::GRAPH_FAILED);
    const gert::Shape *kCache = context->GetInputShape(K_CACHE_INPUT);
    const gert::Shape *scaleCache = context->GetInputShape(SCALE_CACHE_INPUT);
    gert::Shape *kCacheOut = context->GetOutputShape(0);
    gert::Shape *scaleCacheOut = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL(context, kCache, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, scaleCache, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, kCacheOut, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, scaleCacheOut, return ge::GRAPH_FAILED);
    *kCacheOut = *kCache;
    *scaleCacheOut = *scaleCache;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFusedIndexerPrologue(
    gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("FusedIndexerPrologue", "InferDataTypeContext is null."),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_INT8);
    context->SetOutputDataType(1, ge::DT_FLOAT16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedIndexerPrologue)
    .InferShape(InferShapeFusedIndexerPrologue)
    .InferDataType(InferDataTypeFusedIndexerPrologue);
}  // namespace ops
