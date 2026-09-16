/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

namespace ops {
constexpr uint32_t QUERY_INDEX = 2;
constexpr uint32_t BLOCK_TABLE_INDEX = 5;
constexpr uint32_t CACHE_POOL_INDEX = 12;
constexpr int64_t TOPK = 2048;
constexpr int64_t MISS_CAPACITY = 32768;

static ge::graphStatus InferShapeFusedLiManageMtpC8(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("FusedLiManageMtpC8",
                         "InferShapeContext is nullptr."),
               return ge::GRAPH_FAILED);
    const gert::Shape *query = context->GetInputShape(QUERY_INDEX);
    const gert::Shape *blockTable = context->GetInputShape(BLOCK_TABLE_INDEX);
    const gert::Shape *cache = context->GetInputShape(CACHE_POOL_INDEX);
    OPS_LOG_E_IF_NULL(context, query, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, blockTable, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cache, return ge::GRAPH_FAILED);
    OPS_ERR_IF(query->GetDimNum() != 3 || blockTable->GetDimNum() != 2 || cache->GetDimNum() != 2,
               OPS_LOG_E(context, "invalid fused_li_manage_mtp input ranks."),
               return ge::GRAPH_FAILED);
    const int64_t t = query->GetDim(0);
    const int64_t b = blockTable->GetDim(0);
    gert::Shape *topkSrc = context->GetOutputShape(0);
    gert::Shape *topkDst = context->GetOutputShape(1);
    gert::Shape *topkMiss = context->GetOutputShape(2);
    gert::Shape *missSrc = context->GetOutputShape(3);
    gert::Shape *missDst = context->GetOutputShape(4);
    gert::Shape *missCount = context->GetOutputShape(5);
    gert::Shape *cacheOut = context->GetOutputShape(6);
    OPS_LOG_E_IF_NULL(context, topkSrc, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, topkDst, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, topkMiss, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, missSrc, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, missDst, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, missCount, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, cacheOut, return ge::GRAPH_FAILED);
    topkSrc->SetDimNum(3);
    topkSrc->SetDim(0, t);
    topkSrc->SetDim(1, 1);
    topkSrc->SetDim(2, TOPK);
    *topkDst = *topkSrc;
    topkMiss->SetDimNum(1);
    topkMiss->SetDim(0, t);
    missSrc->SetDimNum(2);
    missSrc->SetDim(0, b);
    missSrc->SetDim(1, MISS_CAPACITY);
    *missDst = *missSrc;
    missCount->SetDimNum(1);
    missCount->SetDim(0, b);
    *cacheOut = *cache;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFusedLiManageMtpC8(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("FusedLiManageMtpC8",
                         "InferDataTypeContext is nullptr."),
               return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < 7; ++i) context->SetOutputDataType(i, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedLiManageMtpC8)
    .InferShape(InferShapeFusedLiManageMtpC8)
    .InferDataType(InferDataTypeFusedLiManageMtpC8);
} // namespace ops
