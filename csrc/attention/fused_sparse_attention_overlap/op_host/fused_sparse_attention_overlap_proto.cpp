#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t SELECTION_KV_BLOCK_TABLE_INPUT_INDEX = 11;

ge::graphStatus InferShapeFusedSparseAttentionOverlap(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("FusedSparseAttentionOverlap", "InferShapeContext is nullptr"),
        return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)
    *attentionOutShape = *queryShape;

    const gert::Shape *selectionKvBlockTableShape = context->GetInputShape(SELECTION_KV_BLOCK_TABLE_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, selectionKvBlockTableShape, return ge::GRAPH_FAILED)
    gert::Shape *selectionKvActualSeqShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL(context, selectionKvActualSeqShape, return ge::GRAPH_FAILED)
    *selectionKvActualSeqShape = *selectionKvBlockTableShape;
    selectionKvActualSeqShape->SetDimNum(selectionKvBlockTableShape->GetDimNum() - 1);
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeFusedSparseAttentionOverlap(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("FusedSparseAttentionOverlap", "InferDataTypeContext is nullptr"),
        return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FusedSparseAttentionOverlap)
    .InferShape(InferShapeFusedSparseAttentionOverlap)
    .InferDataType(InferDataTypeFusedSparseAttentionOverlap);
} // namespace ops
