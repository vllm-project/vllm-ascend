#include "register/op_impl_registry.h"
#include <cstdint>

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    const int32_t B = (int32_t)*attrs->GetAttrPointer<int64_t>(0);
    const int32_t H = (int32_t)*attrs->GetAttrPointer<int64_t>(1);
    const int32_t NT = (int32_t)*attrs->GetAttrPointer<int64_t>(2);
    const int32_t NT_Q = (int32_t)*attrs->GetAttrPointer<int64_t>(3);
    const int32_t NG = (int32_t)*attrs->GetAttrPointer<int64_t>(4);
    const int32_t NPAD = (int32_t)*attrs->GetAttrPointer<int64_t>(5);
    const int32_t NT_Q_PAD8 = (int32_t)*attrs->GetAttrPointer<int64_t>(6);
    gert::Shape* threshold_shape = context->GetOutputShape(0);
    threshold_shape->SetDimNum(0);
    threshold_shape->AppendDim(B);
    threshold_shape->AppendDim(NT);
    threshold_shape->AppendDim(H);
    gert::Shape* group_approx_mask_shape = context->GetOutputShape(1);
    group_approx_mask_shape->SetDimNum(0);
    group_approx_mask_shape->AppendDim(B);
    group_approx_mask_shape->AppendDim(H);
    group_approx_mask_shape->AppendDim(NT_Q);
    group_approx_mask_shape->AppendDim(NG);
    gert::Shape* flat_exact_cnt_shape = context->GetOutputShape(2);
    flat_exact_cnt_shape->SetDimNum(0);
    flat_exact_cnt_shape->AppendDim(B);
    flat_exact_cnt_shape->AppendDim(H);
    flat_exact_cnt_shape->AppendDim(NT_Q_PAD8);
    gert::Shape* flat_exact_idx_shape = context->GetOutputShape(3);
    flat_exact_idx_shape->SetDimNum(0);
    flat_exact_idx_shape->AppendDim(B);
    flat_exact_idx_shape->AppendDim(H);
    flat_exact_idx_shape->AppendDim(NT_Q);
    flat_exact_idx_shape->AppendDim(NPAD);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_UINT64);
    context->SetOutputDataType(2, ge::DT_INT32);
    context->SetOutputDataType(3, ge::DT_UINT16);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {
IMPL_OP_INFERSHAPE(SolAttnPrepareRouteStatsKernel).InferShape(ge::InferShape).InferDataType(ge::InferDataType);
}  // namespace ops
