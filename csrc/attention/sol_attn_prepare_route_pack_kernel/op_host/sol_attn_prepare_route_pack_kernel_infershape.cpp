#include "register/op_impl_registry.h"
#include <cstdint>

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    const int32_t HC = (int32_t)*attrs->GetAttrPointer<int64_t>(0);
    const int32_t NT_Q = (int32_t)*attrs->GetAttrPointer<int64_t>(2);
    const int32_t NG = (int32_t)*attrs->GetAttrPointer<int64_t>(3);
    const int32_t NPAD = (int32_t)*attrs->GetAttrPointer<int64_t>(4);
    const int32_t NT_Q_PAD8 = (int32_t)*attrs->GetAttrPointer<int64_t>(5);
    gert::Shape* group_approx_mask_shape = context->GetOutputShape(0);
    group_approx_mask_shape->SetDimNum(0);
    group_approx_mask_shape->AppendDim(HC);
    group_approx_mask_shape->AppendDim(NT_Q);
    group_approx_mask_shape->AppendDim(NG);
    gert::Shape* flat_exact_cnt_shape = context->GetOutputShape(1);
    flat_exact_cnt_shape->SetDimNum(0);
    flat_exact_cnt_shape->AppendDim(HC);
    flat_exact_cnt_shape->AppendDim(NT_Q_PAD8);
    gert::Shape* flat_exact_idx_shape = context->GetOutputShape(2);
    flat_exact_idx_shape->SetDimNum(0);
    flat_exact_idx_shape->AppendDim(HC);
    flat_exact_idx_shape->AppendDim(NT_Q);
    flat_exact_idx_shape->AppendDim(NPAD);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_UINT64);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_UINT16);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {
IMPL_OP_INFERSHAPE(SolAttnPrepareRoutePackKernel).InferShape(ge::InferShape).InferDataType(ge::InferDataType);
}  // namespace ops
