#include "register/op_impl_registry.h"
#include <cstdint>

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    const int32_t B = (int32_t)*attrs->GetAttrPointer<int64_t>(0);
    const int32_t H = (int32_t)*attrs->GetAttrPointer<int64_t>(2);
    const int32_t NT = (int32_t)*attrs->GetAttrPointer<int64_t>(3);
    const int32_t T_PAD_KV = (int32_t)*attrs->GetAttrPointer<int64_t>(4);
    const int32_t NT_Q = (int32_t)*attrs->GetAttrPointer<int64_t>(5);
    const int32_t NT_Q_PAD8 = (int32_t)*attrs->GetAttrPointer<int64_t>(6);
    gert::Shape* k_pad_shape = context->GetOutputShape(0);
    k_pad_shape->SetDimNum(0);
    k_pad_shape->AppendDim(B);
    k_pad_shape->AppendDim(H);
    k_pad_shape->AppendDim(T_PAD_KV);
    k_pad_shape->AppendDim(128);
    gert::Shape* v_pad_shape = context->GetOutputShape(1);
    v_pad_shape->SetDimNum(0);
    v_pad_shape->AppendDim(B);
    v_pad_shape->AppendDim(H);
    v_pad_shape->AppendDim(T_PAD_KV);
    v_pad_shape->AppendDim(128);
    gert::Shape* kc_shape = context->GetOutputShape(2);
    kc_shape->SetDimNum(0);
    kc_shape->AppendDim(B);
    kc_shape->AppendDim(H);
    kc_shape->AppendDim(64);
    kc_shape->AppendDim(128);
    gert::Shape* vc_shape = context->GetOutputShape(3);
    vc_shape->SetDimNum(0);
    vc_shape->AppendDim(B);
    vc_shape->AppendDim(H);
    vc_shape->AppendDim(64);
    vc_shape->AppendDim(128);
    gert::Shape* last_group_lens_shape = context->GetOutputShape(4);
    last_group_lens_shape->SetDimNum(0);
    last_group_lens_shape->AppendDim(64);
    gert::Shape* threshold_shape = context->GetOutputShape(5);
    threshold_shape->SetDimNum(0);
    threshold_shape->AppendDim(B);
    threshold_shape->AppendDim(NT);
    threshold_shape->AppendDim(H);
    gert::Shape* approx_mask_shape = context->GetOutputShape(6);
    approx_mask_shape->SetDimNum(0);
    approx_mask_shape->AppendDim(B);
    approx_mask_shape->AppendDim(H);
    approx_mask_shape->AppendDim(NT_Q);
    approx_mask_shape->AppendDim(1);
    gert::Shape* exact_counts_shape = context->GetOutputShape(7);
    exact_counts_shape->SetDimNum(0);
    exact_counts_shape->AppendDim(B);
    exact_counts_shape->AppendDim(H);
    exact_counts_shape->AppendDim(NT_Q_PAD8);
    gert::Shape* exact_indices_shape = context->GetOutputShape(8);
    exact_indices_shape->SetDimNum(0);
    exact_indices_shape->AppendDim(B);
    exact_indices_shape->AppendDim(H);
    exact_indices_shape->AppendDim(NT_Q);
    exact_indices_shape->AppendDim(64);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_BF16);
    context->SetOutputDataType(2, ge::DT_BF16);
    context->SetOutputDataType(3, ge::DT_BF16);
    context->SetOutputDataType(4, ge::DT_FLOAT);
    context->SetOutputDataType(5, ge::DT_FLOAT);
    context->SetOutputDataType(6, ge::DT_UINT64);
    context->SetOutputDataType(7, ge::DT_INT32);
    context->SetOutputDataType(8, ge::DT_UINT16);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {
IMPL_OP_INFERSHAPE(SolAttnPrepareShortKernel).InferShape(ge::InferShape).InferDataType(ge::InferDataType);
}  // namespace ops
