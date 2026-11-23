#include "register/op_impl_registry.h"
#include <cstdint>

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    const int32_t B = (int32_t)*attrs->GetAttrPointer<int64_t>(0);
    const int32_t H = (int32_t)*attrs->GetAttrPointer<int64_t>(2);
    const int32_t NT = (int32_t)*attrs->GetAttrPointer<int64_t>(3);
    const int32_t NPAD = (int32_t)*attrs->GetAttrPointer<int64_t>(5);
    const int32_t T_PAD_KV = (int32_t)*attrs->GetAttrPointer<int64_t>(6);
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
    gert::Shape* kc_padded_shape = context->GetOutputShape(2);
    kc_padded_shape->SetDimNum(0);
    kc_padded_shape->AppendDim(B);
    kc_padded_shape->AppendDim(H);
    kc_padded_shape->AppendDim(NPAD);
    kc_padded_shape->AppendDim(128);
    gert::Shape* vc_padded_shape = context->GetOutputShape(3);
    vc_padded_shape->SetDimNum(0);
    vc_padded_shape->AppendDim(B);
    vc_padded_shape->AppendDim(H);
    vc_padded_shape->AppendDim(NPAD);
    vc_padded_shape->AppendDim(128);
    gert::Shape* kc_pad_f32_shape = context->GetOutputShape(4);
    kc_pad_f32_shape->SetDimNum(0);
    kc_pad_f32_shape->AppendDim(B);
    kc_pad_f32_shape->AppendDim(H);
    kc_pad_f32_shape->AppendDim(NPAD);
    kc_pad_f32_shape->AppendDim(128);
    gert::Shape* q_bar_f32_shape = context->GetOutputShape(5);
    q_bar_f32_shape->SetDimNum(0);
    q_bar_f32_shape->AppendDim(B);
    q_bar_f32_shape->AppendDim(H);
    q_bar_f32_shape->AppendDim(NT);
    q_bar_f32_shape->AppendDim(128);
    gert::Shape* last_group_lens_shape = context->GetOutputShape(6);
    last_group_lens_shape->SetDimNum(0);
    last_group_lens_shape->AppendDim(64);
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
    context->SetOutputDataType(6, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {
IMPL_OP_INFERSHAPE(SolAttnPrepareQkvKernel).InferShape(ge::InferShape).InferDataType(ge::InferDataType);
}  // namespace ops
