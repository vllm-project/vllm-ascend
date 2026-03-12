#include "register/op_impl_registry.h"


#include <string>

namespace ge {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t GROUPLIST_INDEX = 2;

static ge::graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape* x_shape = context->GetDynamicInputShape(X_INDEX, 0);
    const gert::Shape* weight_shape = context->GetDynamicInputShape(WEIGHT_INDEX, 0);
    bool transpose_weight = static_cast<bool>(*(context->GetAttrs()->GetAttrPointer<bool>(0)));
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    auto weight_desc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    auto weight_format = static_cast<ge::Format>(ge::GetPrimaryFormat(weight_desc->GetStorageFormat()));
    bool weight_nz = weight_format == ge::FORMAT_FRACTAL_NZ;
    int64_t dim_n;
    if (weight_nz) {
      dim_n = transpose_weight ? (weight_shape->GetDim(1) * weight_shape->GetDim(3)) :
                                 (weight_shape->GetDim(2) * weight_shape->GetDim(4));
    } else {
      dim_n = transpose_weight ? weight_shape->GetDim(1) : weight_shape->GetDim(2);
    }
    y_shape->SetDim(1, dim_n);
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto input_dtype = context->GetDynamicInputDataType(X_INDEX, 0);
    context->SetOutputDataType(0, input_dtype);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge