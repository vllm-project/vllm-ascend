#include <register/op_impl_registry.h>
#include "runtime/infer_shape_context.h"

namespace ops {
namespace {

constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_WEIGHT_SCALE_INDEX = 3;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t OUTPUT_Y_SCALE_INDEX = 1;
// SiTU-GLU 把 gate/up 两半并成一半宽。
constexpr int64_t SWIGLU_SPLIT_RATIO = 2;

} // namespace

ge::graphStatus InferShape4GroupedMatmulSituQuantMsd(gert::InferShapeContext *context)
{
    const auto *xShape = context->GetDynamicInputShape(INPUT_X_INDEX, 0);
    auto *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    auto *yScaleShape = context->GetOutputShape(OUTPUT_Y_SCALE_INDEX);
    if (xShape == nullptr || yShape == nullptr || yScaleShape == nullptr || xShape->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const auto *weightScaleShape = context->GetDynamicInputShape(INPUT_WEIGHT_SCALE_INDEX, 0);
    if (weightScaleShape == nullptr || weightScaleShape->GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t weightScaleDim = weightScaleShape->GetDim(weightScaleShape->GetDimNum() - 1);
    if (weightScaleDim <= 0 || weightScaleDim % SWIGLU_SPLIT_RATIO != 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t m = xShape->GetDim(0);
    *yShape = gert::Shape({m, weightScaleDim / SWIGLU_SPLIT_RATIO});
    *yScaleShape = gert::Shape({m});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4GroupedMatmulSituQuantMsd(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(OUTPUT_Y_INDEX, ge::DT_INT8);
    context->SetOutputDataType(OUTPUT_Y_SCALE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmulSituQuantMsd)
    .InferShape(InferShape4GroupedMatmulSituQuantMsd)
    .InferDataType(InferDataType4GroupedMatmulSituQuantMsd);

} // namespace ops
