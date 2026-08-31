#include <register/op_impl_registry.h>
#include "runtime/infer_shape_context.h"

namespace ops {
namespace {

constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_WEIGHT_INDEX = 2;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t OUTPUT_Y_SCALE_INDEX = 1;
constexpr int64_t MX_BLOCK_SPAN = 64;
constexpr int64_t MX_SCALE_ALIGN = 2;

} // namespace

ge::graphStatus InferShape4GroupedMatmulSituQuant(gert::InferShapeContext *context)
{
    const auto *xShape = context->GetInputShape(INPUT_X_INDEX);
    const auto *weightShape = context->GetInputShape(INPUT_WEIGHT_INDEX);
    auto *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    auto *yScaleShape = context->GetOutputShape(OUTPUT_Y_SCALE_INDEX);
    if (xShape == nullptr || weightShape == nullptr || yShape == nullptr || yScaleShape == nullptr ||
        xShape->GetDimNum() != 2 || weightShape->GetDimNum() < 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t m = xShape->GetDim(0);
    const int64_t k = xShape->GetDim(1);
    const int64_t expertCount = weightShape->GetDim(0);
    const int64_t packedWeightElements = weightShape->GetShapeSize();
    if (k <= 0 || expertCount <= 0 || k % 2 != 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t packedElementsPerColumn = expertCount * (k / 2);
    if (packedElementsPerColumn <= 0 || packedWeightElements % packedElementsPerColumn != 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t n = packedWeightElements / packedElementsPerColumn;
    if (n <= 0 || n % 2 != 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t n2 = n / 2;
    *yShape = gert::Shape({m, n2});
    *yScaleShape = gert::Shape({m, (n2 + MX_BLOCK_SPAN - 1) / MX_BLOCK_SPAN, MX_SCALE_ALIGN});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4GroupedMatmulSituQuant(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(OUTPUT_Y_INDEX, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(OUTPUT_Y_SCALE_INDEX, ge::DT_FLOAT8_E8M0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmulSituQuant)
    .InferShape(InferShape4GroupedMatmulSituQuant)
    .InferDataType(InferDataType4GroupedMatmulSituQuant);

} // namespace ops
