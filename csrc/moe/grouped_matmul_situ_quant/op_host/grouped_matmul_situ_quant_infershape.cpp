#include <register/op_impl_registry.h>
#include "runtime/infer_shape_context.h"

namespace ops {
namespace {

constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_WEIGHT_INDEX = 2;
constexpr size_t INPUT_WEIGHT_SCALE_INDEX = 3;
constexpr size_t INPUT_GROUP_LIST_INDEX = 4;
constexpr size_t OUTPUT_Y_INDEX = 0;
constexpr size_t OUTPUT_Y_SCALE_INDEX = 1;
constexpr int64_t MX_BLOCK_SPAN = 64;
constexpr int64_t MX_SCALE_ALIGN = 2;
// SiTU-GLU 把 gate/up 两半并成一半宽。
constexpr int64_t SWIGLU_SPLIT_RATIO = 2;

} // namespace

ge::graphStatus InferShape4GroupedMatmulSituQuant(gert::InferShapeContext *context)
{
    const auto *xShape = context->GetDynamicInputShape(INPUT_X_INDEX, 0);
    auto *yShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    auto *yScaleShape = context->GetOutputShape(OUTPUT_Y_SCALE_INDEX);
    if (xShape == nullptr || yShape == nullptr || yScaleShape == nullptr || xShape->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    // Atlas A2/A3 的 A8W4 MSD：per-token 对称量化，N 从 weight_scale 的最内维推出
    // （单张量堆叠为 (E, N)，TensorList 逐专家为 (N)），y 为激活后的一半宽，y_scale 逐 token。
    const auto *xDesc = context->GetInputDesc(INPUT_X_INDEX);
    if (xDesc != nullptr && xDesc->GetDataType() == ge::DT_INT8) {
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

    const auto *weightShape = context->GetDynamicInputShape(INPUT_WEIGHT_INDEX, 0);
    const auto *groupListShape = context->GetDynamicInputShape(INPUT_GROUP_LIST_INDEX, 0);
    if (weightShape == nullptr || groupListShape == nullptr || weightShape->GetDimNum() < 1 ||
        groupListShape->GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t m = xShape->GetDim(0);
    const int64_t k = xShape->GetDim(1);
    const bool isTensorList = context->GetDynamicInputShape(INPUT_WEIGHT_INDEX, 1) != nullptr;
    const int64_t expertCount = isTensorList ? groupListShape->GetDim(0) : weightShape->GetDim(0);
    const int64_t packedWeightElements = weightShape->GetShapeSize();
    if (k <= 0 || expertCount <= 0 || k % 2 != 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t packedElementsPerColumn = (isTensorList ? 1 : expertCount) * (k / 2);
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
    if (context->GetInputDataType(INPUT_X_INDEX) == ge::DT_INT8) {
        context->SetOutputDataType(OUTPUT_Y_INDEX, ge::DT_INT8);
        context->SetOutputDataType(OUTPUT_Y_SCALE_INDEX, ge::DT_FLOAT);
        return ge::GRAPH_SUCCESS;
    }
    context->SetOutputDataType(OUTPUT_Y_INDEX, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(OUTPUT_Y_SCALE_INDEX, ge::DT_FLOAT8_E8M0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmulSituQuant)
    .InferShape(InferShape4GroupedMatmulSituQuant)
    .InferDataType(InferDataType4GroupedMatmulSituQuant);

} // namespace ops
