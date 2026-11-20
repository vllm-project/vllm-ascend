// SPDX-License-Identifier: Apache-2.0
#include "graph/types.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t kStatsOutput = 11;
constexpr size_t kStatsWidth = 2;

void set_output_shape(gert::InferShapeContext* context, size_t idx, const gert::Shape& shape) {
    auto* output = context->GetOutputShape(idx);
    if (output != nullptr) *output = shape;
}
}  // namespace

// The resolve launch binds every output from the framework, so infershape
// only acknowledges the inputs and pins the stats output shape. Invalidation
// runs as a Triton kernel without touching this aclnn op.
static ge::graphStatus InferShape4HisparseResolveResidency(gert::InferShapeContext* context) {
    if (context == nullptr) return ge::GRAPH_FAILED;
    if (context->GetInputShape(0) == nullptr) return ge::GRAPH_FAILED;
    gert::Shape statsShape;
    statsShape.SetDimNum(1);
    statsShape.SetDim(0, kStatsWidth);
    set_output_shape(context, kStatsOutput, statsShape);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4HisparseResolveResidency(gert::InferDataTypeContext* context) {
    if (context == nullptr) return ge::GRAPH_FAILED;
    context->SetOutputDataType(kStatsOutput, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HisparseResolveResidency)
    .InferShape(InferShape4HisparseResolveResidency)
    .InferDataType(InferDataType4HisparseResolveResidency);
}  // namespace ops
