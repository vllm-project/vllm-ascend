// SPDX-License-Identifier: Apache-2.0
#include "graph/types.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t kModeAttr = 15;
constexpr size_t kErrorRowsInput = 7;
constexpr size_t kStatsOutput = 11;
constexpr size_t kStatsWidth = 2;

constexpr uint32_t kModeResolve = 0;
constexpr uint32_t kModeAccumulate = 2;

void set_output_shape(gert::InferShapeContext* context, size_t idx, const gert::Shape& shape) {
    auto* output = context->GetOutputShape(idx);
    if (output != nullptr) *output = shape;
}
}  // namespace

// One infershape entry serves the two native launch modes selected by the
// `mode` attribute: accumulate produces a [2] int64 stats tensor, resolve
// infers the residency outputs. Invalidation runs as a Triton kernel without
// touching this aclnn op.
static ge::graphStatus InferShape4HiSparseResolveResidency(gert::InferShapeContext* context) {
    if (context == nullptr) return ge::GRAPH_FAILED;
    const auto* attrs = context->GetAttrs();
    const auto* mode = attrs != nullptr ? attrs->GetInt(kModeAttr) : nullptr;
    if (mode == nullptr) return ge::GRAPH_FAILED;

    if (*mode == kModeAccumulate) {
        const gert::Shape* errors = context->GetInputShape(kErrorRowsInput);
        if (errors == nullptr || errors->GetDimNum() != 2) return ge::GRAPH_FAILED;
        gert::Shape statsShape;
        statsShape.SetDimNum(1);
        statsShape.SetDim(0, kStatsWidth);
        set_output_shape(context, kStatsOutput, statsShape);
        return ge::GRAPH_SUCCESS;
    }
    // Resolve mode: outputs are bound by the framework from the launch, so
    // there is nothing to infer beyond acknowledging the inputs.
    if (context->GetInputShape(0) == nullptr) return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4HiSparseResolveResidency(gert::InferDataTypeContext* context) {
    if (context == nullptr) return ge::GRAPH_FAILED;
    const auto* attrs = context->GetAttrs();
    const auto* mode = attrs != nullptr ? attrs->GetInt(kModeAttr) : nullptr;
    if (mode != nullptr && *mode == kModeAccumulate) {
        context->SetOutputDataType(kStatsOutput, ge::DT_INT64);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HiSparseResolveResidency)
    .InferShape(InferShape4HiSparseResolveResidency)
    .InferDataType(InferDataType4HiSparseResolveResidency);
}  // namespace ops
