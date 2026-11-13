// SPDX-License-Identifier: Apache-2.0
#include "graph/types.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
// Argument positions follow the registered def order.
constexpr size_t kLogical = 0;
constexpr size_t kTags = 0;
constexpr size_t kLru = 1;
constexpr size_t kPhysical = 2;
constexpr size_t kAttention = 3;
constexpr size_t kResolved = 4;
constexpr size_t kCounts = 5;
constexpr size_t kSwapHost = 6;
constexpr size_t kSwapDevice = 7;
constexpr size_t kSwapCounts = 8;
constexpr size_t kMissMask = 9;
constexpr size_t kStats = 10;
constexpr size_t kErrors = 11;

void set_output_shape(gert::InferShapeContext* context, size_t idx, const gert::Shape& shape) {
    auto* output = context->GetOutputShape(idx);
    if (output != nullptr) *output = shape;
}
}  // namespace

static ge::graphStatus InferShape4HiSparseResolveResidency(gert::InferShapeContext* context) {
    const gert::Shape* logical = context->GetInputShape(kLogical);
    if (logical == nullptr || logical->GetDimNum() != 2) return ge::GRAPH_FAILED;

    const gert::Shape topkShape = *logical;
    gert::Shape rowVec;
    rowVec.SetDimNum(1);
    rowVec.SetDim(0, logical->GetDim(0));
    gert::Shape statsShape;
    statsShape.SetDimNum(1);
    statsShape.SetDim(0, 2);
    gert::Shape errorShape;
    errorShape.SetDimNum(2);
    errorShape.SetDim(0, logical->GetDim(0));
    errorShape.SetDim(1, 8);

    // tags/lru keep the caller-provided hot-slot geometry: their shapes are
    // not derivable from the logical TopK input (opaque launch contract).
    set_output_shape(context, kPhysical, topkShape);
    set_output_shape(context, kAttention, topkShape);
    set_output_shape(context, kResolved, topkShape);
    set_output_shape(context, kCounts, rowVec);
    set_output_shape(context, kSwapHost, rowVec);
    set_output_shape(context, kSwapDevice, rowVec);
    set_output_shape(context, kSwapCounts, rowVec);
    set_output_shape(context, kMissMask, topkShape);
    set_output_shape(context, kStats, statsShape);
    set_output_shape(context, kErrors, errorShape);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4HiSparseResolveResidency(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(kTags, ge::DT_INT32);
    context->SetOutputDataType(kLru, ge::DT_INT16);
    context->SetOutputDataType(kPhysical, ge::DT_INT32);
    context->SetOutputDataType(kAttention, ge::DT_INT32);
    context->SetOutputDataType(kResolved, ge::DT_INT32);
    context->SetOutputDataType(kCounts, ge::DT_INT32);
    context->SetOutputDataType(kSwapHost, ge::DT_INT32);
    context->SetOutputDataType(kSwapDevice, ge::DT_INT32);
    context->SetOutputDataType(kSwapCounts, ge::DT_INT32);
    // miss_mask callers may pass uint8/bool views; int32 is the declared default.
    context->SetOutputDataType(kMissMask, ge::DT_INT32);
    context->SetOutputDataType(kStats, ge::DT_INT64);
    context->SetOutputDataType(kErrors, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HiSparseResolveResidency)
    .InferShape(InferShape4HiSparseResolveResidency)
    .InferDataType(InferDataType4HiSparseResolveResidency);
}  // namespace ops
