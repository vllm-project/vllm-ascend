/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

using namespace gert;

namespace ops {
namespace {
constexpr size_t MIXED_INDEX = 0;
constexpr size_t STATE_INDEX = 5;
constexpr size_t OUT_INDEX = 0;
constexpr size_t STATE_OUT_INDEX = 1;
} // namespace

static ge::graphStatus InferShapeFusedGdnDecode(InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto mixedShape = context->GetInputShape(MIXED_INDEX);
    auto outShape = context->GetOutputShape(OUT_INDEX);
    auto stateOutShape = context->GetOutputShape(STATE_OUT_INDEX);
    auto stateShape = context->GetInputShape(STATE_INDEX);
    if (mixedShape == nullptr || stateShape == nullptr || outShape == nullptr || stateOutShape == nullptr) {
        OP_LOGE("FusedGdnDecode", "infershape null shape: mixed=%p state=%p out=%p stateOut=%p", mixedShape,
                stateShape, outShape, stateOutShape);
        return ge::GRAPH_FAILED;
    }
    if (mixedShape->GetDimNum() < 2 || stateShape->GetDimNum() < 4) {
        OP_LOGE("FusedGdnDecode", "infershape invalid dim num: mixed=%zu state=%zu", mixedShape->GetDimNum(),
                stateShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    outShape->SetDimNum(4);
    outShape->SetDim(0, mixedShape->GetDim(0));
    outShape->SetDim(1, 1);
    outShape->SetDim(2, stateShape->GetDim(1));
    outShape->SetDim(3, stateShape->GetDim(2));

    stateOutShape->SetDimNum(4);
    stateOutShape->SetDim(0, stateShape->GetDim(0));
    stateOutShape->SetDim(1, stateShape->GetDim(1));
    stateOutShape->SetDim(2, stateShape->GetDim(2));
    stateOutShape->SetDim(3, stateShape->GetDim(3));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFusedGdnDecode(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(OUT_INDEX, context->GetInputDataType(MIXED_INDEX));
    context->SetOutputDataType(STATE_OUT_INDEX, context->GetInputDataType(STATE_INDEX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedGdnDecode)
    .InferShape(InferShapeFusedGdnDecode)
    .InferDataType(InferDataTypeFusedGdnDecode);

} // namespace ops
