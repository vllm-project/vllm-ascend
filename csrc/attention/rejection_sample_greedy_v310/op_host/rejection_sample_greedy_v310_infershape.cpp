/**
 * @file rejection_sample_greedy_v310_infershape.cpp
 * @brief InferShape and InferDataType for RejectionSampleGreedyV310
 */

#include "log/ops_log.h"
#include "register/op_def_registry.h"

namespace ops {

static constexpr int INPUT_CUMULATIVE_COUNTS = 0;
static constexpr int OUTPUT_TOKEN_IDS = 0;
static constexpr int ATTR_ALIGNED_OUTPUT_LEN = 1;

static ge::graphStatus InferShape4RejectionSampleGreedyV310(gert::InferShapeContext* context)
{
    const gert::Shape* cumulativeShape = context->GetInputShape(INPUT_CUMULATIVE_COUNTS);
    if (cumulativeShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* alignedOutputLen = attrs->GetAttrPointer<int64_t>(ATTR_ALIGNED_OUTPUT_LEN);
    if (alignedOutputLen == nullptr || *alignedOutputLen <= 0) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(OUTPUT_TOKEN_IDS);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batchSize = cumulativeShape->GetDimNum() > 0 ? cumulativeShape->GetDim(0) : 0;
    outputShape->SetDimNum(2);
    outputShape->SetDim(0, batchSize);
    outputShape->SetDim(1, *alignedOutputLen);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4RejectionSampleGreedyV310(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_TOKEN_IDS, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RejectionSampleGreedyV310)
    .InferShape(InferShape4RejectionSampleGreedyV310)
    .InferDataType(InferDataType4RejectionSampleGreedyV310);

}  // namespace ops
