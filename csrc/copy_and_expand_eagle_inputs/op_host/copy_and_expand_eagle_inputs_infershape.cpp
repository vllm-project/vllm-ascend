/**
 * @file copy_and_expand_eagle_inputs_infershape.cpp
 * @brief InferShape and InferDataType for CopyAndExpandEagleInputs
 */

#include "register/op_def_registry.h"
#include "log/ops_log.h"

#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                           \
    do {                                                                                                   \
        if (unlikely((ptr) == nullptr)) {                                                                  \
            const char* name = (unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr)) ? \
                                   "nil" :                                                                 \
                                   (context)->GetNodeName();                                               \
            OPS_LOG_E(name, "%s is nullptr!", #ptr);                                                       \
            return ge::GRAPH_FAILED;                                                                       \
        }                                                                                                  \
    } while (0)

static constexpr int IDX_TARGET_TOKEN_IDS = 0;
static constexpr int IDX_TARGET_POSITIONS = 1;
static constexpr int IDX_NEXT_TOKEN_IDS = 2;
static constexpr int IDX_QUERY_START_LOC = 3;
static constexpr int IDX_QUERY_END_LOC = 4;

static constexpr int OUT_INPUT_IDS = 0;
static constexpr int OUT_POSITIONS = 1;
static constexpr int OUT_REJECTED_MASK = 2;
static constexpr int OUT_MASKED_MASK = 3;
static constexpr int OUT_NEW_TOKEN_INDICES = 4;
static constexpr int OUT_HIDDEN_STATE_MAPPING = 5;
static constexpr int OUTPUT_NUM = 6;

static constexpr int ATTR_NUM_PADDING_SLOTS = 2;
static constexpr int ATTR_TOTAL_INPUT_TOKENS = 4;

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4CopyAndExpandEagleInputs(gert::InferShapeContext* context)
{
    // Get input shapes
    const gert::Shape* targetTokenIdsShape = context->GetInputShape(IDX_TARGET_TOKEN_IDS);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetTokenIdsShape);
    const gert::Shape* queryStartLocShape = context->GetInputShape(IDX_QUERY_START_LOC);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryStartLocShape);

    // Derive dimensions from input shapes
    int64_t totalInputTokens = targetTokenIdsShape->GetDim(0);
    int64_t numReqs = queryStartLocShape->GetDim(0) - 1;

    // Get num_padding_slots_per_request from attributes
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t numPaddingSlotsPerReq = *(attrs->GetAttrPointer<int64_t>(ATTR_NUM_PADDING_SLOTS));

    // Compute total_draft_tokens = total_input_tokens + (num_padding_slots_per_request - 1) * num_reqs
    int64_t totalDraftTokens = totalInputTokens + (numPaddingSlotsPerReq - 1) * numReqs;

    // Get and validate all output shapes
    gert::Shape* outShapes[OUTPUT_NUM];
    for (int i = 0; i < OUTPUT_NUM; ++i) {
        outShapes[i] = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outShapes[i]);
        outShapes[i]->SetDimNum(1);
    }

    // out_input_ids, out_positions, out_rejected_mask, out_masked_mask: [total_draft_tokens]
    outShapes[OUT_INPUT_IDS]->SetDim(0, totalDraftTokens);
    outShapes[OUT_POSITIONS]->SetDim(0, totalDraftTokens);
    outShapes[OUT_REJECTED_MASK]->SetDim(0, totalDraftTokens);
    outShapes[OUT_MASKED_MASK]->SetDim(0, totalDraftTokens);

    // out_new_token_indices: [num_reqs * num_padding_slots_per_request]
    outShapes[OUT_NEW_TOKEN_INDICES]->SetDim(0, numReqs * numPaddingSlotsPerReq);

    // out_hidden_state_mapping: [total_input_tokens]
    outShapes[OUT_HIDDEN_STATE_MAPPING]->SetDim(0, totalInputTokens);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4CopyAndExpandEagleInputs(gert::InferDataTypeContext* context)
{
    // out_input_ids: INT32
    context->SetOutputDataType(OUT_INPUT_IDS, DT_INT32);
    // out_positions: INT32
    context->SetOutputDataType(OUT_POSITIONS, DT_INT32);
    // out_is_rejected_token_mask: INT8
    context->SetOutputDataType(OUT_REJECTED_MASK, DT_INT8);
    // out_is_masked_token_mask: INT8
    context->SetOutputDataType(OUT_MASKED_MASK, DT_INT8);
    // out_new_token_indices: INT32
    context->SetOutputDataType(OUT_NEW_TOKEN_INDICES, DT_INT32);
    // out_hidden_state_mapping: INT32
    context->SetOutputDataType(OUT_HIDDEN_STATE_MAPPING, DT_INT32);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CopyAndExpandEagleInputs)
    .InferShape(InferShape4CopyAndExpandEagleInputs)
    .InferDataType(InferDataType4CopyAndExpandEagleInputs);

} // namespace ops
