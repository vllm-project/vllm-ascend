/**
 * @file copy_and_expand_dflash_inputs_infershape.cpp
 * @brief InferShape and InferDataType for CopyAndExpandDflashInputs
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

// Inputs
static constexpr int IDX_NEXT_TOKEN_IDS = 0;
static constexpr int IDX_TARGET_POSITIONS = 1;
static constexpr int IDX_QUERY_START_LOC = 2;
static constexpr int IDX_NUM_REJECTED_TOKENS = 3;
static constexpr int IDX_BLOCK_TABLE = 4;

// Outputs
static constexpr int OUT_INPUT_IDS = 0;
static constexpr int OUT_CONTEXT_POSITIONS = 1;
static constexpr int OUT_QUERY_POSITIONS = 2;
static constexpr int OUT_CONTEXT_SLOT_MAPPING = 3;
static constexpr int OUT_QUERY_SLOT_MAPPING = 4;
static constexpr int OUT_TOKEN_INDICES = 5;
static constexpr int OUTPUT_NUM = 6;

// Attributes
static constexpr int ATTR_NUM_QUERY_PER_REQ = 1;

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4CopyAndExpandDflashInputs(gert::InferShapeContext* context)
{
    // Get input shapes
    const gert::Shape* targetPositionsShape = context->GetInputShape(IDX_TARGET_POSITIONS);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetPositionsShape);
    const gert::Shape* queryStartLocShape = context->GetInputShape(IDX_QUERY_START_LOC);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryStartLocShape);

    // Derive dimensions from input shapes
    int64_t totalInputTokens = targetPositionsShape->GetDim(0);
    int64_t numReqs = queryStartLocShape->GetDim(0) - 1;

    // Get num_query_per_req from attributes
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t numQueryPerReq = *(attrs->GetAttrPointer<int64_t>(ATTR_NUM_QUERY_PER_REQ));

    // Compute total_query_tokens
    int64_t totalQueryTokens = numReqs * numQueryPerReq;

    // Get and validate all output shapes
    gert::Shape* outShapes[OUTPUT_NUM];
    for (int i = 0; i < OUTPUT_NUM; ++i) {
        outShapes[i] = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outShapes[i]);
        outShapes[i]->SetDimNum(1);
    }

    // out_input_ids: [total_query_tokens]
    outShapes[OUT_INPUT_IDS]->SetDim(0, totalQueryTokens);

    // out_context_positions: [total_input_tokens]
    outShapes[OUT_CONTEXT_POSITIONS]->SetDim(0, totalInputTokens);

    // out_query_positions: [total_query_tokens]
    outShapes[OUT_QUERY_POSITIONS]->SetDim(0, totalQueryTokens);

    // out_context_slot_mapping: [total_input_tokens]
    outShapes[OUT_CONTEXT_SLOT_MAPPING]->SetDim(0, totalInputTokens);

    // out_query_slot_mapping: [total_query_tokens]
    outShapes[OUT_QUERY_SLOT_MAPPING]->SetDim(0, totalQueryTokens);

    // out_token_indices: [num_reqs * (num_query_per_req - 1)] (skip bonus token)
    outShapes[OUT_TOKEN_INDICES]->SetDim(0, numReqs * (numQueryPerReq - 1));

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4CopyAndExpandDflashInputs(gert::InferDataTypeContext* context)
{
    // out_input_ids: INT32
    context->SetOutputDataType(OUT_INPUT_IDS, DT_INT32);
    // out_context_positions: INT64
    context->SetOutputDataType(OUT_CONTEXT_POSITIONS, DT_INT64);
    // out_query_positions: INT64
    context->SetOutputDataType(OUT_QUERY_POSITIONS, DT_INT64);
    // out_context_slot_mapping: INT32
    context->SetOutputDataType(OUT_CONTEXT_SLOT_MAPPING, DT_INT32);
    // out_query_slot_mapping: INT32
    context->SetOutputDataType(OUT_QUERY_SLOT_MAPPING, DT_INT32);
    // out_token_indices: INT32
    context->SetOutputDataType(OUT_TOKEN_INDICES, DT_INT32);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CopyAndExpandDflashInputs)
    .InferShape(InferShape4CopyAndExpandDflashInputs)
    .InferDataType(InferDataType4CopyAndExpandDflashInputs);

} // namespace ops