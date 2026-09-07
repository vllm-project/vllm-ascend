/**
 * @file dsa_local_metadata_infershape.cpp
 * @brief InferShape and InferDataType for DsaLocalMetadata
 */

#include "register/op_def_registry.h"
#include "log/ops_log.h"

#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                           \
    do {                                                                                                   \
        if (unlikely((ptr) == nullptr)) {                                                                  \
            const char *name = (unlikely(((context) == nullptr) || (context)->GetNodeName() == nullptr)) ? \
                                   "nil" :                                                                 \
                                   (context)->GetNodeName();                                               \
            OPS_LOG_E(name, "%s is nullptr!", #ptr);                                                       \
            return ge::GRAPH_FAILED;                                                                       \
        }                                                                                                  \
    } while (0)

static constexpr int IDX_QUERY_START_LOC = 0;
static constexpr int IDX_SEQ_LENS = 1;

static constexpr int OUT_LOCAL_QUERY_START_LOC = 0;
static constexpr int OUT_LOCAL_SEQ_LENS = 1;
static constexpr int OUT_START_POS = 2;
static constexpr int OUTPUT_NUM = 3;

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4DsaLocalMetadata(gert::InferShapeContext *context)
{
    const gert::Shape *queryStartLocShape = context->GetInputShape(IDX_QUERY_START_LOC);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryStartLocShape);
    const gert::Shape *seqLensShape = context->GetInputShape(IDX_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqLensShape);

    gert::Shape *outShapes[OUTPUT_NUM];
    for (int i = 0; i < OUTPUT_NUM; ++i) {
        outShapes[i] = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outShapes[i]);
    }

    // local_query_start_loc: same shape as query_start_loc [num_reqs + 1]
    outShapes[OUT_LOCAL_QUERY_START_LOC]->SetDimNum(1);
    outShapes[OUT_LOCAL_QUERY_START_LOC]->SetDim(0, queryStartLocShape->GetDim(0));
    // local_seq_lens: same shape as seq_lens [num_reqs]
    outShapes[OUT_LOCAL_SEQ_LENS]->SetDimNum(1);
    outShapes[OUT_LOCAL_SEQ_LENS]->SetDim(0, seqLensShape->GetDim(0));
    // start_pos_out: same shape as seq_lens [num_reqs]
    outShapes[OUT_START_POS]->SetDimNum(1);
    outShapes[OUT_START_POS]->SetDim(0, seqLensShape->GetDim(0));

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4DsaLocalMetadata(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(OUT_LOCAL_QUERY_START_LOC, DT_INT32);
    context->SetOutputDataType(OUT_LOCAL_SEQ_LENS, DT_INT32);
    context->SetOutputDataType(OUT_START_POS, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DsaLocalMetadata)
    .InferShape(InferShape4DsaLocalMetadata)
    .InferDataType(InferDataType4DsaLocalMetadata);

} // namespace ops
