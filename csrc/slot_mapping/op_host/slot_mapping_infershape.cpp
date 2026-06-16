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

using namespace ge;

namespace ops {

static constexpr int ATTR_MAX_NUM_TOKENS = 1;
static constexpr int OUT_SLOT_MAPPING = 0;

static ge::graphStatus InferShape4SlotMapping(gert::InferShapeContext* context)
{
    gert::Shape* slotMappingShape = context->GetOutputShape(OUT_SLOT_MAPPING);
    OP_CHECK_NULL_WITH_CONTEXT(context, slotMappingShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* pMaxNumTokens = attrs->GetAttrPointer<int64_t>(ATTR_MAX_NUM_TOKENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, pMaxNumTokens);

    slotMappingShape->SetDimNum(1);
    slotMappingShape->SetDim(0, *pMaxNumTokens);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4SlotMapping(gert::InferDataTypeContext* context)
{
    // Aligned with slot_mapping_def.cpp output dtype (int32).
    context->SetOutputDataType(OUT_SLOT_MAPPING, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SlotMapping)
    .InferShape(InferShape4SlotMapping)
    .InferDataType(InferDataType4SlotMapping);

}  // namespace ops
