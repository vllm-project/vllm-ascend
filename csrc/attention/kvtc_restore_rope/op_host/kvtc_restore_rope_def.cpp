#include "register/op_def_registry.h"

#include <cstddef>
#include <cstdint>

namespace {

constexpr size_t RECONSTRUCTED_IR_INDEX = 0;
constexpr size_t DST_BLOCK_IDS_IR_INDEX = 1;
constexpr size_t TOKEN_STARTS_IR_INDEX = 2;
constexpr size_t COS_SIN_IR_INDEX = 3;
constexpr size_t CACHES_IR_INDEX = 4;

ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* reconstructed =
        context->GetRequiredInputShape(RECONSTRUCTED_IR_INDEX);
    const gert::Shape* dstIds =
        context->GetRequiredInputShape(DST_BLOCK_IDS_IR_INDEX);
    const gert::Shape* tokenStarts =
        context->GetRequiredInputShape(TOKEN_STARTS_IR_INDEX);
    const gert::Shape* cosSin =
        context->GetRequiredInputShape(COS_SIN_IR_INDEX);
    const gert::Shape* firstCache =
        context->GetDynamicInputShape(CACHES_IR_INDEX, 0);
    gert::Shape* status = context->GetOutputShape(0);

    if (reconstructed == nullptr || dstIds == nullptr ||
        tokenStarts == nullptr || cosSin == nullptr ||
        firstCache == nullptr || status == nullptr ||
        reconstructed->GetDimNum() != 2 || dstIds->GetDimNum() != 1 ||
        tokenStarts->GetDimNum() != 1 || cosSin->GetDimNum() != 2 ||
        firstCache->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const int64_t g = firstCache->GetDim(0);
    const int64_t b = firstCache->GetDim(1);
    const int64_t h = firstCache->GetDim(2);
    const int64_t d = firstCache->GetDim(3);
    const int64_t s = dstIds->GetDim(0);
    if (g <= 0 || b <= 0 || h <= 0 || d <= 0 || s <= 0 ||
        tokenStarts->GetDim(0) != s || cosSin->GetDim(0) <= 0 ||
        cosSin->GetDim(1) != d || d % 16 != 0 || d > 128) {
        return ge::GRAPH_FAILED;
    }

    int64_t layers = 0;
    while (true) {
        const gert::Shape* cache = context->GetDynamicInputShape(
            CACHES_IR_INDEX, static_cast<size_t>(layers));
        if (cache == nullptr) {
            break;
        }
        if (cache->GetDimNum() != 4 || cache->GetDim(0) != g ||
            cache->GetDim(1) != b || cache->GetDim(2) != h ||
            cache->GetDim(3) != d) {
            return ge::GRAPH_FAILED;
        }
        ++layers;
    }
    if (layers <= 0 || reconstructed->GetDim(0) != s * b ||
        reconstructed->GetDim(1) != layers * h * d) {
        return ge::GRAPH_FAILED;
    }

    status->SetDimNum(1);
    status->SetDim(0, 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

} // namespace

namespace ops {

class KvtcRestoreRope : public OpDef {
public:
    explicit KvtcRestoreRope(const char* name) : OpDef(name)
    {
        this->Input("reconstructed")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("dst_block_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("token_starts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // [rope_cache_rows,D], FP32, layout [cos | +sin].
        this->Input("cos_sin_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("caches")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("status")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(InferShape);
        this->SetInferDataType(InferDataType);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(KvtcRestoreRope);

} // namespace ops
