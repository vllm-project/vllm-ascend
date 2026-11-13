#include "register/op_def_registry.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr size_t CACHES_IR_INDEX = 0;
constexpr size_t BLOCK_IDS_IR_INDEX = 1;
constexpr size_t TOKEN_STARTS_IR_INDEX = 2;
constexpr size_t INV_ROPE_IR_INDEX = 3;

ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* firstCache =
        context->GetDynamicInputShape(CACHES_IR_INDEX, 0);
    const gert::Shape* blockIds =
        context->GetRequiredInputShape(BLOCK_IDS_IR_INDEX);
    const gert::Shape* tokenStarts =
        context->GetRequiredInputShape(TOKEN_STARTS_IR_INDEX);
    const gert::Shape* invRope =
        context->GetRequiredInputShape(INV_ROPE_IR_INDEX);
    gert::Shape* output = context->GetOutputShape(0);

    if (firstCache == nullptr || blockIds == nullptr ||
        tokenStarts == nullptr || invRope == nullptr || output == nullptr ||
        firstCache->GetDimNum() != 4 || blockIds->GetDimNum() != 1 ||
        tokenStarts->GetDimNum() != 1 || invRope->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t numGpuBlocks = firstCache->GetDim(0);
    const int64_t blockSize = firstCache->GetDim(1);
    const int64_t numKvHeads = firstCache->GetDim(2);
    const int64_t headSize = firstCache->GetDim(3);
    const int64_t selectedBlocks = blockIds->GetDim(0);
    if (numGpuBlocks <= 0 || blockSize <= 0 || numKvHeads <= 0 ||
        headSize <= 0 || selectedBlocks <= 0 ||
        tokenStarts->GetDim(0) != selectedBlocks ||
        invRope->GetDim(0) <= 0 || invRope->GetDim(1) != headSize) {
        return ge::GRAPH_FAILED;
    }

    int64_t numLayers = 0;
    while (true) {
        const gert::Shape* cache = context->GetDynamicInputShape(
            CACHES_IR_INDEX, static_cast<size_t>(numLayers));
        if (cache == nullptr) {
            break;
        }
        if (cache->GetDimNum() != 4 ||
            cache->GetDim(0) != numGpuBlocks ||
            cache->GetDim(1) != blockSize ||
            cache->GetDim(2) != numKvHeads ||
            cache->GetDim(3) != headSize) {
            return ge::GRAPH_FAILED;
        }
        ++numLayers;
    }
    if (numLayers <= 0 || headSize % 16 != 0 || headSize > 128 ||
        numLayers > std::numeric_limits<int64_t>::max() /
            numKvHeads / headSize) {
        return ge::GRAPH_FAILED;
    }

    const int64_t m = selectedBlocks * blockSize;
    const int64_t k = numLayers * numKvHeads * headSize;
    if (m <= 0 || k <= 0) {
        return ge::GRAPH_FAILED;
    }

    output->SetDimNum(2);
    output->SetDim(0, m);
    output->SetDim(1, k);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

} // namespace

namespace ops {

class KvtcPrepareRope : public OpDef {
public:
    explicit KvtcPrepareRope(const char* name) : OpDef(name)
    {
        this->Input("caches")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("block_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("token_starts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // [rope_cache_rows, head_size], FP32, layout [cos | -sin].
        this->Input("inv_cos_sin_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("prepared")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(InferShape);
        this->SetInferDataType(InferDataType);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(KvtcPrepareRope);

} // namespace ops
