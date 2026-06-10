#include "register/op_def_registry.h"

using namespace ge;

namespace ops {

static graphStatus InferShapeForGumbelSample(gert::InferShapeContext* context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    // idx_mapping（输入 1）的 dim0 = num_tokens
    const gert::Shape* idxMappingShape = context->GetInputShape(1);
    if (idxMappingShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (idxMappingShape->GetDimNum() < 1) {
        return GRAPH_FAILED;
    }

    gert::Shape* sampledShape = context->GetOutputShape(0);
    if (sampledShape == nullptr) {
        return GRAPH_FAILED;
    }
    sampledShape->SetDimNum(1);
    sampledShape->SetDim(0, idxMappingShape->GetDim(0));   // sampled.shape = [num_reqs]

    // output_processed_logits 是调用方传入的可选输出 buffer，形状保持调用方指定值。
    gert::Shape* processedShape = context->GetOutputShape(1);
    if (processedShape != nullptr) {
        const gert::Shape* currentProcessedShape = context->GetOutputShape(1);
        if (currentProcessedShape != nullptr && currentProcessedShape->GetDimNum() == 3) {
            processedShape->SetDimNum(3);
            processedShape->SetDim(0, currentProcessedShape->GetDim(0));
            processedShape->SetDim(1, currentProcessedShape->GetDim(1));
            processedShape->SetDim(2, currentProcessedShape->GetDim(2));
        }
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForGumbelSample(gert::InferDataTypeContext* context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, DT_INT64);
    context->SetOutputDataType(1, DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GumbelSample)
    .InferShape(InferShapeForGumbelSample)
    .InferDataType(InferDataTypeForGumbelSample);

}  // namespace ops
