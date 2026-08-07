/**
 * Copyright (c) 2026 TriangleMix contributors.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0 (the "License"). Please refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text.
 */

#include "register/op_def_registry.h"

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* queryShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = *queryShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class TrianglePagedSparseAttention : public OpDef {
public:
    explicit TrianglePagedSparseAttention(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("query_start").Int();
        this->Attr("seq_len").Int();
        this->Attr("prompt_len").Int();
        this->Attr("scale").Float();
        this->Attr("q_tile").AttrType(OPTIONAL).Int(32);
        this->Attr("page_size").AttrType(OPTIONAL).Int(128);
        this->Attr("sink_tokens").AttrType(OPTIONAL).Int(8);
        this->Attr("local_window").AttrType(OPTIONAL).Int(512);
        this->Attr("dense_tail").AttrType(OPTIONAL).Int(128);

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(TrianglePagedSparseAttention);
}  // namespace ops
