// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vllm-ascend project

#include "register/op_def_registry.h"

namespace ops {
class RearrangeQkvDma : public OpDef {
public:
    explicit RearrangeQkvDma(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("qDim").Int();
        this->Attr("kDim").Int();
        this->Attr("vDim").Int();

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(RearrangeQkvDma);
}  // namespace ops
