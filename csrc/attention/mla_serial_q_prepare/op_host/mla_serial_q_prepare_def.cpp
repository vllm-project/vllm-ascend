// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class MlaSerialQPrepare : public OpDef {
public:
    explicit MlaSerialQPrepare(const char* name) : OpDef(name)
    {
        this->Input("up").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cos").ParamType(OPTIONAL).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sin").ParamType(OPTIONAL).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("packed").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("q_rope").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("heads").AttrType(REQUIRED).Int();
        this->Attr("use_rope").AttrType(REQUIRED).Int();
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(MlaSerialQPrepare);
}
