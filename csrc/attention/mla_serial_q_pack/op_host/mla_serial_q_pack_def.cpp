// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class MlaSerialQPack : public OpDef {
public:
    explicit MlaSerialQPack(const char* name) : OpDef(name)
    {
        this->Input("up").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("packed").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("heads").AttrType(REQUIRED).Int();
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(MlaSerialQPack);
}
