// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class VllmMhcExpand : public OpDef {
public:
    explicit VllmMhcExpand(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("mult").AttrType(REQUIRED).Int();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(VllmMhcExpand);
}  // namespace ops
