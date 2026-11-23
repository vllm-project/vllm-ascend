#include "register/op_def_registry.h"

namespace ops {
class SolAttnPrepareDiagThreshKernel : public OpDef {
public:
    explicit SolAttnPrepareDiagThreshKernel(const char* name) : OpDef(name)
    {
        this->Input("q_bar_f32")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("kc_pad_f32")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("thresh_bh")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("B")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("H")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("NT")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("NG")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("NPAD")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("scale_log2")
            .AttrType(REQUIRED)
            .Float(0);
        this->Attr("tau")
            .AttrType(REQUIRED)
            .Float(0);
        this->Attr("inv_nt")
            .AttrType(REQUIRED)
            .Float(0);
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true).DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};

OP_ADD(SolAttnPrepareDiagThreshKernel);
}
