#include "register/op_def_registry.h"

namespace ops {
class SolAttnShortFwdM128Kernel : public OpDef {
public:
    explicit SolAttnShortFwdM128Kernel(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("lse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("B")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("T")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("H")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("NT")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("NT_Q")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("NT_Q_PAD8")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("T_LSE")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("sink_start")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("sink_end")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("scale_log2")
            .AttrType(REQUIRED)
            .Float(0);
        this->Attr("scale_sq")
            .AttrType(REQUIRED)
            .Float(0);
        this->Attr("tau")
            .AttrType(REQUIRED)
            .Float(0);
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true).DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};

OP_ADD(SolAttnShortFwdM128Kernel);
}
