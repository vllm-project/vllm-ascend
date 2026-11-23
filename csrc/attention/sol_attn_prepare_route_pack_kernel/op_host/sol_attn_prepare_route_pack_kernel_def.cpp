#include "register/op_def_registry.h"

namespace ops {
class SolAttnPrepareRoutePackKernel : public OpDef {
public:
    explicit SolAttnPrepareRoutePackKernel(const char* name) : OpDef(name)
    {
        this->Input("raw_scores")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("thresh_bh")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("group_approx_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("flat_exact_cnt")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("flat_exact_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("HC")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("NT")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("NT_Q")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("NG")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("NPAD")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("NT_Q_PAD8")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("TILE_M")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("sink_s0")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("sink_s1")
            .AttrType(REQUIRED)
            .Int(0);
        this->Attr("scale_log2")
            .AttrType(REQUIRED)
            .Float(0);
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true).DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};

OP_ADD(SolAttnPrepareRoutePackKernel);
}
