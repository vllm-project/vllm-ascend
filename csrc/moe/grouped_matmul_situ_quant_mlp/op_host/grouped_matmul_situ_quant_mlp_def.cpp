#include <register/op_def_registry.h>

namespace ops {

class GroupedMatmulSituQuantMlp : public OpDef {
public:
    explicit GroupedMatmulSituQuantMlp(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});
        this->Input("x_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("group_list").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("up_weight").ParamType(REQUIRED).DataType({ge::DT_INT4}).Format({ge::FORMAT_FRACTAL_NZ});
        this->Input("up_scale").ParamType(REQUIRED).DataType({ge::DT_UINT64}).Format({ge::FORMAT_ND});
        this->Input("up_compensation").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("down_weight").ParamType(REQUIRED).DataType({ge::DT_INT4}).Format({ge::FORMAT_FRACTAL_NZ});
        this->Input("down_scale").ParamType(REQUIRED).DataType({ge::DT_UINT64}).Format({ge::FORMAT_ND});
        this->Input("down_compensation").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Attr("group_list_type").AttrType(OPTIONAL).Int(0);
        this->Attr("beta").AttrType(OPTIONAL).Float(4.0f);
        this->Attr("linear_beta").AttrType(OPTIONAL).Float(25.0f);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
    }
};

OP_ADD(GroupedMatmulSituQuantMlp);

} // namespace ops
