#include <register/op_def_registry.h>

namespace ops {

class GroupedMatmulSituQuantMsd : public OpDef {
public:
    explicit GroupedMatmulSituQuantMsd(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_INT4, ge::DT_INT4})
            .Format({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ});
        this->Input("weight_scale")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_UINT64, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("group_list")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight_assist_matrix")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("smooth_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("group_list_type").AttrType(OPTIONAL).Int(0);
        this->Attr("beta").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("linear_beta").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("dequant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("dequant_dtype").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_dtype").AttrType(OPTIONAL).Int(0);
        this->Attr("transpose_weight").AttrType(OPTIONAL).Bool(0);
        this->Attr("tuning_config").AttrType(OPTIONAL).ListInt({0});
        this->Attr("activate_left").AttrType(OPTIONAL).Bool(true);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "grouped_matmul_situ_quant_msd");
        this->AICore().AddConfig("ascend910_93", config);
    }
};

OP_ADD(GroupedMatmulSituQuantMsd);

} // namespace ops
