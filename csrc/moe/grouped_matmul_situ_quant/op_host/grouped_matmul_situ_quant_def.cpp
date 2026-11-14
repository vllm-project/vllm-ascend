#include <register/op_def_registry.h>

namespace ops {

/*
 * GroupedMatmulSituQuant 覆盖两条通路，共用同一套输入/属性原型（同一个 aclnn 符号）：
 *   - Ascend950（arch35）：MX A8W4，x=MXFP8、weight=MXFP4（NZ）、scale=E8M0，
 *     kernel 入口 opFile=grouped_matmul_situ_quant（现有 A5 实现，本文件只补槽位不动其语义）；
 *   - Atlas A2/A3（arch32）：A8W4 MSD，x=INT8、weight=INT4、weight_scale=UINT64、y=INT8，
 *     kernel 入口 opFile=grouped_matmul_situ_quant_apt。
 *
 * 输入槽位顺序：A5 原有的 5 个槽位（x/x_scale/weight/weight_scale/group_list）位置保持不变，
 * A2A3 需要的 weight_assist_matrix / bias / smooth_scale 追加在尾部。
 * A5 不使用后三个槽位：bias / smooth_scale 传 nullptr；weight_assist_matrix 是 DYNAMIC
 * （TensorList），GE 侧 GenKey 不接受空列表，由 op_api 层的兼容入口代传一份占位。
 */
class GroupedMatmulSituQuant : public OpDef {
public:
    explicit GroupedMatmulSituQuant(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // Weights arrive as transposed views of NZ-format storage (loader does
        // npu_format_cast(...).transpose(1,2)); NZ tensors cannot be made
        // contiguous (.contiguous() rejects internal format), and the kernel
        // already reads them via format metadata — accept them as-is.
        // K3 A5 MXFP4 uses C0=16; both stacked and TensorList weights
        // use this one compiled format variant.
        this->Input("weight")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT4_E2M1})
            .Format({ge::FORMAT_FRACTAL_NZ_C0_16})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ_C0_16})
            .IgnoreContiguous();
        this->Input("weight_scale")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("group_list")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // A2/A3 的离线补偿矩阵（8*sum(w)），A5 通路不读取。
        this->Input("weight_assist_matrix")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("smooth_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // A5 原有属性保持在最前，A2A3 扩展属性追加在尾部（tiling 侧按位置索引取用）。
        this->Attr("group_list_type").AttrType(OPTIONAL).Int(0);
        this->Attr("beta").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("linear_beta").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("dequant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("dequant_dtype").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_dtype").AttrType(OPTIONAL).Int(0);
        this->Attr("transpose_weight").AttrType(OPTIONAL).Bool(0);
        this->Attr("tuning_config").AttrType(OPTIONAL).ListInt({0});
        // SiTU 的 activate_left 恒为 true（gate 取前半），A5 侧不读取该属性。
        this->Attr("activate_left").AttrType(OPTIONAL).Bool(true);

        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "grouped_matmul_situ_quant");

        // ---- Atlas A2 / A3：A8W4 MSD ----
        OpAICoreConfig a8w4MsdCfg;
        a8w4MsdCfg.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        a8w4MsdCfg.Input("x_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // A8W4 的 weight 是 INT4：ND（[E, N, K]）与 FRACTAL_NZ 两种排布。
        a8w4MsdCfg.Input("weight")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_INT4, ge::DT_INT4})
            .Format({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ});
        // A8W4 MSD 的 weight_scale 为打包后的 UINT64（每 4 个 int4 共享一组 scale）。
        a8w4MsdCfg.Input("weight_scale")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_UINT64, ge::DT_UINT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        a8w4MsdCfg.Input("group_list")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        a8w4MsdCfg.Input("weight_assist_matrix")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        a8w4MsdCfg.Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        a8w4MsdCfg.Input("smooth_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        a8w4MsdCfg.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        a8w4MsdCfg.Output("y_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        a8w4MsdCfg.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "grouped_matmul_situ_quant_apt");
        this->AICore().AddConfig("ascend910_93", a8w4MsdCfg);

        this->AICore().AddConfig("ascend950", regbaseCfg);
    }
};

OP_ADD(GroupedMatmulSituQuant);

} // namespace ops
