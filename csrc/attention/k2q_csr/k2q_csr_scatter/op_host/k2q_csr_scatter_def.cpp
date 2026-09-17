/**
 * K2qCsrScatter op definition.
 * Inputs: q2k, cu_seqlens, scratch | Outputs: q_ind, slot
 * Attrs: total_rows, max_kv, use_simt, q_global_offset
 */
#include "register/op_def.h"
#include "register/op_def_registry.h"

namespace ops {
class K2qCsrScatter : public OpDef {
public:
    explicit K2qCsrScatter(const char *name) : OpDef(name)
    {
        this->Input("q2k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("scratch")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("q_ind")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("slot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("total_rows").AttrType(REQUIRED).Int(0);
        this->Attr("max_kv").AttrType(REQUIRED).Int(0);
        this->Attr("use_simt").AttrType(REQUIRED).Int(0);
        this->Attr("q_global_offset").AttrType(REQUIRED).Int(0);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "k2q_csr_scatter");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);

        OpAICoreConfig a5Config;
        a5Config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "k2q_csr_scatter_apt");
        this->AICore().AddConfig("ascend950", a5Config);
    }
};
OP_ADD(K2qCsrScatter);
} // namespace ops
