/**
 * K2qCsrRowPrefix op definition.
 * Inputs: scratch | Outputs: row_ptr
 * Attrs: total_rows, max_kv, use_simt, num_heads, num_tokens, topk, batch
 */
#include "register/op_def.h"
#include "register/op_def_registry.h"

namespace ops {
class K2qCsrRowPrefix : public OpDef {
public:
    explicit K2qCsrRowPrefix(const char *name) : OpDef(name)
    {
        this->Input("scratch")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("row_ptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("total_rows").AttrType(REQUIRED).Int(0);
        this->Attr("max_kv").AttrType(REQUIRED).Int(0);
        this->Attr("use_simt").AttrType(REQUIRED).Int(0);
        this->Attr("num_heads").AttrType(REQUIRED).Int(0);
        this->Attr("num_tokens").AttrType(REQUIRED).Int(0);
        this->Attr("topk").AttrType(REQUIRED).Int(0);
        this->Attr("batch").AttrType(REQUIRED).Int(0);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "k2q_csr_row_prefix");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);

        OpAICoreConfig a5Config;
        a5Config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "k2q_csr_row_prefix_apt");
        this->AICore().AddConfig("ascend950", a5Config);
    }
};
OP_ADD(K2qCsrRowPrefix);
} // namespace ops
