#include "register/op_def_registry.h"

namespace ops {

class FirstFillScatterCopy : public OpDef {
public:
    explicit FirstFillScatterCopy(const char *name) : OpDef(name)
    {
        this->Input("hbm_k_rope").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("hbm_kv_cache").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dram_k_rope").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dram_kv_cache").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("hbm_block_table").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dram_block_table").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("miss_src_ids").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("miss_dst_slots").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("miss_counts").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("num_cache_tokens").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();

        this->Output("hbm_k_rope").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
        this->Output("hbm_kv_cache").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend910b", config);
    }
};

OP_ADD(FirstFillScatterCopy);

}  // namespace ops
