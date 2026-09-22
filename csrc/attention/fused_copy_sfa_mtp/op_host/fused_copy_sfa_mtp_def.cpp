#include "register/op_def_registry.h"

namespace ops {

class FusedCopySfaMtp : public OpDef {
public:
    explicit FusedCopySfaMtp(const char *name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("value").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("sparse_indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("cache_tokens").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("hbm_block_table").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("actual_seq_lengths_query").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("actual_seq_lengths_kv").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("query_rope").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("hbm_key_rope").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dram_key_rope").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dram_kv_cache").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dram_block_table").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("topk_source_ids").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("topk_miss_counts").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("miss_src_ids").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("miss_dst_slots").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("miss_counts").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();

        this->Output("attention_out").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});

        this->Attr("scale_value").AttrType(REQUIRED).Float(1.0);
        this->Attr("sparse_block_size").AttrType(REQUIRED).Int(1);
        this->Attr("layout_query").AttrType(OPTIONAL).String("TND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("PA_BSND");
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend910b", config);
    }
};

OP_ADD(FusedCopySfaMtp);

}  // namespace ops
