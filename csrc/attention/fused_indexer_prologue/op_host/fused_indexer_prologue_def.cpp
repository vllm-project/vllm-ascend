/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "register/op_def_registry.h"

namespace ops {
class FusedIndexerPrologue : public OpDef {
public:
    explicit FusedIndexerPrologue(const char *name) : OpDef(name)
    {
        this->Input("latent").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("wk_weight").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("norm_weight").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("cos").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("sin").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("slot_mapping").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        // Cache tensors deliberately preserve their physical block/page
        // strides instead of being materialized as contiguous temporaries.
        this->Input("k_cache").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT8}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("scale_cache").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();

        this->Output("k_cache_out").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT8}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Output("scale_cache_out").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();

        this->Attr("epsilon").AttrType(REQUIRED).Float(1e-6F);
        this->Attr("cache_block_size").AttrType(REQUIRED).Int(0);
        this->Attr("k_cache_stride0").AttrType(REQUIRED).Int(0);
        this->Attr("k_cache_stride1").AttrType(REQUIRED).Int(0);
        this->Attr("scale_cache_stride0").AttrType(REQUIRED).Int(0);
        this->Attr("scale_cache_stride1").AttrType(REQUIRED).Int(0);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
    }
};

OP_ADD(FusedIndexerPrologue);
}  // namespace ops
